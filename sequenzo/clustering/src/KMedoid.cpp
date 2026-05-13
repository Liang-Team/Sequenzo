#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <cfloat>
#include <climits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
namespace py = pybind11;

class KMedoid {
protected:
    int nelements;
    int nclusters;
    int npass;

    vector<int>    tclusterid;        // current cluster assignment (0-based cluster index)
    vector<int>    saved;             // checkpoint assignments for convergence detection
    vector<int>    clusterMembership; // flattened [nclusters x nelements] membership list
    vector<int>    clusterSize;       // current size of each cluster

    py::array_t<double> diss;
    py::array_t<int>    centroids;
    py::array_t<double> weights;

public:
    KMedoid(int nelements, py::array_t<double> diss,
            py::array_t<int> centroids, int npass,
            py::array_t<double> weights)
        : nelements(nelements),
          nclusters(static_cast<int>(centroids.size())),
          npass(npass),
          diss(diss), centroids(centroids), weights(weights)
    {
        tclusterid.resize(nelements);
        saved.resize(nelements);
        clusterMembership.resize(nelements * nclusters);
        clusterSize.resize(nclusters, 0);
    }

    // -----------------------------------------------------------------------
    // PAM BUILD initialisation (deterministic greedy selection).
    // Mirrors WeightedCluster's buildInitialCentroids(); same parallel pattern
    // as sequenzo/clustering/src/PAM.cpp::buildInitialCentroids().
    // -----------------------------------------------------------------------
    void buildInitialCentroids() {
        auto ptr_diss      = diss.unchecked<2>();
        auto ptr_weights   = weights.unchecked<1>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        double build_maxdist = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(max:build_maxdist) schedule(static)
#endif
        for (int i = 0; i < nelements; ++i)
            for (int j = i + 1; j < nelements; ++j) {
                double v = ptr_diss(i, j);
                if (v > build_maxdist) build_maxdist = v;
            }
        build_maxdist = 1.1 * build_maxdist + 1.0;

        vector<int>    is_medoid(nelements, 0);
        vector<double> build_dysma(nelements, build_maxdist);

        for (int kk = 0; kk < nclusters; ++kk) {
            double ammax = 0.0;
            int    nmax  = -1;

#ifdef _OPENMP
            #pragma omp parallel
            {
                double local_ammax = 0.0;
                int    local_nmax  = -1;

                #pragma omp for schedule(static) nowait
                for (int i = 0; i < nelements; ++i) {
                    if (is_medoid[i]) continue;
                    double beter = 0.0;
                    for (int j = 0; j < nelements; ++j) {
                        beter += ptr_weights[j] * fmax(0.0, build_dysma[j] - ptr_diss(i, j));
                    }
                    if (local_ammax <= beter) { local_ammax = beter; local_nmax = i; }
                }

                #pragma omp critical
                {
                    if (local_nmax != -1 &&
                        (local_ammax > ammax ||
                         (local_ammax == ammax && local_nmax > nmax))) {
                        ammax = local_ammax;
                        nmax  = local_nmax;
                    }
                }
            }
#else
            for (int i = 0; i < nelements; ++i) {
                if (is_medoid[i]) continue;
                double beter = 0.0;
                for (int j = 0; j < nelements; ++j) {
                    beter += ptr_weights[j] * fmax(0.0, build_dysma[j] - ptr_diss(i, j));
                }
                if (ammax <= beter) { ammax = beter; nmax = i; }
            }
#endif

            is_medoid[nmax] = 1;
            ptr_centroids[kk] = nmax;

            for (int j = 0; j < nelements; ++j)
                if (ptr_diss(nmax, j) < build_dysma[j])
                    build_dysma[j] = ptr_diss(nmax, j);
        }
    }

    // -----------------------------------------------------------------------
    // Random medoid selection for restart passes (ipass > 0).
    // Fisher-Yates partial shuffle; re-draws if any pair has distance == 0.
    // -----------------------------------------------------------------------
    void getrandommedoids() {
        auto ptr_diss      = diss.unchecked<2>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        static thread_local mt19937 rng(random_device{}());

        vector<int> indices(nelements);
        iota(indices.begin(), indices.end(), 0);

        while (true) {
            for (int i = 0; i < nclusters; ++i) {
                uniform_int_distribution<int> d(i, nelements - 1);
                int j = d(rng);
                swap(indices[i], indices[j]);
            }

            bool valid = true;
            for (int a = 0; a < nclusters && valid; ++a)
                for (int b = a + 1; b < nclusters && valid; ++b)
                    if (ptr_diss(indices[a], indices[b]) <= 0.0)
                        valid = false;

            if (valid) {
                for (int i = 0; i < nclusters; ++i)
                    ptr_centroids[i] = indices[i];
                return;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Update medoids: for each cluster pick the member that minimises total
    // weighted distance to all other cluster members.
    // Outer k loop is parallelised with OpenMP (clusters are independent).
    // -----------------------------------------------------------------------
    void getclustermedoids() {
        auto ptr_weights   = weights.unchecked<1>();
        auto ptr_diss      = diss.unchecked<2>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int k = 0; k < nclusters; ++k) {
            int    size   = clusterSize[k];
            double best   = DBL_MAX;
            int    bestID = clusterMembership[k * nelements];

            for (int i = 0; i < size; ++i) {
                int    ii      = clusterMembership[k * nelements + i];
                double current = 0.0;

                for (int j = 0; j < size; ++j) {
                    if (i == j) continue;
                    int jj = clusterMembership[k * nelements + j];
                    current += ptr_weights[jj] * ptr_diss(ii, jj);
                    if (current >= best) break;   // early-stop heuristic
                }

                if (current < best) {
                    best   = current;
                    bestID = ii;
                }
            }
            ptr_centroids[k] = bestID;
        }
    }

    // -----------------------------------------------------------------------
    // One inner convergence pass (assign → update medoids → repeat until
    // convergence or no cost improvement).
    //
    // Assignment uses a two-phase approach to admit OpenMP:
    //   Phase 1 (parallel):   compute closest medoid + distance for every i.
    //   Phase 2 (sequential): fill clusterMembership in order 0..n-1.
    // -----------------------------------------------------------------------
    double runInnerPass() {
        auto ptr_weights   = weights.unchecked<1>();
        auto ptr_diss      = diss.unchecked<2>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        vector<int> local_assign(nelements);

        double total   = DBL_MAX;
        int    counter = 0;
        int    period  = 10;

        while (true) {
            PyErr_CheckSignals();

            double prev = total;
            total = 0.0;

            if (counter > 0) getclustermedoids();

            if (counter % period == 0) {
                for (int i = 0; i < nelements; ++i) saved[i] = tclusterid[i];
                if (period < INT_MAX / 2) period *= 2;
            }
            ++counter;

            // --- Phase 1: parallel closest-medoid computation + total reduction ---
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) reduction(+:total)
#endif
            for (int i = 0; i < nelements; ++i) {
                double dist_min = DBL_MAX;
                int    assign   = 0;
                for (int k = 0; k < nclusters; ++k) {
                    double td = ptr_diss(i, ptr_centroids[k]);
                    if (td < dist_min) { dist_min = td; assign = k; }
                }
                local_assign[i] = assign;
                total += ptr_weights[i] * dist_min;
            }

            // --- Phase 2: sequential membership fill (deterministic order) --
            for (int k = 0; k < nclusters; ++k) clusterSize[k] = 0;
            for (int i = 0; i < nelements; ++i) {
                int a = local_assign[i];
                tclusterid[i] = a;
                clusterMembership[a * nelements + clusterSize[a]] = i;
                ++clusterSize[a];
            }

            // If any cluster is empty, reinitialise from scratch and restart.
            bool empty = false;
            for (int k = 0; k < nclusters; ++k) {
                if (clusterSize[k] == 0) {
                    buildInitialCentroids();
                    counter = 0;
                    period  = 10;
                    empty   = true;
                    break;
                }
            }
            if (empty) continue;

            if (total >= prev) break;

            bool same = true;
            for (int i = 0; i < nelements; ++i) {
                if (saved[i] != tclusterid[i]) { same = false; break; }
            }
            if (same) break;
        }

        return total;
    }

    // -----------------------------------------------------------------------
    // Outer restart loop:
    //   ipass == 0             : PAM BUILD (deterministic, matches WeightedCluster)
    //   ipass == 1 .. npass-1  : random medoids
    //   npass == 0             : use caller-supplied centroids, run once
    //
    // The best solution (lowest total weighted distance) is kept.
    // -----------------------------------------------------------------------
    py::array_t<int> runclusterloop() {
        vector<int> best_tclusterid(nelements, -1);
        vector<int> best_centroids_vec(nclusters, 0);
        double      best_total = DBL_MAX;
        bool        best_found = false;

        int niter = (npass > 0) ? npass : 1;

        for (int ipass = 0; ipass < niter; ++ipass) {
            if (npass > 0) {
                if (ipass == 0)
                    buildInitialCentroids();
                else
                    getrandommedoids();
            }
            // npass == 0: use caller-supplied centroids unchanged.

            double total = runInnerPass();

            bool is_new_best = false;
            if (!best_found) {
                is_new_best = true;
            } else {
                bool differs = false;
                for (int i = 0; i < nelements; ++i)
                    if (best_tclusterid[i] != tclusterid[i]) { differs = true; break; }
                if (differs && total < best_total) is_new_best = true;
            }

            if (is_new_best) {
                best_total = total;
                best_found = true;
                for (int i = 0; i < nelements; ++i)
                    best_tclusterid[i] = tclusterid[i];
                auto ptr_c = centroids.unchecked<1>();
                for (int k = 0; k < nclusters; ++k)
                    best_centroids_vec[k] = ptr_c[k];
            }
        }

        for (int i = 0; i < nelements; ++i)
            tclusterid[i] = best_tclusterid[i];
        auto ptr_c = centroids.mutable_unchecked<1>();
        for (int k = 0; k < nclusters; ++k)
            ptr_c[k] = best_centroids_vec[k];

        return getResultArray();
    }

    py::array_t<int> getResultArray() const {
        py::array_t<int> result(nelements);
        auto results  = result.mutable_unchecked<1>();
        auto centroid = centroids.unchecked<1>();
        for (int i = 0; i < nelements; ++i)
            results(i) = centroid(tclusterid[i]);
        return result;
    }
};
