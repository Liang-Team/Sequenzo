#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cfloat>
#include <climits>
#include <cmath>
#define WEIGHTED_CLUST_TOL -1e-10
using namespace std;
namespace py = pybind11;

class PAM {
public:
    PAM(int nelements, py::array_t<double> diss,
        py::array_t<int> centroids, int npass, py::array_t<double> weights) {
        try {
            this->nelements = nelements;
            this->centroids = centroids;
            this->npass = npass;
            this->weights = weights;
            this->diss = diss;
            this->maxdist = 0.0;
            this->nclusters = static_cast<int>(centroids.size());
            this->tclusterid.resize(nelements);
            this->computeMaxDist();

            dysma.resize(nelements, maxdist);
            dysmb.resize(nelements, maxdist);
        } catch (const exception &e) {
            py::print("Error: ", e.what());
        }
    }

    // Computes the maximum distance between any two elements in the distance matrix.
    void computeMaxDist() {
        auto ptr_diss = diss.unchecked<2>();

        int nthreads = 1;
#ifdef _OPENMP
        #pragma omp parallel
        {
            #pragma omp single
            nthreads = omp_get_num_threads();
        }
#endif

        std::vector<double> thread_max(nthreads, 0.0);

#ifdef _OPENMP
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
#else
        {
            int tid = 0;
#endif
            double local = 0.0;

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < nelements; ++i) {
                for (int j = i + 1; j < nelements; ++j) {
                    double val = ptr_diss(i, j);
                    if (val > local) local = val;
                }
            }

            thread_max[tid] = local;
        }

        double max_val = 0.0;
        for (double val : thread_max) {
            if (val > max_val) max_val = val;
        }

        maxdist = 1.1 * max_val + 1.0;
    }

    // -----------------------------------------------------------------------
    // PAM BUILD: deterministic greedy initialisation (mirrors WeightedCluster).
    // Parallelised with OpenMP over the candidate-medoid loop (outer i-loop).
    // Tie-breaking: last (highest-indexed) element wins, matching the
    // sequential `ammax <= beter` condition — preserved by the critical-section
    // merge rule "equal score → higher index wins".
    // -----------------------------------------------------------------------
    void buildInitialCentroids() {
        auto ptr_diss      = diss.unchecked<2>();
        auto ptr_weights   = weights.unchecked<1>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        // Reuse maxdist already computed in the constructor — avoids O(n²) recomputation.
        double build_maxdist = maxdist;

        vector<int>    is_medoid(nelements, 0);
        vector<double> build_dysma(nelements, build_maxdist);

        for (int kk = 0; kk < nclusters; ++kk) {
            double ammax = 0.0;
            int    nmax  = -1;

            // Parallel search for the best candidate (max weighted gain).
            // Each thread keeps a local best; the critical-section merge
            // preserves "last tied index wins" by preferring higher nmax.
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
                    // <= matches WeightedCluster (last tied element wins within thread range)
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

    // Runs the PAM SWAP loop until no improving swap exists (dzsky >= 0).
    py::array_t<int> runclusterloop() {
        auto ptr_weights   = weights.unchecked<1>();
        auto ptr_diss      = diss.unchecked<2>();
        auto ptr_centroids = centroids.mutable_unchecked<1>();

        // When npass > 0 always use PAM BUILD for initialisation (deterministic),
        // matching R's wcKMedoids default behaviour.
        if (npass > 0) {
            buildInitialCentroids();
        }

        double dzsky;
        int hbest = -1;
        int nbest = -1;
        int k;
        double total = -1.0;
        int nclusters = static_cast<int>(centroids.size());

        do {
            // Update dysma (nearest medoid distance) and dysmb (2nd nearest)
            // for every point, and record cluster assignment.
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nelements; i++) {
                dysmb[i] = maxdist;
                dysma[i] = maxdist;

                for (int k = 0; k < nclusters; k++) {
                    int icluster = ptr_centroids(k);
                    double dist = ptr_diss(i, icluster);

                    if (dysma[i] > dist) {
                        dysmb[i] = dysma[i];
                        dysma[i] = dist;
                        tclusterid[i] = k;
                    } else if (dysmb[i] > dist) {
                        dysmb[i] = dist;
                    }
                }
            }

            if (total < 0) {
                total = 0;
                #pragma omp parallel for reduction(+:total) schedule(static)
                for (int i = 0; i < nelements; i++) {
                    total += ptr_weights[i] * dysma[i];
                }
            }

            dzsky = 1;

            // Precompute medoid lookup to avoid O(k) scan per element.
            vector<bool> is_medoid_flag(nelements, false);
            for (int kk = 0; kk < nclusters; kk++) is_medoid_flag[ptr_centroids[kk]] = true;

            // Find the best non-medoid h to swap with one of the current medoids.
            // schedule(static): only k elements are skipped out of n, so load is ~uniform.
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < nelements; h++) {
                if (is_medoid_flag[h])
                    continue;

                double local_dzsky = dzsky;
                int local_hbest = -1;
                int local_nbest = -1;

                for (int k = 0; k < nclusters; k++) {
                    int i = ptr_centroids[k];
                    double dz = 0.0;

                    for (int j = 0; j < nelements; j++) {
                        if (ptr_diss(i, j) == dysma[j]) {
                            double small = (dysmb[j] > ptr_diss(h, j)) ? ptr_diss(h, j) : dysmb[j];
                            dz += ptr_weights[j] * (-dysma[j] + small);
                        } else if (ptr_diss(h, j) < dysma[j]) {
                            dz += ptr_weights[j] * (-dysma[j] + ptr_diss(h, j));
                        }
                    }

                    if (dz < local_dzsky) {
                        local_dzsky = dz;
                        local_hbest = h;
                        local_nbest = i;
                    }
                }

                #pragma omp critical
                {
                    if (local_dzsky < dzsky) {
                        dzsky = local_dzsky;
                        hbest = local_hbest;
                        nbest = local_nbest;
                    }
                }
            }

            if (dzsky < 0) {
                for (k = 0; k < nclusters; k++) {
                    if (ptr_centroids[k] == nbest) {
                        ptr_centroids[k] = hbest;
                    }
                }
                total += dzsky;
            }

        } while (dzsky < 0);

        return getResultArray();
    }

    py::array_t<int> getResultArray() const {
        py::array_t<int> result(nelements);
        auto results  = result.mutable_unchecked<1>();
        auto centroid = centroids.unchecked<1>();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nelements; ++i) {
            results(i) = centroid(tclusterid[i]);
        }

        return result;
    }


protected:
    int nelements;
    py::array_t<double> diss;
    py::array_t<int>    centroids;
    int npass;
    py::array_t<double> weights;
    vector<int>    tclusterid;
    vector<double> dysmb;
    int nclusters;
    double maxdist;
    vector<double> dysma;
};
