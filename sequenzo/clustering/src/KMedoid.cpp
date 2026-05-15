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

    vector<int>    tclusterid;
    vector<int>    saved;
    vector<int>    clusterMembership;
    vector<int>    clusterSize;

    py::array_t<double> diss;
    py::array_t<int>    centroids;
    py::array_t<double> weights;

    const double* diss_ptr;
    const double* cond_ptr;
    const double* wt_ptr;
    bool use_condensed;

    inline double get_dist(int i, int j) const {
        if (!use_condensed)
            return diss_ptr[static_cast<size_t>(i) * nelements + j];
        if (i == j) return 0.0;
        // scipy condensed format: row-major upper triangle
        // index for pair (a, b) with a < b: a*(2n-a-1)/2 + (b-a-1)
        int a = i, b = j;
        if (a > b) { int t = a; a = b; b = t; }
        return cond_ptr[static_cast<size_t>(a) * (2 * nelements - a - 1) / 2 + (b - a - 1)];
    }

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

        wt_ptr = weights.data();

        if (diss.ndim() == 1) {
            use_condensed = true;
            cond_ptr = diss.data();
            diss_ptr = nullptr;
        } else {
            use_condensed = false;
            diss_ptr = diss.data();
            cond_ptr = nullptr;
        }
    }

    void buildInitialCentroids() {
        int* cent_ptr = centroids.mutable_data();
        const int n = nelements;

        double build_maxdist = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(max:build_maxdist) schedule(static)
#endif
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double v = get_dist(i, j);
                if (v > build_maxdist) build_maxdist = v;
            }
        build_maxdist = 1.1 * build_maxdist + 1.0;

        vector<int>    is_medoid(n, 0);
        vector<double> build_dysma(n, build_maxdist);

        for (int kk = 0; kk < nclusters; ++kk) {
            double ammax = 0.0;
            int    nmax  = -1;

#ifdef _OPENMP
            #pragma omp parallel
            {
                double local_ammax = 0.0;
                int    local_nmax  = -1;

                #pragma omp for schedule(static) nowait
                for (int i = 0; i < n; ++i) {
                    if (is_medoid[i]) continue;
                    double beter = 0.0;
                    for (int j = 0; j < n; ++j) {
                        double diff = build_dysma[j] - get_dist(i, j);
                        beter += wt_ptr[j] * fmax(0.0, diff);
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
            for (int i = 0; i < n; ++i) {
                if (is_medoid[i]) continue;
                double beter = 0.0;
                for (int j = 0; j < n; ++j) {
                    double diff = build_dysma[j] - get_dist(i, j);
                    beter += wt_ptr[j] * fmax(0.0, diff);
                }
                if (ammax <= beter) { ammax = beter; nmax = i; }
            }
#endif

            is_medoid[nmax] = 1;
            cent_ptr[kk] = nmax;

            for (int j = 0; j < n; ++j) {
                double d = get_dist(nmax, j);
                if (d < build_dysma[j])
                    build_dysma[j] = d;
            }
        }
    }

    void getrandommedoids() {
        int* cent_ptr = centroids.mutable_data();
        const int n = nelements;

        static thread_local mt19937 rng(random_device{}());

        vector<int> indices(n);
        iota(indices.begin(), indices.end(), 0);

        while (true) {
            for (int i = 0; i < nclusters; ++i) {
                uniform_int_distribution<int> d(i, n - 1);
                int j = d(rng);
                swap(indices[i], indices[j]);
            }

            bool valid = true;
            for (int a = 0; a < nclusters && valid; ++a)
                for (int b = a + 1; b < nclusters && valid; ++b)
                    if (get_dist(indices[a], indices[b]) <= 0.0)
                        valid = false;

            if (valid) {
                for (int i = 0; i < nclusters; ++i)
                    cent_ptr[i] = indices[i];
                return;
            }
        }
    }

    void getclustermedoids() {
        int* cent_ptr = centroids.mutable_data();
        const int n = nelements;

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int k = 0; k < nclusters; ++k) {
            int    size   = clusterSize[k];
            double best   = DBL_MAX;
            int    bestID = clusterMembership[k * n];
            const int* members = &clusterMembership[k * n];

            for (int i = 0; i < size; ++i) {
                int    ii     = members[i];
                double current = 0.0;

                for (int j = 0; j < size; ++j) {
                    if (i == j) continue;
                    int jj = members[j];
                    current += wt_ptr[jj] * get_dist(ii, jj);
                    if (current >= best) break;
                }

                if (current < best) {
                    best   = current;
                    bestID = ii;
                }
            }
            cent_ptr[k] = bestID;
        }
    }

    double runInnerPass() {
        const int n = nelements;

        vector<int> local_assign(n);

        double total   = DBL_MAX;
        int    counter = 0;
        int    period  = 10;

        while (true) {
            PyErr_CheckSignals();

            double prev = total;
            total = 0.0;

            if (counter > 0) getclustermedoids();

            const int* cent_ptr = centroids.data();

            if (counter % period == 0) {
                for (int i = 0; i < n; ++i) saved[i] = tclusterid[i];
                if (period < INT_MAX / 2) period *= 2;
            }
            ++counter;

#ifdef _OPENMP
            #pragma omp parallel for schedule(static) reduction(+:total)
#endif
            for (int i = 0; i < n; ++i) {
                double dist_min = DBL_MAX;
                int    assign   = 0;
                for (int k = 0; k < nclusters; ++k) {
                    double td = get_dist(i, cent_ptr[k]);
                    if (td < dist_min) { dist_min = td; assign = k; }
                }
                local_assign[i] = assign;
                total += wt_ptr[i] * dist_min;
            }

            for (int k = 0; k < nclusters; ++k) clusterSize[k] = 0;
            for (int i = 0; i < n; ++i) {
                int a = local_assign[i];
                tclusterid[i] = a;
                clusterMembership[a * n + clusterSize[a]] = i;
                ++clusterSize[a];
            }

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
            for (int i = 0; i < n; ++i) {
                if (saved[i] != tclusterid[i]) { same = false; break; }
            }
            if (same) break;
        }

        return total;
    }

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
                const int* cent = centroids.data();
                for (int i = 0; i < nelements; ++i)
                    best_tclusterid[i] = tclusterid[i];
                for (int k = 0; k < nclusters; ++k)
                    best_centroids_vec[k] = cent[k];
            }
        }

        for (int i = 0; i < nelements; ++i)
            tclusterid[i] = best_tclusterid[i];
        int* cent = centroids.mutable_data();
        for (int k = 0; k < nclusters; ++k)
            cent[k] = best_centroids_vec[k];

        return getResultArray();
    }

    py::array_t<int> getResultArray() const {
        py::array_t<int> result(nelements);
        int* res_ptr = result.mutable_data();
        const int* cent = centroids.data();
        for (int i = 0; i < nelements; ++i)
            res_ptr[i] = cent[tclusterid[i]];
        return result;
    }
};
