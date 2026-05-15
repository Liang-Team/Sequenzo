#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#define WEIGHTED_CLUST_TOL -1e-10
#include <cfloat>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

class PAMonce {
public:
    PAMonce(int nelement, py::array_t<double> diss, py::array_t<int> centroids, int npass, py::array_t<double> weights){
        try {
            this->nelement = nelement;
            this->diss = diss;
            this->centroids = centroids;
            this->npass = npass;
            this->weights = weights;

            this->wt_ptr = weights.data();

            if (diss.ndim() == 1) {
                use_condensed = true;
                cond_ptr = diss.data();
                diss_ptr = nullptr;
            } else {
                use_condensed = false;
                diss_ptr = diss.data();
                cond_ptr = nullptr;
            }

            clusterid = py::array_t<int>(nelement);
            tclusterid.resize(nelement, -1);

            maxdist = computeMaxDist();
            dysma.resize(nelement, maxdist);
            dysmb.resize(nelement, maxdist);

            nclusters = centroids.size();
        } catch (const std::exception& e){
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    inline double get_dist(int i, int j) const {
        if (!use_condensed)
            return diss_ptr[static_cast<size_t>(i) * nelement + j];
        if (i == j) return 0.0;
        // scipy condensed format: row-major upper triangle
        // index for pair (a, b) with a < b: a*(2n-a-1)/2 + (b-a-1)
        int a = i, b = j;
        if (a > b) { int t = a; a = b; b = t; }
        return cond_ptr[static_cast<size_t>(a) * (2 * nelement - a - 1) / 2 + (b - a - 1)];
    }

    void buildInitialCentroids() {
        int* cent_ptr = centroids.mutable_data();
        const int n = nelement;

        double build_maxdist = maxdist;

        std::vector<int>    is_medoid(n, 0);
        std::vector<double> build_dysma(n, build_maxdist);

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

    double computeMaxDist() {
        const int n = nelement;
        double max_val = 0.0;

#ifdef _OPENMP
        #pragma omp parallel for reduction(max:max_val) schedule(static)
#endif
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double v = get_dist(i, j);
                if (v > max_val) max_val = v;
            }

        if (max_val <= 0.0) max_val = 1.0;
        return 1.1 * max_val + 1.0;
    }

    // -----------------------------------------------------------------------
    // FastPAM1 SWAP: O(n^2) per iteration instead of O(kn^2).
    //
    // For each candidate h, scan all points j once and accumulate the swap
    // cost delta[k] for every medoid k simultaneously:
    //   base(h) = sum_j  w_j * min(0, d(h,j) - dysma[j])
    //   delta[m_j] += w_j * (min(dysmb[j], d(h,j)) - dysma[j]) - base_contrib_j
    //   total_delta(m, h) = base + delta[m]
    // Reference: Schubert & Rousseeuw, "Faster k-Medoids Clustering" (2019).
    // -----------------------------------------------------------------------
    py::array_t<int> runclusterloop() {
        int* cent_ptr = centroids.mutable_data();
        int* cid_ptr  = clusterid.mutable_data();
        const int n = nelement;

        for (int i = 0; i < n; i++) cid_ptr[i] = -1;

        if (npass > 0) {
            buildInitialCentroids();
        }

        double dzsky = 1;
        int hbest = -1, nbest = -1;
        double total = -1;

        std::vector<int> is_medoid(n, 0);

        do {
            std::fill(is_medoid.begin(), is_medoid.end(), 0);
            for (int k = 0; k < nclusters; k++) is_medoid[cent_ptr[k]] = 1;

#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < n; i++) {
                double best1 = maxdist, best2 = maxdist;
                int best_k = 0;
                for (int k = 0; k < nclusters; k++) {
                    double dist = get_dist(i, cent_ptr[k]);
                    if (best1 > dist) {
                        best2 = best1;
                        best1 = dist;
                        best_k = k;
                    } else if (best2 > dist) {
                        best2 = dist;
                    }
                }
                dysma[i] = best1;
                dysmb[i] = best2;
                tclusterid[i] = best_k;
            }

            if (total < 0) {
                total = 0;
                #pragma omp parallel for reduction(+:total) schedule(static)
                for (int i = 0; i < n; i++) {
                    total += wt_ptr[i] * dysma[i];
                }
            }

            dzsky = 1;
            hbest = -1;
            nbest = -1;

#ifdef _OPENMP
            #pragma omp parallel
            {
                double local_dzsky = 1.0;
                int local_hbest = -1, local_nbest = -1;
                std::vector<double> delta(nclusters);

                #pragma omp for schedule(static) nowait
                for (int h = 0; h < n; h++) {
                    if (is_medoid[h]) continue;

                    double base = 0.0;
                    std::fill(delta.begin(), delta.end(), 0.0);

                    for (int j = 0; j < n; j++) {
                        double d_hj = get_dist(h, j);
                        double d_nearest = dysma[j];
                        int m_j = tclusterid[j];

                        double base_contrib = wt_ptr[j] * fmin(0.0, d_hj - d_nearest);
                        base += base_contrib;

                        double actual_contrib = wt_ptr[j] * (fmin(dysmb[j], d_hj) - d_nearest);
                        delta[m_j] += actual_contrib - base_contrib;
                    }

                    for (int k = 0; k < nclusters; k++) {
                        double total_delta = base + delta[k];
                        if (total_delta < local_dzsky) {
                            local_dzsky = total_delta;
                            local_hbest = h;
                            local_nbest = cent_ptr[k];
                        }
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
#else
            {
                std::vector<double> delta(nclusters);

                for (int h = 0; h < n; h++) {
                    if (is_medoid[h]) continue;

                    double base = 0.0;
                    std::fill(delta.begin(), delta.end(), 0.0);

                    for (int j = 0; j < n; j++) {
                        double d_hj = get_dist(h, j);
                        double d_nearest = dysma[j];
                        int m_j = tclusterid[j];

                        double base_contrib = wt_ptr[j] * fmin(0.0, d_hj - d_nearest);
                        base += base_contrib;

                        double actual_contrib = wt_ptr[j] * (fmin(dysmb[j], d_hj) - d_nearest);
                        delta[m_j] += actual_contrib - base_contrib;
                    }

                    for (int k = 0; k < nclusters; k++) {
                        double total_delta = base + delta[k];
                        if (total_delta < dzsky) {
                            dzsky = total_delta;
                            hbest = h;
                            nbest = cent_ptr[k];
                        }
                    }
                }
            }
#endif

            if (dzsky < WEIGHTED_CLUST_TOL) {
                for (int k = 0; k < nclusters; k++) {
                    if (cent_ptr[k] == nbest) {
                        cent_ptr[k] = hbest;
                    }
                }
                total += dzsky;
            }
        } while (dzsky < WEIGHTED_CLUST_TOL);

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; j++) {
            cid_ptr[j] = cent_ptr[tclusterid[j]];
        }

        return clusterid;
    }


private:
    int nelement;
    py::array_t<double> diss;
    py::array_t<int>    centroids;
    int npass;
    py::array_t<double> weights;

    const double* diss_ptr;
    const double* cond_ptr;
    const double* wt_ptr;
    bool use_condensed;

    py::array_t<int>     clusterid;
    std::vector<int>     tclusterid;

    double maxdist;
    std::vector<double> dysma;
    std::vector<double> dysmb;

    int nclusters;
};
