#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <sstream>
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

            clusterid = py::array_t<int>(nelement);
            tclusterid.resize(nelement, -1);

            maxdist = find_max_value(diss);
            dysma.resize(nelement, maxdist);
            dysmb.resize(nelement, maxdist);

            fvect.resize(nelement, 0);
            nclusters = centroids.size();
        } catch (const std::exception& e){
            py::print("Error in constructor: ", e.what());
            throw;
        }
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
        auto ptr_centroids = centroids.mutable_data();

        // Reuse maxdist already computed in the constructor — avoids O(n²) recomputation.
        double build_maxdist = maxdist;

        std::vector<int>    is_medoid(nelement, 0);
        std::vector<double> build_dysma(nelement, build_maxdist);

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
                for (int i = 0; i < nelement; ++i) {
                    if (is_medoid[i]) continue;
                    double beter = 0.0;
                    for (int j = 0; j < nelement; ++j) {
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
            for (int i = 0; i < nelement; ++i) {
                if (is_medoid[i]) continue;
                double beter = 0.0;
                for (int j = 0; j < nelement; ++j) {
                    beter += ptr_weights[j] * fmax(0.0, build_dysma[j] - ptr_diss(i, j));
                }
                if (ammax <= beter) { ammax = beter; nmax = i; }
            }
#endif

            is_medoid[nmax] = 1;
            ptr_centroids[kk] = nmax;

            for (int j = 0; j < nelement; ++j)
                if (ptr_diss(nmax, j) < build_dysma[j])
                    build_dysma[j] = ptr_diss(nmax, j);
        }
    }

    double find_max_value(py::array_t<double> diss) {
        auto buf_info = diss.shape();
        auto ptr = diss.unchecked<2>();

        int rows = buf_info[0];
        int cols = buf_info[1];

        double max_val = std::numeric_limits<double>::lowest();

#ifdef _OPENMP
        #pragma omp parallel reduction(max:max_val)
        {
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    double val = ptr(i, j);
                    if (std::isfinite(val) && val > max_val)
                        max_val = val;
                }
            }
        }
#else
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                double val = ptr(i, j);
                if (std::isfinite(val) && val > max_val)
                    max_val = val;
            }
#endif

        if (!std::isfinite(max_val) || max_val <= 0.0) {
            max_val = 1.0;
        }

        return max_val;
    }

    py::array_t<int> runclusterloop() {
        auto ptr_diss      = diss.unchecked<2>();
        auto ptr_weights   = weights.unchecked<1>();
        auto ptr_centroids = centroids.mutable_data();
        auto ptr_clusterid = clusterid.mutable_data();

        for (int i = 0; i < nelement; i++) {
            ptr_clusterid[i] = -1;
        }

        // When npass > 0 always use PAM BUILD for initialisation (deterministic),
        // matching R's wcKMedoids(method="PAMonce").
        if (npass > 0) {
            buildInitialCentroids();
        }

        double dzsky = 1;
        int hbest = -1, nbest = -1;
        double total = -1;

        do {
            // Find nearest and second-nearest medoid for every point.
            // Each element i is independent → fully parallelisable.
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nelement; i++) {
                dysma[i] = maxdist;
                dysmb[i] = maxdist;
                int best_k = 0;
                for (int k = 0; k < nclusters; k++) {
                    int i_cluster = ptr_centroids[k];
                    double dist = ptr_diss(i, i_cluster);
                    if (!std::isfinite(dist)) {
                        dist = maxdist;
                    }

                    // Use strict > to match R PAM/PAMonce: first encountered medoid
                    // wins for equal-distance ties.
                    if (dysma[i] > dist) {
                        dysmb[i] = dysma[i];
                        dysma[i] = dist;
                        best_k = k;
                    } else if (dysmb[i] > dist) {
                        dysmb[i] = dist;
                    }
                }
                tclusterid[i] = best_k;
            }

            if (total < 0) {
                total = 0;
                #pragma omp parallel for reduction(+:total) schedule(static)
                for (int i = 0; i < nelement; i++) {
                    total += ptr_weights(i) * dysma[i];
                }
            }

            dzsky = 1;
            hbest = -1;
            nbest = -1;

            // For each current medoid i, find the best replacement h.
            for (int k = 0; k < nclusters; k++) {
                int i = ptr_centroids[k];
                double removeCost = 0;

                // Cost of removing this medoid.
                #pragma omp parallel for reduction(+:removeCost) schedule(static)
                for (int j = 0; j < nelement; j++) {
                    if (tclusterid[j] == k) {
                        removeCost += ptr_weights(j) * (dysmb[j] - dysma[j]);
                        fvect[j] = dysmb[j];
                    } else {
                        fvect[j] = dysma[j];
                    }
                }

                // Find the best new medoid h to replace i.
                #pragma omp parallel
                {
                    double local_dzsky = 1;
                    int local_hbest = -1, local_nbest = -1;

                    #pragma omp for schedule(static) nowait
                    for (int h = 0; h < nelement; h++) {
                        double dist_hi = ptr_diss(h, i);
                        if (!std::isfinite(dist_hi) || dist_hi <= 0) {
                            continue;
                        }
                        double addGain = removeCost;
                        for (int j = 0; j < nelement; j++) {
                            double dist_hj = ptr_diss(h, j);
                            if (!std::isfinite(dist_hj)) {
                                continue;
                            }
                            if (dist_hj < fvect[j]) {
                                addGain += ptr_weights(j) * (dist_hj - fvect[j]);
                            }
                        }

                        if (local_dzsky > addGain) {
                            local_dzsky = addGain;
                            local_hbest = h;
                            local_nbest = i;
                        }
                    }

                    #pragma omp critical
                    {
                        if (dzsky > local_dzsky) {
                            dzsky = local_dzsky;
                            hbest = local_hbest;
                            nbest = local_nbest;
                        }
                    }
                }
            }

            if (dzsky < WEIGHTED_CLUST_TOL) {
                for (int k = 0; k < nclusters; k++) {
                    if (ptr_centroids[k] == nbest) {
                        ptr_centroids[k] = hbest;
                    }
                }
                total += dzsky;
            }
        } while (dzsky < WEIGHTED_CLUST_TOL);

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < nelement; j++) {
            ptr_clusterid[j] = ptr_centroids[tclusterid[j]];
        }

        return clusterid;
    }


private:
    int nelement;
    py::array_t<double> diss;
    py::array_t<int>    centroids;
    int npass;
    py::array_t<double> weights;

    py::array_t<int>     clusterid;
    std::vector<int>     tclusterid;

    double maxdist;
    std::vector<double> dysma;
    std::vector<double> dysmb;

    std::vector<double> fvect;
    int nclusters;
};
