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
            this->nclusters = static_cast<int>(centroids.size());
            this->tclusterid.resize(nelements);

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

            this->maxdist = 0.0;
            this->computeMaxDist();

            dysma.resize(nelements, maxdist);
            dysmb.resize(nelements, maxdist);
        } catch (const exception &e) {
            py::print("Error: ", e.what());
        }
    }

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

    void computeMaxDist() {
        const int n = nelements;
        double max_val = 0.0;

#ifdef _OPENMP
        #pragma omp parallel for reduction(max:max_val) schedule(static)
#endif
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double v = get_dist(i, j);
                if (v > max_val) max_val = v;
            }

        maxdist = 1.1 * max_val + 1.0;
    }

    void buildInitialCentroids() {
        int* cent_ptr = centroids.mutable_data();
        const int n = nelements;

        double build_maxdist = maxdist;

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

    py::array_t<int> runclusterloop() {
        int* cent_ptr = centroids.mutable_data();
        const int n = nelements;

        if (npass > 0) {
            buildInitialCentroids();
        }

        double dzsky;
        int hbest = -1;
        int nbest = -1;
        int k;
        double total = -1.0;

        do {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                dysmb[i] = maxdist;
                dysma[i] = maxdist;

                for (int k = 0; k < nclusters; k++) {
                    double dist = get_dist(i, cent_ptr[k]);

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
                for (int i = 0; i < n; i++) {
                    total += wt_ptr[i] * dysma[i];
                }
            }

            dzsky = 1;

            vector<bool> is_medoid_flag(n, false);
            for (int kk = 0; kk < nclusters; kk++) is_medoid_flag[cent_ptr[kk]] = true;

            #pragma omp parallel for schedule(static)
            for (int h = 0; h < n; h++) {
                if (is_medoid_flag[h])
                    continue;

                double local_dzsky = 1.0;
                int local_hbest = -1;
                int local_nbest = -1;

                for (int k = 0; k < nclusters; k++) {
                    int i = cent_ptr[k];
                    double dz = 0.0;

                    for (int j = 0; j < n; j++) {
                        double d_ij = get_dist(i, j);
                        double d_hj = get_dist(h, j);
                        if (d_ij == dysma[j]) {
                            double small = (dysmb[j] > d_hj) ? d_hj : dysmb[j];
                            dz += wt_ptr[j] * (-dysma[j] + small);
                        } else if (d_hj < dysma[j]) {
                            dz += wt_ptr[j] * (-dysma[j] + d_hj);
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
                    if (cent_ptr[k] == nbest) {
                        cent_ptr[k] = hbest;
                    }
                }
                total += dzsky;
            }

        } while (dzsky < 0);

        return getResultArray();
    }

    py::array_t<int> getResultArray() const {
        py::array_t<int> result(nelements);
        int* res_ptr = result.mutable_data();
        const int* cent = centroids.data();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nelements; ++i) {
            res_ptr[i] = cent[tclusterid[i]];
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

    const double* diss_ptr;
    const double* cond_ptr;
    const double* wt_ptr;
    bool use_condensed;
};
