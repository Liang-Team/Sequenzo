#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cfloat>
#include <cmath>

namespace py = pybind11;

class weightedinertia {
public:
    weightedinertia(py::array_t<double> distmatrix, py::array_t<int> individuals, py::array_t<double> weights)
        : distmatrix(distmatrix), individuals(individuals), weights(weights),
          ilen(static_cast<int>(individuals.size()))
    {
    }

    py::array_t<double> tmrWeightedInertiaContrib() {
        const double* dist_ptr  = distmatrix.data();
        const int*    indiv_ptr = individuals.data();
        const double* wt_ptr    = weights.data();
        const int n = static_cast<int>(distmatrix.shape(0));

        py::array_t<double> result_local(ilen);
        double* res_ptr = result_local.mutable_data();

        for (int i = 0; i < ilen; i++) {
            res_ptr[i] = 0.0;
        }

        double totweights = 0.0;
        for (int i = 0; i < ilen; i++) {
            totweights += wt_ptr[indiv_ptr[i]];
        }

        int nthreads = 1;
        #ifdef _OPENMP
        #pragma omp parallel
        {
            #pragma omp single
            nthreads = omp_get_num_threads();
        }
        #endif

        std::vector<std::vector<double>> result_private(nthreads, std::vector<double>(ilen, 0.0));

        #pragma omp parallel
        {
            #ifdef _OPENMP
            int tid = omp_get_thread_num();
            #else
            int tid = 0;
            #endif
            auto& local = result_private[tid];

            #pragma omp for schedule(static)
            for (int i = 0; i < ilen; ++i) {
                int pos_i = indiv_ptr[i];
                const double* row_i = dist_ptr + static_cast<size_t>(pos_i) * n;
                double i_weight = wt_ptr[pos_i];

                for (int j = i + 1; j < ilen; ++j) {
                    int pos_j = indiv_ptr[j];
                    double diss = row_i[pos_j];

                    local[i] += diss * wt_ptr[pos_j];
                    local[j] += diss * i_weight;
                }
            }
        }

        for (int t = 0; t < nthreads; ++t) {
            for (int i = 0; i < ilen; ++i) {
                res_ptr[i] += result_private[t][i];
            }
        }

        if (totweights > 0) {
            const double inv_totweights = 1.0 / totweights;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < ilen; ++i) {
                res_ptr[i] *= inv_totweights;
            }
        }

        return result_local;
    }

private:
    py::array_t<double> distmatrix;
    py::array_t<int> individuals;
    py::array_t<double> weights;
    int ilen;
};
