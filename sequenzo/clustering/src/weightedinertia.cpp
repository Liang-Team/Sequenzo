#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class weightedinertia {
public:
    weightedinertia(py::array_t<double> distmatrix, py::array_t<int> individuals, py::array_t<double> weights) {
        std::cout << std::flush;  // 刷新 C++ 输出

        try {
            this->distmatrix = distmatrix;
            this->individuals = individuals;
            this->weights = weights;

            ilen = individuals.size();

            result = py::array_t<double>(ilen);
        } catch (const std::exception& e) {
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    py::array_t<double> tmrWeightedInertiaContrib() {
        auto ptr_dist = distmatrix.unchecked<2>();
        auto ptr_indiv = individuals.unchecked<1>();
        auto ptr_weights = weights.unchecked<1>();

        py::array_t<double> result_local(ilen);
        auto ptr_result = result_local.mutable_unchecked<1>();

        for (int i = 0; i < ilen; i++) {
            ptr_result(i) = 0.0;
        }

        double totweights = 0.0;

        #pragma omp parallel for reduction(+:totweights)
        for (int i = 0; i < ilen; i++) {
            totweights += ptr_weights(ptr_indiv(i));
        }

        // 每个线程使用局部 result 副本，最后归约合并
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
                int pos_i = ptr_indiv(i);
                double i_weight = ptr_weights(pos_i);

                for (int j = i + 1; j < ilen; ++j) {
                    int pos_j = ptr_indiv(j);
                    double diss = ptr_dist(pos_i, pos_j);

                    local[i] += diss * ptr_weights(pos_j);
                    local[j] += diss * i_weight;
                }
            }
        }

        // 合并各线程的 result_private 到主 result
        for (int t = 0; t < nthreads; ++t) {
            for (int i = 0; i < ilen; ++i) {
                ptr_result(i) += result_private[t][i];
            }
        }

        if (totweights > 0) {
            #pragma omp parallel for
            for (int i = 0; i < ilen; ++i) {
                ptr_result(i) /= totweights;
            }
        }

        return result_local;
    }

private:
    py::array_t<double> distmatrix;  // 距离矩阵
    py::array_t<int> individuals;    // 某组数据点的集合
    py::array_t<double> weights;     // 权重数组

    int ilen;
    py::array_t<double> result;
};