#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <xsimd/xsimd.hpp>
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

        auto ptr_result = result.mutable_unchecked<1>();

        // 初始化结果为 0
        for (int i = 0; i < ilen; i++) {
            ptr_result(i) = 0;
        }

        // 计算总权重
        double totweights = 0.0;
        for (int i = 0; i < ilen; i++) {
            totweights += ptr_weights(ptr_indiv(i));
        }

        // 计算每个个体的贡献
        #pragma omp parallel for
        for (int i = 0; i < ilen; i++) {
            int pos_i = ptr_indiv(i);
            double i_weight = ptr_weights(pos_i);

            for (int j = i + 1; j < ilen; j++) {
                int pos_j = ptr_indiv(j);

                double diss = ptr_dist(pos_i, pos_j);

                xsimd::batch<double> diss_batch(diss);
                xsimd::batch<double> i_weight_batch(i_weight);

                ptr_result(i) += diss_batch.get(0) * ptr_weights(pos_j);
                ptr_result(j) += diss_batch.get(0) * i_weight_batch.get(0);
            }

            if (totweights > 0) {
                ptr_result(i) /= totweights;
            }
        }

        return result;
    }

private:
    py::array_t<double> distmatrix;  // 距离矩阵
    py::array_t<int> individuals;    // 某组数据点的集合
    py::array_t<double> weights;     // 权重数组

    int ilen;
    py::array_t<double> result;
};