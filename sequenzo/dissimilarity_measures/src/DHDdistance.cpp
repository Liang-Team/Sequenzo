#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <xsimd/xsimd.hpp>

namespace py = pybind11;

class DHDdistance{
public:
    DHDdistance(py::array_t<int> sequences, py::array_t<double> sm, int norm, double maxdist, py::array_t<int> refseqS)
            : norm(norm), maxdist(maxdist){
        py::print("[>] Starting (Dynamic) Hamming Distance(DHD/HAM)...");
        std::cout << std::flush;

        try{
            this->sequences = sequences;
            this->sm = sm;

            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            len = seq_shape[1];

            dist_matrix = py::array_t<double>({nseq, nseq});

            // about reference sequences :
            nans = nseq;

            rseq1 = refseqS.at(0);
            rseq2 = refseqS.at(1);
            if(rseq1 < rseq2){
                nseq = rseq1;
                nans = nseq * (rseq2 - rseq1);
            }else{
                rseq1 = rseq1 - 1;
            }
            refdist_matrix = py::array_t<double>({nseq, (rseq2-rseq1)});
        } catch (const std::exception& e){
            py::print("Error in constructor: ", e.what());
            throw ;
        }
    }

    double compute_distance(int is, int js) {
        try {
            int m = len;
            int n = len;
            int minimum = m;
            if(n < m) minimum = n;
            double cost = 0;

            auto ptr_sm = sm.unchecked<3>();
            auto ptr_seq = sequences.unchecked<2>();

            // 使用 SIMD 批量处理
            const int simd_width = xsimd::batch<double>::size;
            int i = 0;

            for(; i + simd_width <= minimum; i += simd_width) {
                alignas(32) int seq_is[simd_width];
                alignas(32) int seq_js[simd_width];
                alignas(32) double tmp[simd_width];

                // 加载序列
                for(int j = 0; j < simd_width; j++) {
                    seq_is[j] = ptr_seq(is, i + j);
                    seq_js[j] = ptr_seq(js, i + j);
                }
                xsimd::batch<int> batch_seq_is = xsimd::load_unaligned(seq_is);
                xsimd::batch<int> batch_seq_js = xsimd::load_unaligned(seq_js);

                // 比较是否相等
                auto equal_mask = (batch_seq_is == batch_seq_js);
                for(int j = 0; j < simd_width; j++) {
                    tmp[j] = equal_mask.get(j) ? 0.0 : ptr_sm(i + j, seq_is[j], seq_js[j]);
                }

                xsimd::batch<double> costs = xsimd::load_unaligned(tmp);
                cost += xsimd::reduce_add(costs);
            }

            // 处理尾部：用 SIMD 填充无效数据
            for(; i < minimum; i += simd_width) {
                alignas(32) double tmp[simd_width];
                int bound = std::min(simd_width, minimum - i);
                for(int j = 0; j < simd_width; j++) {
                    tmp[j] = (j < bound) ? ptr_sm(i + j, ptr_seq(is, i + j), ptr_seq(js, i + j)) : 0.0;
                }
                xsimd::batch<double> costs = xsimd::load_unaligned(tmp);
                cost += xsimd::reduce_add(costs);
            }

            return normalize_distance(cost, maxdist, maxdist, maxdist, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            return dp_utils::compute_all_distances_simple(
                nseq,
                dist_matrix,
                [this](int i, int j){ return this->compute_distance(i, j); }
            );
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            return dp_utils::compute_refseq_distances_simple(
                nseq,
                rseq1,
                rseq2,
                refdist_matrix,
                [this](int is, int rseq){ return this->compute_distance(is, rseq); }
            );
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<double> sm;
    int norm;
    int nseq;
    int len;
    py::array_t<double> dist_matrix;
    double maxdist;

    py::array_t<int> refseqS;
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};