#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include "utils.h"

namespace py = pybind11;

class LCPdistance{
public:
    LCPdistance(py::array_t<int> sequences, int norm, int sign, py::array_t<int> refseqS)
                : norm(norm), sign(sign){
        py::print("[>] Starting (Reverse) Longest Common Prefix(LCP/RLCP)...");
        std::cout << std::flush;

        try{
            this->sequences = sequences;

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

            int length = 0;
            auto ptr_seq = sequences.unchecked<2>();

            if(sign > 0){
                while(ptr_seq(is, length) == ptr_seq(js, length) && length < minimum){
                    length ++;
                }
            } else{
                length = 1;
                while(ptr_seq(is, (m - length)) == ptr_seq(js, (n - length)) && length <= minimum){
                    length ++;
                }
                length --;
            }

            return normalize_distance(n+m-2.0*length, n+m, m, n, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            auto buffer = dist_matrix.mutable_unchecked<2>();

            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < nseq; i++) {
                for (int j = i; j < nseq; j++) {
                    buffer(i, j) = compute_distance(i, j);
                }
            }

            #pragma omp for schedule(static)
            for (int i = 0; i < nseq; ++i) {
                for (int j = 0; j < i; ++j) {  // 遍历下三角的每一行
                    buffer(i, j) = buffer(j, i);
                }
            }

            return dist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto buffer = refdist_matrix.mutable_unchecked<2>();

            double cmpres = 0;
#pragma omp parallel for collapse(2) schedule(dynamic)
            for (int rseq = rseq1; rseq < rseq2; rseq ++) {
                for (int is = 0; is < nseq; is ++) {
                    if(is == rseq){
                        cmpres = 0;
                    }else{
                        cmpres = compute_distance(is, rseq);
                    }

                    buffer(is, rseq-rseq1) = cmpres;
                }
            }

            return refdist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    int norm;
    int nseq;
    int len;
    int sign;
    py::array_t<double> dist_matrix;

    py::array_t<int> refseqS;
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};