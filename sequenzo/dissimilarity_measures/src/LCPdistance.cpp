#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"

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