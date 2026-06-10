/*
 * LCPdistance: Longest Common Prefix (LCP) and Reverse LCP (RLCP).
 * Optimized: raw pointer cache, removed try/catch from hot path.
 * Uses updated dp_utils.h (dynamic scheduling + diagonal skip).
 */

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

        this->sequences = sequences;

        auto seq_shape = sequences.shape();
        nseq = seq_shape[0];
        len = seq_shape[1];

        // Cache the row-major buffer once for the distance loop.
        seq_ptr = sequences.data();

        nans = nseq;
        rseq1 = refseqS.at(0);
        rseq2 = refseqS.at(1);
        if(rseq1 < rseq2){
            nseq = rseq1;
            nans = nseq * (rseq2 - rseq1);
        }else{
            rseq1 = rseq1 - 1;
        }
    }

    double compute_distance(int is, int js) {
        const int m = len;
        const int n = len;
        const int minimum = (m < n) ? m : n;

        const int* row_i = seq_ptr + static_cast<ptrdiff_t>(is) * len;
        const int* row_j = seq_ptr + static_cast<ptrdiff_t>(js) * len;

        int length = 0;

        if (sign > 0) {
            // Forward LCP
            while (length < minimum && row_i[length] == row_j[length]) {
                length++;
            }
        } else {
            // Reverse LCP
            while (length < minimum &&
                   row_i[m - 1 - length] == row_j[n - 1 - length]) {
                length++;
            }
        }

        return normalize_distance(n+m-2.0*length, n+m, m, n, norm);
    }

    py::array_t<double> compute_all_distances() {
        auto dist_matrix = py::array_t<double>({nseq, nseq});
        return dp_utils::compute_all_distances_simple(
            nseq,
            dist_matrix,
            [this](int i, int j){ return this->compute_distance(i, j); }
        );
    }

    py::array_t<double> compute_condensed_distances() {
        return dp_utils::compute_condensed_distances_simple(
            nseq,
            [this](int i, int j){ return this->compute_distance(i, j); }
        );
    }

    py::array_t<double> compute_refseq_distances() {
        auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
        return dp_utils::compute_refseq_distances_simple(
            nseq,
            rseq1,
            rseq2,
            refdist_matrix,
            [this](int is, int rseq){ return this->compute_distance(is, rseq); }
        );
    }

private:
    py::array_t<int> sequences;
    int norm;
    int nseq;
    int len;
    int sign;
    const int* seq_ptr = nullptr;

    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
};
