/*
 * LCPdistance: Longest Common Prefix (LCP) and Reverse LCP (RLCP).
 * Optimized: raw pointer cache, removed try/catch from hot path.
 * Uses updated dp_utils.h (dynamic scheduling + diagonal skip).
 *
 * Input contract: full position-wise sequences of equal length. State codes such
 * as 0 are treated as ordinary states, not padding sentinels.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "utils.h"
#include "dp_utils.h"
#include "lcp_input_validation.h"

namespace py = pybind11;

using IntArrayC = py::array_t<int, py::array::c_style | py::array::forcecast>;

class LCPdistance{
public:
    LCPdistance(IntArrayC sequences, int norm, int sign, py::array_t<int> refseqS)
                : norm(norm), sign(sign){
        py::print("[>] Starting (Reverse) Longest Common Prefix(LCP/RLCP)...");
        std::cout << std::flush;

        lcp_input::require_2d(sequences, "sequences");
        lcp_input::require_sign(sign);
        lcp_input::require_norm(norm);

        this->sequences = sequences;

        auto seq_shape = sequences.shape();
        original_nseq_ = static_cast<int>(seq_shape[0]);
        len = static_cast<int>(seq_shape[1]);

        seq_ptr = sequences.data();

        const auto refseq_cfg = lcp_input::parse_refseq(refseqS, original_nseq_);
        has_refseq = refseq_cfg.has_refseq;
        nseq = refseq_cfg.nseq;
        rseq1 = refseq_cfg.rseq1;
        rseq2 = refseq_cfg.rseq2;
    }

    double compute_distance(int is, int js) {
        lcp_input::require_pair_indices(is, js, original_nseq_);
        return compute_distance_unchecked(is, js);
    }

    py::array_t<double> compute_all_distances() {
        auto dist_matrix = py::array_t<double>({nseq, nseq});
        return dp_utils::compute_all_distances_simple(
            nseq,
            dist_matrix,
            [this](int i, int j){ return this->compute_distance_unchecked(i, j); }
        );
    }

    py::array_t<double> compute_condensed_distances() {
        return dp_utils::compute_condensed_distances_simple(
            nseq,
            [this](int i, int j){ return this->compute_distance_unchecked(i, j); }
        );
    }

    py::array_t<double> compute_refseq_distances() {
        lcp_input::require_refseq_for_compute({has_refseq, nseq, rseq1, rseq2});
        auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
        return dp_utils::compute_refseq_distances_simple(
            nseq,
            rseq1,
            rseq2,
            refdist_matrix,
            [this](int is, int rseq){ return this->compute_distance_unchecked(is, rseq); }
        );
    }

private:
    double compute_distance_unchecked(int is, int js) {
        const int m = len;
        const int n = len;
        const int minimum = (m < n) ? m : n;

        const int* row_i = seq_ptr + static_cast<ptrdiff_t>(is) * len;
        const int* row_j = seq_ptr + static_cast<ptrdiff_t>(js) * len;

        int length = 0;

        if (sign > 0) {
            while (length < minimum && row_i[length] == row_j[length]) {
                length++;
            }
        } else {
            while (length < minimum &&
                   row_i[m - 1 - length] == row_j[n - 1 - length]) {
                length++;
            }
        }

        return normalize_distance(n+m-2.0*length, n+m, m, n, norm);
    }

    py::array_t<int> sequences;
    int norm;
    int nseq;
    int original_nseq_;
    int len;
    int sign;
    const int* seq_ptr = nullptr;

    bool has_refseq = false;
    int rseq1 = -1;
    int rseq2 = -1;
};
