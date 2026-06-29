/*
 * LCPprodDistance: DSS-based LCP with product-duration weighting.
 *
 * The raw dissimilarity is constructed from prefix-duration vectors:
 *
 *   d(x, y) = A(x, x) + A(y, y) - 2 A(x, y),
 *
 * where A(x, y) is the sum of duration products over the common
 * DSS prefix or suffix. The resulting raw value is non-negative
 * up to floating-point tolerance.
 *
 * Only norm="none" is supported. This is intentional as LCPprod/RLCPprod
 * are returned on their raw squared-duration scale, and automatic
 * normalization is not defined for this measure.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "utils.h"
#include "dp_utils.h"
#include "lcp_input_validation.h"

namespace py = pybind11;

class LCPprodDistance {
public:
    LCPprodDistance(py::array_t<int> sequences,
                    py::array_t<double> durations,
                    py::array_t<int> seqlength,
                    py::array_t<double> totaldur,
                    int norm,
                    int sign,
                    py::array_t<int> refseqS)
            : norm(norm), sign(sign) {
        py::print(sign > 0 ? "[>] Starting LCPprod (DSS-based LCP with product duration)..."
                          : "[>] Starting RLCPprod (DSS-based LCP with product duration)...");
        std::cout << std::flush;

        lcp_input::require_lcpprod_norm(norm);
        lcp_input::validate_spell_distance_inputs(sequences, durations, seqlength, true);
        lcp_input::require_sign(sign);

        this->sequences = sequences;
        this->durations = durations;
        this->seqlength = seqlength;
        this->totaldur = totaldur;

        auto seq_shape = sequences.shape();
        original_nseq_ = static_cast<int>(seq_shape[0]);
        ncols = static_cast<int>(seq_shape[1]);

        lcp_input::validate_totaldur(totaldur, durations, seqlength, original_nseq_);

        const auto refseq_cfg = lcp_input::parse_refseq(refseqS, original_nseq_);
        has_refseq = refseq_cfg.has_refseq;
        nseq = refseq_cfg.nseq;
        rseq1 = refseq_cfg.rseq1;
        rseq2 = refseq_cfg.rseq2;
    }

    double compute_distance(int i, int j) {
        lcp_input::require_pair_indices(i, j, original_nseq_);
        return compute_distance_unchecked(i, j);
    }

    py::array_t<double> compute_all_distances() {
        auto dist_matrix = py::array_t<double>({nseq, nseq});
        return dp_utils::compute_all_distances_simple(
                nseq,
                dist_matrix,
                [this](int i, int j) { return this->compute_distance_unchecked(i, j); }
        );
    }

    py::array_t<double> compute_condensed_distances() {
        return dp_utils::compute_condensed_distances_simple(
                nseq,
                [this](int i, int j) { return this->compute_distance_unchecked(i, j); }
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
                [this](int is, int rseq) { return this->compute_distance_unchecked(is, rseq); }
        );
    }

private:
    double compute_distance_unchecked(int i, int j) {
        auto ptr_seq = sequences.unchecked<2>();
        auto ptr_dur = durations.unchecked<2>();
        auto ptr_len = seqlength.unchecked<1>();

        int len_i = ptr_len(i);
        int len_j = ptr_len(j);
        int minlen = std::min(len_i, len_j);

        double self_i = 0.0;
        for (int t = 0; t < len_i; t++) {
            const double d = ptr_dur(i, t);
            self_i += d * d;
        }

        double self_j = 0.0;
        for (int t = 0; t < len_j; t++) {
            const double d = ptr_dur(j, t);
            self_j += d * d;
        }

        if (len_i == 0 || len_j == 0) {
            return self_i + self_j;
        }

        int k = 0;
        if (sign > 0) {
            while (k < minlen && ptr_seq(i, k) == ptr_seq(j, k)) {
                k++;
            }
        } else {
            while (
                k < minlen &&
                ptr_seq(i, len_i - 1 - k) == ptr_seq(j, len_j - 1 - k)
            ) {
                k++;
            }
        }

        double shared = 0.0;
        if (sign > 0) {
            for (int t = 0; t < k; t++) {
                shared += ptr_dur(i, t) * ptr_dur(j, t);
            }
        } else {
            for (int t = 0; t < k; t++) {
                shared +=
                    ptr_dur(i, len_i - 1 - t) *
                    ptr_dur(j, len_j - 1 - t);
            }
        }

        double raw = self_i + self_j - 2.0 * shared;

        if (raw < 0.0 && raw > -1e-12) {
            raw = 0.0;
        }

        if (raw < 0.0) {
            throw std::runtime_error(
                "LCPprod distance became negative beyond numerical tolerance."
            );
        }

        return raw;
    }

    py::array_t<int> sequences;
    py::array_t<double> durations;
    py::array_t<int> seqlength;
    py::array_t<double> totaldur;
    int norm;
    int sign;
    int nseq;
    int original_nseq_;
    int ncols;

    bool has_refseq = false;
    int rseq1;
    int rseq2;
};
