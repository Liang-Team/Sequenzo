/*
 * LCPmstDistance: DSS-based LCP with Minimal Shared Time (duration-aware).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"
#include "lcp_input_validation.h"

namespace py = pybind11;

class LCPmstDistance {
public:
    LCPmstDistance(py::array_t<int> sequences,
                   py::array_t<double> durations,
                   py::array_t<int> seqlength,
                   py::array_t<double> totaldur,
                   int norm,
                   int sign,
                   py::array_t<int> refseqS)
            : norm(norm), sign(sign) {
        py::print(sign > 0 ? "[>] Starting LCPmst (DSS-based LCP with minimal shared time)..."
                          : "[>] Starting RLCPmst (DSS-based LCP with minimal shared time)...");
        std::cout << std::flush;

        lcp_input::require_lcpmst_norm(norm);
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
        auto ptr_total = totaldur.unchecked<1>();

        int len_i = ptr_len(i);
        int len_j = ptr_len(j);
        int minlen = std::min(len_i, len_j);

        double Tx = ptr_total(i);
        double Ty = ptr_total(j);

        if (len_i == 0 || len_j == 0) {
            double raw = Tx + Ty;
            if (norm == 0) return raw;
            double denom = (norm == 2) ? std::sqrt(Tx * Ty) : (Tx + Ty);
            if (denom == 0.0) return (raw == 0.0) ? 0.0 : 1.0;
            double d = normalize_distance(raw, Tx + Ty, Tx, Ty, norm);
            return lcp_input::clamp_unit_interval_or_throw(d, "normalized LCPmst distance");
        }

        int k = 0;
        if (sign > 0) {
            while (k < minlen && ptr_seq(i, k) == ptr_seq(j, k)) {
                k++;
            }
        } else {
            while (k < minlen && ptr_seq(i, len_i - 1 - k) == ptr_seq(j, len_j - 1 - k)) {
                k++;
            }
        }

        double A = 0.0;
        if (sign > 0) {
            for (int t = 0; t < k; t++) {
                A += std::min(ptr_dur(i, t), ptr_dur(j, t));
            }
        } else {
            for (int t = 0; t < k; t++) {
                A += std::min(ptr_dur(i, len_i - 1 - t), ptr_dur(j, len_j - 1 - t));
            }
        }

        double raw = Tx + Ty - 2.0 * A;

        if (norm == 0) return raw;

        double maxdist = Tx + Ty;
        double d = normalize_distance(raw, maxdist, Tx, Ty, norm);
        return lcp_input::clamp_unit_interval_or_throw(d, "normalized LCPmst distance");
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
