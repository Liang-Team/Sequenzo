/*
 * LCPmstDistance: DSS-based LCP with Minimal Shared Time (duration-aware).
 *
 * Python supplies distinct successive state (DSS) spell sequences and spell durations.
 * Element index t refers to the t-th spell-state in the DSS prefix, not a calendar position.
 * The common prefix length k is the number of consecutive matching spell states from the
 * start (forward) or from the end (reverse, RLCPmst).
 *
 * Forward (sign > 0): A_mst = sum_{t=0}^{k-1} min(d_x(t), d_y(t))
 * Reverse (sign < 0): suffix match from the last spell backward.
 *
 * Raw distance: raw_d = T_x + T_y - 2 * A_mst
 * where T_x, T_y are total spell durations (sum of spell lengths per sequence).
 *
 * Usage (Python):
 *   d = get_distance_matrix(seqdata, method="LCPmst", norm="gmean")
 *   d = get_distance_matrix(seqdata, method="RLCPmst", norm="gmean")
 *
 * Source:
 *   Elzinga, C. H. (2007). Sequence analysis: Metric representations of
 *   categorical time series. Manuscript, Dept of Social Science Research
 *   Methods, Vrije Universiteit, Amsterdam.
 *
 * @File    : LCPmstDistance.cpp
 * @Author  : Yuqi Liang 梁彧祺
 * @Desc    : Duration-aware LCP using minimal shared time (MST). Supports LCPmst and RLCPmst.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

class LCPmstDistance {
public:
    /*
     * Constructor.
     * - sequences: DSS spell-state matrix, shape (nseq, max_spells).
     * - durations: spell durations, shape (nseq, max_spells); d(t) is duration of spell t.
     * - seqlength: number of spells per sequence, shape (nseq,).
     * - totaldur: precomputed total duration per sequence, shape (nseq,); T_i = sum_t durations[i,t].
     * - norm: normalization index (0=none, 1=maxlength, 2=gmean, 3=maxdist, 4=YujianBo).
     * - sign: 1 = forward (LCPmst), -1 = reverse from end (RLCPmst).
     * - refseqS: reference sequence indices [rseq1, rseq2).
     */
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

        try {
            this->sequences = sequences;
            this->durations = durations;
            this->seqlength = seqlength;
            this->totaldur = totaldur;

            auto seq_shape = sequences.shape();
            nseq = static_cast<int>(seq_shape[0]);
            ncols = static_cast<int>(seq_shape[1]);

            // Reference sequence range (same convention as LCPdistance / LCPspell).
            nans = nseq;
            rseq1 = refseqS.at(0);
            rseq2 = refseqS.at(1);
            if (rseq1 < rseq2) {
                nseq = rseq1;
                nans = nseq * (rseq2 - rseq1);
            } else {
                rseq1 = rseq1 - 1;
            }
        } catch (const std::exception& e) {
            py::print("Error in LCPmstDistance constructor: ", e.what());
            throw;
        }
    }

    /*
     * Compute LCPmst distance between sequence i and sequence j.
     *
     * Forward (sign > 0): common prefix from start.
     * Reverse (sign < 0): common suffix from end (RLCPmst).
     */
    double compute_distance(int i, int j) {
        try {
            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_dur = durations.unchecked<2>();
            auto ptr_len = seqlength.unchecked<1>();
            auto ptr_total = totaldur.unchecked<1>();

            int len_i = ptr_len(i);
            int len_j = ptr_len(j);
            int minlen = std::min(len_i, len_j);

            double Tx = ptr_total(i);
            double Ty = ptr_total(j);

            // Edge case: empty sequence
            if (len_i == 0 || len_j == 0) {
                double raw = Tx + Ty;
                if (norm == 0) return raw;
                double denom = (norm == 1) ? std::max(Tx, Ty) : (norm == 2) ? std::sqrt(Tx * Ty) : (Tx + Ty);
                if (denom == 0.0) return (raw == 0.0) ? 0.0 : 1.0;
                double d = normalize_distance(raw, Tx + Ty, Tx, Ty, norm);
                return std::max(0.0, std::min(1.0, d));
            }

            // Step 1: find match length k (prefix if forward, suffix if reverse)
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

            // Step 2: accumulate A_mst over matched positions
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

            // Step 3: raw distance
            double raw = Tx + Ty - 2.0 * A;

            if (norm == 0) return raw;

            // Step 4: normalize. For LCPmst, maxdist = Tx + Ty is a natural upper bound
            // because A_mst >= 0 and A_mst <= min(Tx, Ty).
            double maxdist = Tx + Ty;
            double d = normalize_distance(raw, maxdist, Tx, Ty, norm);
            return std::max(0.0, std::min(1.0, d));
        } catch (const std::exception& e) {
            py::print("Error in LCPmstDistance::compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            auto dist_matrix = py::array_t<double>({nseq, nseq});
            return dp_utils::compute_all_distances_simple(
                    nseq,
                    dist_matrix,
                    [this](int i, int j) { return this->compute_distance(i, j); }
            );
        } catch (const std::exception& e) {
            py::print("Error in LCPmstDistance::compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_condensed_distances() {
        try {
            return dp_utils::compute_condensed_distances_simple(
                    nseq,
                    [this](int i, int j) { return this->compute_distance(i, j); }
            );
        } catch (const std::exception& e) {
            py::print("Error in LCPmstDistance::compute_condensed_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
            return dp_utils::compute_refseq_distances_simple(
                    nseq,
                    rseq1,
                    rseq2,
                    refdist_matrix,
                    [this](int is, int rseq) { return this->compute_distance(is, rseq); }
            );
        } catch (const std::exception& e) {
            py::print("Error in LCPmstDistance::compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<double> durations;
    py::array_t<int> seqlength;
    py::array_t<double> totaldur;
    int norm;
    int sign;
    int nseq;
    int ncols;

    int nans;
    int rseq1;
    int rseq2;
};
