/*
 * LCPprodDistance: Position-wise LCP with Product Duration (duration-aware).
 *
 * Unlike LCPspell (which compares spell-by-spell), this variant compares
 * sequences position-wise at aligned time indices t = 0, 1, ..., L-1.
 * The common prefix length k is the number of consecutive matching states
 * from the start (forward) or from the end (reverse, RLCPprod).
 *
 * Forward (sign > 0): A_prod = sum_{t=0}^{k-1} dx[t] * dy[t]
 * Reverse (sign < 0): A_prod = sum_{t=0}^{k-1} dx[len_i-1-t] * dy[len_j-1-t]
 *
 * Raw distance: raw_d = T_x + T_y - 2 * A_prod
 * where T_x = sum_t dx[t], T_y = sum_t dy[t] (total duration per sequence).
 *
 * Note: raw_d_prod can be negative. When norm != "none", clamp to [0, 1].
 *
 * Usage (Python):
 *   d = get_distance_matrix(seqdata, method="LCPprod", norm="gmean", durations=dur_matrix)
 *   d = get_distance_matrix(seqdata, method="RLCPprod", norm="gmean", durations=dur_matrix)
 *
 * Source:
 *   Elzinga, C. H. (2007). Sequence analysis: Metric representations of
 *   categorical time series. Manuscript, Dept of Social Science Research
 *   Methods, Vrije Universiteit, Amsterdam.
 *
 * @File    : LCPprodDistance.cpp
 * @Author  : Yuqi Liang 梁彧祺
 * @Desc    : Duration-aware LCP using product of durations. Supports LCPprod and RLCPprod.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

class LCPprodDistance {
public:
    /*
     * Constructor.
     * - sequences: state matrix, shape (nseq, ncols); row i holds states at each time index.
     * - durations: position-wise durations, shape (nseq, ncols); durations[t] is the
     *   time length for position t. Default 1.0 for regular panels.
     * - seqlength: effective length per sequence (positions), shape (nseq,).
     * - totaldur: precomputed total duration per sequence, shape (nseq,); T_i = sum_t durations[i,t].
     * - norm: normalization index (0=none, 1=maxlength, 2=gmean, 3=maxdist, 4=YujianBo).
     * - sign: 1 = forward (LCPprod), -1 = reverse from end (RLCPprod).
     * - refseqS: reference sequence indices [rseq1, rseq2).
     */
    LCPprodDistance(py::array_t<int> sequences,
                    py::array_t<double> durations,
                    py::array_t<int> seqlength,
                    py::array_t<double> totaldur,
                    int norm,
                    int sign,
                    py::array_t<int> refseqS)
            : norm(norm), sign(sign) {
        py::print(sign > 0 ? "[>] Starting LCPprod (position-wise LCP with product duration)..."
                          : "[>] Starting RLCPprod (position-wise LCP with product duration)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->durations = durations;
            this->seqlength = seqlength;
            this->totaldur = totaldur;

            auto seq_shape = sequences.shape();
            nseq = static_cast<int>(seq_shape[0]);
            ncols = static_cast<int>(seq_shape[1]);

            dist_matrix = py::array_t<double>({nseq, nseq});

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
            refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
        } catch (const std::exception& e) {
            py::print("Error in LCPprodDistance constructor: ", e.what());
            throw;
        }
    }

    /*
     * Compute LCPprod distance between sequence i and sequence j.
     *
     * Forward (sign > 0): common prefix from start.
     * Reverse (sign < 0): common suffix from end (RLCPprod).
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

            // Step 2: accumulate A_prod over matched positions
            double A = 0.0;
            if (sign > 0) {
                for (int t = 0; t < k; t++) {
                    A += ptr_dur(i, t) * ptr_dur(j, t);
                }
            } else {
                for (int t = 0; t < k; t++) {
                    A += ptr_dur(i, len_i - 1 - t) * ptr_dur(j, len_j - 1 - t);
                }
            }

            // Step 3: raw distance (can be negative or exceed T_x+T_y)
            double raw = Tx + Ty - 2.0 * A;

            if (norm == 0) return raw;

            // Step 4: normalize and clamp to [0, 1] (Option 1 per requirements)
            double maxdist = Tx + Ty;
            double d = normalize_distance(raw, maxdist, Tx, Ty, norm);
            return std::max(0.0, std::min(1.0, d));
        } catch (const std::exception& e) {
            py::print("Error in LCPprodDistance::compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            return dp_utils::compute_all_distances_simple(
                    nseq,
                    dist_matrix,
                    [this](int i, int j) { return this->compute_distance(i, j); }
            );
        } catch (const std::exception& e) {
            py::print("Error in LCPprodDistance::compute_all_distances: ", e.what());
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
                    [this](int is, int rseq) { return this->compute_distance(is, rseq); }
            );
        } catch (const std::exception& e) {
            py::print("Error in LCPprodDistance::compute_refseq_distances: ", e.what());
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
    py::array_t<double> dist_matrix;

    int nans;
    int rseq1;
    int rseq2;
    py::array_t<double> refdist_matrix;
};
