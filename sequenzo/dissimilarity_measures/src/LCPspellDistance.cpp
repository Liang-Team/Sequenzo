/*
 * LCPspellDistance: Spell-based Longest Common Prefix distance.
 *
 * Unlike position-wise LCP (which compares state at the same time index),
 * LCPspell compares sequences spell-by-spell: the k-th spell of sequence A
 * is compared with the k-th spell of sequence B. Two spells "match" if they
 * have the same state; we do not require the same start time (e.g. "state 1
 * from 2000" and "state 1 from 2005" both count as the same spell state).
 *
 * Formula:
 *   d_raw = (n + m - 2*L) + timecost * duration_penalty_norm
 * where:
 *   - n, m = number of spells in sequence A and B.
 *   - L = number of matched spells (from first spell onward, same state counts as match).
 *   - duration_penalty_norm = sum over matched spells of |dur_A(k) - dur_B(k)| / max_dur
 *     (dimensionless; max_dur = max duration over all spells in the dataset).
 *   - timecost (duration_weight): relative weight for duration differences; independent
 *     of duration units (month/year) once normalized by max_dur.
 *
 * maxdist = (n + m) + timecost * min(n, m), so normalized distance d = d_raw / maxdist
 * lies in [0, 1] without clamping.
 *
 * timecost (Python expcost):
 *   - timecost = 0: ignore duration; only spell count and state order matter.
 *   - timecost > 0: duration-aware; same state but different length adds normalized penalty.
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : LCPspellDistance.cpp
 * @Time    : 2026/1/29 22:42
 * @Desc    : Spell-based Longest Common Prefix distance (with max_dur normalization).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

class LCPspellDistance {
public:
    /*
     * Constructor.
     * - sequences: spell states, shape (nseq, max_spells); row i holds the
     *   state of each spell for sequence i (only positions 0..seqlength(i)-1 valid).
     * - seqdur: spell durations, shape (nseq, max_spells).
     * - seqlength: number of spells per sequence, shape (nseq,).
     * - norm: normalization index (see utils.h).
     * - sign: 1 = forward LCPspell, -1 = reverse RLCPspell.
     * - refseqS: reference sequence indices [rseq1, rseq2).
     * - timecost: duration_weight (Python expcost). 0 = ignore duration;
     *   >0 = add dimensionless penalty sum(|dur_A-dur_B|/max_dur) on matched spells.
     */
    LCPspellDistance(py::array_t<int> sequences,
                     py::array_t<double> seqdur,
                     py::array_t<int> seqlength,
                     int norm,
                     int sign,
                     py::array_t<int> refseqS,
                     double timecost)
            : norm(norm), sign(sign), timecost(timecost) {
        py::print("[>] Starting (Reverse) Longest Common Prefix on spells (LCPspell/RLCPspell)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->seqdur = seqdur;
            this->seqlength = seqlength;

            auto seq_shape = sequences.shape();
            nseq = static_cast<int>(seq_shape[0]);
            max_spells = static_cast<int>(seq_shape[1]);

            dist_matrix = py::array_t<double>({nseq, nseq});

            // Maximum duration over all valid spells: used to normalize duration
            // penalty so that timecost is unit-independent (dimensionless).
            max_dur = 0.0;
            auto ptr_dur = seqdur.unchecked<2>();
            auto ptr_len = seqlength.unchecked<1>();
            for (int i = 0; i < nseq; i++) {
                int len_i = ptr_len(i);
                for (int k = 0; k < len_i && k < max_spells; k++) {
                    double d = ptr_dur(i, k);
                    if (d > max_dur) max_dur = d;
                }
            }

            // Reference sequence range (same convention as LCPdistance / OMspell).
            // max_dur is used in compute_distance to form duration_penalty_norm.
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
            py::print("Error in LCPspellDistance constructor: ", e.what());
            throw;
        }
    }

    /*
     * Compute spell-based LCP distance between sequence is and sequence js.
     * - Forward (sign > 0): compare spell 0 with spell 0, spell 1 with 1, ...
     * - Reverse (sign < 0): compare last spell with last, second-to-last with second-to-last, ...
     * Raw distance: (n + m - 2*L) + timecost * duration_penalty_norm, where
     * duration_penalty_norm = sum of |dur_A(k)-dur_B(k)| over matched spells, divided by max_dur
     * (dimensionless; 0 when max_dur == 0). maxdist includes duration upper bound so d stays in [0,1].
     */
    double compute_distance(int is, int js) {
        try {
            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_dur = seqdur.unchecked<2>();
            auto ptr_len = seqlength.unchecked<1>();

            int n = ptr_len(is);
            int m = ptr_len(js);
            int min_nm = (n < m) ? n : m;

            if (min_nm == 0) {
                double raw = static_cast<double>(n + m);
                double maxdist = raw;
                return normalize_distance(raw, maxdist, static_cast<double>(n), static_cast<double>(m), norm);
            }

            int L = 0;
            double duration_penalty_sum = 0.0;

            if (sign > 0) {
                while (L < min_nm && ptr_seq(is, L) == ptr_seq(js, L)) {
                    duration_penalty_sum += std::fabs(ptr_dur(is, L) - ptr_dur(js, L));
                    L++;
                }
            } else {
                while (L < min_nm && ptr_seq(is, n - 1 - L) == ptr_seq(js, m - 1 - L)) {
                    duration_penalty_sum += std::fabs(ptr_dur(is, n - 1 - L) - ptr_dur(js, m - 1 - L));
                    L++;
                }
            }

            // Dimensionless duration penalty: sum |dur_A - dur_B| / max_dur (0 if max_dur == 0).
            double duration_penalty_norm = (max_dur > 0.0) ? (duration_penalty_sum / max_dur) : 0.0;
            double raw = (n + m - 2.0 * L) + timecost * duration_penalty_norm;
            // Upper bound so that d = raw/maxdist is in [0,1]: spell-count term <= n+m, duration term <= timecost*min(n,m).
            double maxdist = static_cast<double>(n + m) + timecost * static_cast<double>(min_nm);
            return normalize_distance(raw, maxdist, static_cast<double>(n), static_cast<double>(m), norm);
        } catch (const std::exception& e) {
            py::print("Error in LCPspellDistance::compute_distance: ", e.what());
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
            py::print("Error in LCPspellDistance::compute_all_distances: ", e.what());
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
            py::print("Error in LCPspellDistance::compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<double> seqdur;
    py::array_t<int> seqlength;
    int norm;
    int sign;
    double timecost;
    int nseq;
    int max_spells;
    double max_dur;
    py::array_t<double> dist_matrix;

    int nans;
    int rseq1;
    int rseq2;
    py::array_t<double> refdist_matrix;
};
