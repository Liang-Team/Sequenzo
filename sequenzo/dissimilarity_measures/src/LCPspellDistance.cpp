/*
 * LCPspellDistance: Spell-based Longest Common Prefix distance.
 *
 * Unlike position-wise LCP (which compares state at the same time index),
 * LCPspell compares sequences spell-by-spell: the k-th spell of sequence A
 * is compared with the k-th spell of sequence B. Two spells "match" if they
 * have the same state; we do not require the same start time (e.g. "state 1
 * from 2000" and "state 1 from 2005" both count as the same spell state).
 *
 * expcost (timecost in C++):
 *   - expcost = 0: ignore duration; only state equality matters (same state
 *     in the same spell order gives a match regardless of spell length).
 *   - expcost > 0: duration-aware; when two spells have the same state, we
 *     add a penalty proportional to |dur_A - dur_B|, similar to OMspell.
 *     Larger expcost makes "same state, different length" more distant.
 *
 * Usage (Python):
 *   from sequenzo import load_dataset, SequenceData, get_distance_matrix
 *
 *   seqdata = SequenceData(df, time=time_list, id_col="country",
 *                         states=states, labels=states)
 *
 *   # State-only: ignore duration (expcost=0)
 *   d = get_distance_matrix(seqdata, method="LCPspell", norm="gmean", expcost=0)
 *
 *   # Duration-aware: same state but different length adds distance (like OMspell)
 *   d2 = get_distance_matrix(seqdata, method="LCPspell", norm="gmean", expcost=0.5)
 *
 *   # Reverse: compare from the last spell (RLCPspell)
 *   d3 = get_distance_matrix(seqdata, method="RLCPspell", norm="gmean", expcost=0.5)
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : LCPspellDistance.cpp
 * @Time    : 2026/1/29 22:42
 * @Desc    : Spell-based Longest Common Prefix distance.
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
     *   state of each spell for sequence i (only positions 0..seqlength(i)-1
     *   are valid).
     * - seqdur: spell durations, shape (nseq, max_spells).
     * - seqlength: number of spells per sequence, shape (nseq,).
     * - norm: normalization index (see utils.h).
     * - sign: 1 = forward LCPspell (compare from first spell), -1 = reverse
     *   RLCPspell (compare from last spell).
     * - refseqS: reference sequence indices [rseq1, rseq2).
     * - timecost: expcost from Python. 0 = ignore duration (state-only match);
     *   >0 = add penalty proportional to |dur_A - dur_B| on matched spells (like OMspell).
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

            // Compute maximum duration over all valid spell positions (used for
            // normalizing the duration-penalty term when timecost > 0).
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
     *   and count how many consecutive spells have the same state.
     * - Reverse (sign < 0): compare last spell with last spell, then
     *   second-to-last with second-to-last, ...
     * For each matched spell we add timecost * |dur_A - dur_B| to the
     * distance (duration penalty). When timecost == 0, only state equality
     * matters.
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
                double d = normalize_distance(raw, maxdist, static_cast<double>(n), static_cast<double>(m), norm);
                return (d < 0.0) ? 0.0 : (d > 1.0 ? 1.0 : d);
            }

            int L = 0;           // length of spell-based common prefix
            double duration_penalty = 0.0;

            if (sign > 0) {
                // Forward: compare first spell with first spell, second with second, ...
                while (L < min_nm && ptr_seq(is, L) == ptr_seq(js, L)) {
                    duration_penalty += std::fabs(ptr_dur(is, L) - ptr_dur(js, L));
                    L++;
                }
            } else {
                // Reverse: compare last spell with last spell, then second-to-last, ...
                while (L < min_nm && ptr_seq(is, n - 1 - L) == ptr_seq(js, m - 1 - L)) {
                    duration_penalty += std::fabs(ptr_dur(is, n - 1 - L) - ptr_dur(js, m - 1 - L));
                    L++;
                }
            }

            double raw = (n + m - 2.0 * L) + timecost * duration_penalty;
            // Use same maxdist as position-wise LCP (n+m) so normalization is comparable;
            // raw can exceed maxdist when timecost > 0, so we clamp the result to [0, 1].
            double maxdist = static_cast<double>(n + m);
            double d = normalize_distance(raw, maxdist, static_cast<double>(n), static_cast<double>(m), norm);
            return (d < 0.0) ? 0.0 : (d > 1.0 ? 1.0 : d);
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
