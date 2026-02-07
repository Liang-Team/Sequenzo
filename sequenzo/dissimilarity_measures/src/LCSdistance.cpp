/*
 * LCSdistance: Longest Common Subsequence distance.
 *
 * Port of TraMineR's cLCS from src/ffunctions.c (seqLLCS) and seqdist(method="LCS").
 * For each pair of sequences we compute the length L of a longest common subsequence (LCS),
 * then the raw distance is:  raw = len_i + len_j - 2*L.
 * This matches the edit-distance interpretation with indel cost 1 and no substitutions
 * (only insertions/deletions). Normalization uses the same scheme as LCP/RLCP (e.g. gmean).
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : LCSdistance.cpp
 * @Time    : 2026/2/7 8:36
 * @Desc    : 
 * Reference: TraMineR src/ffunctions.c.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

class LCSdistance {
public:
    /*
     * Constructor.
     * - sequences: state sequences, shape (nseq, maxlen); only positions 0..seqlength(i)-1
     *   are valid for sequence i.
     * - seqlength: number of valid positions per sequence, shape (nseq,).
     * - norm: normalization index (see utils.h; 0=none, 1=maxlength, 2=gmean, 3=maxdist, 4=YujianBo).
     * - refseqS: reference sequence indices [rseq1, rseq2).
     */
    LCSdistance(py::array_t<int> sequences,
                py::array_t<int> seqlength,
                int norm,
                py::array_t<int> refseqS)
            : norm(norm) {
        py::print("[>] Starting Longest Common Subsequence (LCS)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->seqlength = seqlength;

            auto seq_shape = sequences.shape();
            nseq = static_cast<int>(seq_shape[0]);
            maxlen = static_cast<int>(seq_shape[1]);

            dist_matrix = py::array_t<double>({nseq, nseq});

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
            py::print("Error in LCSdistance constructor: ", e.what());
            throw;
        }
    }

    /*
     * Compute LCS length for two sequences (TraMineR cLCS logic).
     * iseq, jseq: row indices. m = length of iseq, n = length of jseq.
     * Returns length of a longest common subsequence (integer).
     */
    int compute_LCS_length(int is, int js) {
        auto ptr_seq = sequences.unchecked<2>();
        auto ptr_len = seqlength.unchecked<1>();

        const int m = ptr_len(is);
        const int n = ptr_len(js);
        if (m == 0 || n == 0) return 0;

        // DP table: L[i][j] = LCS length of iseq[0..i-1] and jseq[0..j-1].
        // We use two rows to save memory (current and previous).
        std::vector<int> prev(n + 1, 0);
        std::vector<int> curr(n + 1, 0);

        for (int i = 1; i <= m; i++) {
            int si = ptr_seq(is, i - 1);
            for (int j = 1; j <= n; j++) {
                int sj = ptr_seq(js, j - 1);
                if (si == sj) {
                    curr[j] = 1 + prev[j - 1];
                } else {
                    int a = prev[j];
                    int b = curr[j - 1];
                    curr[j] = (a >= b) ? a : b;
                }
            }
            prev.swap(curr);
        }
        return prev[n];
    }

    /*
     * Compute LCS distance between sequence is and sequence js.
     * Raw distance = len_i + len_j - 2*L (L = LCS length). Then normalize.
     */
    double compute_distance(int is, int js) {
        try {
            auto ptr_len = seqlength.unchecked<1>();
            int m = ptr_len(is);
            int n = ptr_len(js);

            if (m == 0 && n == 0) {
                return normalize_distance(0.0, 0.0, 0.0, 0.0, norm);
            }
            if (m == 0 || n == 0) {
                double raw = static_cast<double>(m + n);
                double maxdist = raw;
                return normalize_distance(raw, maxdist, static_cast<double>(m), static_cast<double>(n), norm);
            }

            int L = compute_LCS_length(is, js);
            double raw = static_cast<double>(m + n - 2 * L);
            double maxdist = static_cast<double>(m + n);
            return normalize_distance(raw, maxdist, static_cast<double>(m), static_cast<double>(n), norm);
        } catch (const std::exception& e) {
            py::print("Error in LCSdistance::compute_distance: ", e.what());
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
            py::print("Error in LCSdistance::compute_all_distances: ", e.what());
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
            py::print("Error in LCSdistance::compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<int> seqlength;
    int norm;
    int nseq;
    int maxlen;
    py::array_t<double> dist_matrix;

    int nans;
    int rseq1;
    int rseq2;
    py::array_t<double> refdist_matrix;
};
