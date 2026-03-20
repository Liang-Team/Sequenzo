/*
 * LCSdistance: Longest Common Subsequence distance.
 *
 * Optimizations vs original:
 * 1. Raw pointer cache (eliminates pybind11 accessor overhead in O(L^2) loop)
 * 2. Pre-allocated per-thread DP buffers (eliminates ~50M heap alloc/dealloc at n=10000)
 * 3. Prefix/suffix trimming (reduces DP matrix dimensions for sequences sharing common ends)
 * 4. OpenMP parallel with dynamic scheduling (original was serial via compute_all_distances_simple)
 * 5. Removed try/catch from hot path
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstring>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class LCSdistance {
public:
    LCSdistance(py::array_t<int> sequences,
                py::array_t<int> seqlength,
                int norm,
                py::array_t<int> refseqS)
            : norm(norm) {
        py::print("[>] Starting Longest Common Subsequence (LCS)...");
        std::cout << std::flush;

        this->sequences = sequences;
        this->seqlength = seqlength;

        auto seq_shape = sequences.shape();
        nseq = static_cast<int>(seq_shape[0]);
        maxlen = static_cast<int>(seq_shape[1]);

        // [OPT-1] Cache raw pointers
        seq_ptr = sequences.data();
        len_ptr = seqlength.data();

        // DP buffer size: need (maxlen+1) ints for prev and curr rows
        fmatsize = maxlen + 1;

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
    }

    // [OPT-2] Takes pre-allocated int buffers from caller (per-thread).
    // Original allocated two std::vector<int>(n+1) per call = ~50M heap alloc/dealloc
    // pairs for n=10000 unique sequences.
    double compute_distance(int is, int js, int* __restrict__ prev, int* __restrict__ curr) {
        const int m = len_ptr[is];
        const int n = len_ptr[js];

        if (m == 0 && n == 0)
            return normalize_distance(0.0, 0.0, 0.0, 0.0, norm);
        if (m == 0 || n == 0) {
            double raw = static_cast<double>(m + n);
            return normalize_distance(raw, raw, static_cast<double>(m), static_cast<double>(n), norm);
        }

        const int* row_i = seq_ptr + static_cast<ptrdiff_t>(is) * maxlen;
        const int* row_j = seq_ptr + static_cast<ptrdiff_t>(js) * maxlen;

        // [OPT-3] Prefix/suffix trimming.
        // If sequences share common prefix of length k, those k positions each
        // contribute exactly 1 to LCS. Similarly for common suffix.
        // DP only runs on the middle portion: O((m-k-s) * (n-k-s)) instead of O(m*n).
        // For 20 random states: expected common prefix ~L/20, saves ~10% of DP.
        // For real-world data with common start/end states: much larger savings.
        int prefix = 0;
        const int min_mn = std::min(m, n);
        while (prefix < min_mn && row_i[prefix] == row_j[prefix]) {
            prefix++;
        }

        int suffix = 0;
        const int max_suffix = std::min(m - prefix, n - prefix);
        while (suffix < max_suffix && row_i[m - 1 - suffix] == row_j[n - 1 - suffix]) {
            suffix++;
        }

        const int m_mid = m - prefix - suffix;
        const int n_mid = n - prefix - suffix;

        if (m_mid == 0 || n_mid == 0) {
            // Entire LCS is prefix + suffix (no DP needed)
            int L = prefix + suffix;
            double raw = static_cast<double>(m + n - 2 * L);
            double maxdist = static_cast<double>(m + n);
            return normalize_distance(raw, maxdist, static_cast<double>(m), static_cast<double>(n), norm);
        }

        // DP on middle portion only
        const int* mid_i = row_i + prefix;
        const int* mid_j = row_j + prefix;

        // Zero-init prev row for middle portion
        std::memset(prev, 0, (n_mid + 1) * sizeof(int));

        for (int i = 1; i <= m_mid; i++) {
            curr[0] = 0;
            const int si = mid_i[i - 1];
            for (int j = 1; j <= n_mid; j++) {
                if (si == mid_j[j - 1]) {
                    curr[j] = 1 + prev[j - 1];
                } else {
                    int a = prev[j];
                    int b = curr[j - 1];
                    curr[j] = (a >= b) ? a : b;
                }
            }
            std::swap(prev, curr);
        }

        int L = prefix + suffix + prev[n_mid];
        double raw = static_cast<double>(m + n - 2 * L);
        double maxdist = static_cast<double>(m + n);
        return normalize_distance(raw, maxdist, static_cast<double>(m), static_cast<double>(n), norm);
    }

    // [OPT-4] Custom OpenMP loop with per-thread pre-allocated buffers.
    // Original used compute_all_distances_simple (no buffer support), forcing
    // heap allocation inside each compute_distance call.
    py::array_t<double> compute_all_distances() {
        auto buffer = dist_matrix.mutable_unchecked<2>();

        #pragma omp parallel
        {
            int* prev = new int[fmatsize]();
            int* curr = new int[fmatsize]();

            #pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < nseq; i++) {
                buffer(i, i) = 0.0;
                for (int j = i + 1; j < nseq; j++) {
                    buffer(i, j) = compute_distance(i, j, prev, curr);
                }
            }

            delete[] prev;
            delete[] curr;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nseq; i++) {
            for (int j = i + 1; j < nseq; j++) {
                buffer(j, i) = buffer(i, j);
            }
        }

        return dist_matrix;
    }

    py::array_t<double> compute_refseq_distances() {
        auto buffer = refdist_matrix.mutable_unchecked<2>();

        #pragma omp parallel
        {
            int* prev = new int[fmatsize]();
            int* curr = new int[fmatsize]();

            #pragma omp for schedule(dynamic, 4)
            for (int rseq = rseq1; rseq < rseq2; rseq++) {
                for (int is = 0; is < nseq; is++) {
                    buffer(is, rseq - rseq1) = (is == rseq) ? 0.0 : compute_distance(is, rseq, prev, curr);
                }
            }

            delete[] prev;
            delete[] curr;
        }

        return refdist_matrix;
    }

private:
    py::array_t<int> sequences;
    py::array_t<int> seqlength;
    int norm;
    int nseq;
    int maxlen;
    int fmatsize;
    const int* seq_ptr = nullptr;
    const int* len_ptr = nullptr;
    py::array_t<double> dist_matrix;

    int nans;
    int rseq1;
    int rseq2;
    py::array_t<double> refdist_matrix;
};
