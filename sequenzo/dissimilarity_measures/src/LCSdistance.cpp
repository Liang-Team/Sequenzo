#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif
#if defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace py = pybind11;

class LCSdistance {
private:
    struct BitWorkspace {
        int words = 0;
        int states = 0;
        std::vector<std::uint64_t> masks;
        std::vector<std::uint64_t> bits;
        std::vector<unsigned char> active;
        std::vector<int> dirty_states;

        void reset(int requested_words, int requested_states) {
            if (words != requested_words || states != requested_states) {
                words = requested_words;
                states = requested_states;
                masks.resize(static_cast<size_t>(states + 1) * static_cast<size_t>(words));
                bits.resize(static_cast<size_t>(words));
                active.assign(static_cast<size_t>(states + 1), 0);
                dirty_states.clear();
                std::fill(masks.begin(), masks.end(), 0ULL);
            } else {
                for (int state : dirty_states) {
                    std::fill(
                        masks.begin() + static_cast<size_t>(state) * words,
                        masks.begin() + static_cast<size_t>(state + 1) * words,
                        0ULL
                    );
                    active[state] = 0;
                }
                dirty_states.clear();
            }
            std::fill(bits.begin(), bits.end(), 0ULL);
        }
    };

public:
    LCSdistance(py::array_t<int> sequences,
                py::array_t<int> seqlength,
                int norm,
                py::array_t<int> refseqS,
                double distance_scale = 1.0)
            : norm(norm), distance_scale(distance_scale) {
        py::print("[>] Starting Longest Common Subsequence (LCS)...");
        std::cout << std::flush;

        this->sequences = sequences;
        this->seqlength = seqlength;

        auto seq_shape = sequences.shape();
        nseq = static_cast<int>(seq_shape[0]);
        maxlen = static_cast<int>(seq_shape[1]);

        seq_ptr = sequences.data();
        len_ptr = seqlength.data();
        max_state = find_max_state();

        rseq1 = refseqS.at(0);
        rseq2 = refseqS.at(1);
        if (rseq1 < rseq2) {
            nseq = rseq1;
        } else {
            rseq1 = rseq1 - 1;
        }
    }

    double compute_distance(int is, int js, BitWorkspace& work) const {
        const int m = len_ptr[is];
        const int n = len_ptr[js];

        const double ml = distance_scale * static_cast<double>(m);
        const double nl = distance_scale * static_cast<double>(n);

        if (m == 0 && n == 0)
            return normalize_distance(0.0, 0.0, 0.0, 0.0, norm);
        if (m == 0 || n == 0) {
            const double raw = distance_scale * static_cast<double>(m + n);
            return normalize_distance(raw, raw, ml, nl, norm);
        }

        const int* row_i = seq_ptr + static_cast<ptrdiff_t>(is) * maxlen;
        const int* row_j = seq_ptr + static_cast<ptrdiff_t>(js) * maxlen;

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
        const int middle_lcs = (m_mid == 0 || n_mid == 0)
            ? 0
            : lcs_bitparallel(row_i + prefix, m_mid, row_j + prefix, n_mid, work);

        const int lcs = prefix + suffix + middle_lcs;
        const double raw = distance_scale * static_cast<double>(m + n - 2 * lcs);
        const double maxdist = distance_scale * static_cast<double>(m + n);
        return normalize_distance(raw, maxdist, ml, nl, norm);
    }

    py::array_t<double> compute_all_distances() {
        auto dist_matrix = py::array_t<double>({nseq, nseq});
        auto buffer = dist_matrix.mutable_unchecked<2>();
        const int row_chunk = dp_utils::pairwise_row_chunk(nseq);

        #pragma omp parallel
        {
            BitWorkspace work;

            #pragma omp for schedule(dynamic, row_chunk)
            for (int i = 0; i < nseq; i++) {
                buffer(i, i) = 0.0;
                for (int j = i + 1; j < nseq; j++) {
                    buffer(i, j) = compute_distance(i, j, work);
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nseq; i++) {
            for (int j = i + 1; j < nseq; j++) {
                buffer(j, i) = buffer(i, j);
            }
        }

        return dist_matrix;
    }

    py::array_t<double> compute_condensed_distances() {
        const long long condensed_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
        py::array_t<double> condensed(
            std::array<py::ssize_t, 1>{static_cast<py::ssize_t>(condensed_len)}
        );
        auto buffer = condensed.mutable_unchecked<1>();
        const int row_chunk = dp_utils::pairwise_row_chunk(nseq);

        #pragma omp parallel
        {
            BitWorkspace work;

            #pragma omp for schedule(dynamic, row_chunk)
            for (int i = 0; i < nseq - 1; i++) {
                for (int j = i + 1; j < nseq; j++) {
                    buffer(dp_utils::condensed_index(nseq, i, j)) = compute_distance(i, j, work);
                }
            }
        }

        return condensed;
    }

    py::array_t<double> compute_refseq_distances() {
        auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
        auto buffer = refdist_matrix.mutable_unchecked<2>();
        const int nref = rseq2 - rseq1;
        const long long total = static_cast<long long>(nseq) * static_cast<long long>(nref);

        #pragma omp parallel
        {
            BitWorkspace work;

            #pragma omp for schedule(dynamic, 16)
            for (long long idx = 0; idx < total; idx++) {
                const int is = static_cast<int>(idx / nref);
                const int col = static_cast<int>(idx % nref);
                const int rseq = rseq1 + col;
                buffer(is, col) = (is == rseq) ? 0.0 : compute_distance(is, rseq, work);
            }
        }

        return refdist_matrix;
    }

private:
    py::array_t<int> sequences;
    py::array_t<int> seqlength;
    int norm;
    double distance_scale;
    int nseq;
    int maxlen;
    int max_state;
    const int* seq_ptr = nullptr;
    const int* len_ptr = nullptr;
    int rseq1;
    int rseq2;

    int find_max_state() const {
        int result = 0;
        const int original_nseq = static_cast<int>(sequences.shape(0));
        for (int i = 0; i < original_nseq; i++) {
            const int* row = seq_ptr + static_cast<ptrdiff_t>(i) * maxlen;
            for (int j = 0; j < len_ptr[i]; j++) {
                if (row[j] > result) {
                    result = row[j];
                }
            }
        }
        return result;
    }

    static int popcount64(std::uint64_t value) {
    #if defined(_MSC_VER)
        return static_cast<int>(__popcnt64(value));
    #else
        return __builtin_popcountll(value);
    #endif
    }

    int lcs_bitparallel(const int* left, int left_len, const int* right, int right_len, BitWorkspace& work) const {
        if (right_len > left_len) {
            std::swap(left, right);
            std::swap(left_len, right_len);
        }

        const int words = (right_len + 63) / 64;
        work.reset(words, max_state);

        for (int pos = 0; pos < right_len; pos++) {
            const int state = right[pos];
            if (state >= 0 && state <= max_state) {
                if (!work.active[state]) {
                    work.active[state] = 1;
                    work.dirty_states.push_back(state);
                }
                work.masks[static_cast<size_t>(state) * words + pos / 64] |= (1ULL << (pos & 63));
            }
        }

        const std::uint64_t last_mask = (right_len & 63)
            ? ((1ULL << (right_len & 63)) - 1ULL)
            : ~0ULL;

        for (int i = 0; i < left_len; i++) {
            const int state = left[i];
            const std::uint64_t* mask = (state >= 0 && state <= max_state)
                ? work.masks.data() + static_cast<size_t>(state) * words
                : nullptr;
            std::uint64_t shift_carry = 1ULL;
            std::uint64_t borrow = 0ULL;

            for (int w = 0; w < words; w++) {
                const std::uint64_t old_bits = work.bits[w];
                const std::uint64_t matches = (mask == nullptr) ? 0ULL : mask[w];
                const std::uint64_t x = old_bits | matches;
                const std::uint64_t y = (old_bits << 1) | shift_carry;
                shift_carry = old_bits >> 63;

                const std::uint64_t subtrahend = y + borrow;
                const std::uint64_t next_borrow = (subtrahend < y || x < subtrahend) ? 1ULL : 0ULL;
                const std::uint64_t diff = x - subtrahend;
                work.bits[w] = x & ~diff;
                borrow = next_borrow;
            }

            work.bits[words - 1] &= last_mask;
        }

        int lcs = 0;
        for (int w = 0; w < words; w++) {
            lcs += popcount64(work.bits[w]);
        }
        return lcs;
    }
};
