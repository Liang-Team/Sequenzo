/*
 * OMlocDistance: Optimal Matching with localized indel costs (context-dependent).
 *
 * Optimizations vs original:
 *   [OPT-1] Inlined getIndel into compute_distance hot loop. Original created new
 *           unchecked<>() accessors per getIndel call (~O(m*n + m²) calls per pair).
 *           Now uses lambda capturing accessors created once at function entry.
 *   [OPT-2] curr[0] incremental accumulation: O(m²) → O(m).
 *           Original: for each row i, recomputed sum from k=0..i-1.
 *           Key insight: context (prev_jstate, j_state) stabilizes after i=1's j-loop
 *           to (seq_j[n-1], seq_j[n-1]). So i=1 uses initial context (1 call),
 *           i=2 recomputes 2 terms with stable context, i≥3 increments by 1 term.
 *   [OPT-3] Manual 3-way min replaces std::min({a,b,c}).
 *
 * @Author  : Yuqi Liang 梁彧祺, Yapeng Wei 卫亚鹏
 * @File    : OMlocDistance.cpp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class OMlocDistance {
public:
    OMlocDistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, 
                  py::array_t<int> seqlength, py::array_t<int> refseqS,
                  double expcost, double context, py::array_t<double> indellist = py::array_t<double>())
            : indel(indel), norm(norm) {

        py::print("[>] Starting Optimal Matching with localized indel costs (OMloc)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->sm = sm;
            this->seqlength = seqlength;

            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            seqlen = seq_shape[1];
            alphasize = sm.shape()[0];

            use_indellist = (indellist.size() == static_cast<py::ssize_t>(alphasize) || indellist.size() == static_cast<py::ssize_t>(alphasize - 1));
            indellist_0based = (indellist.size() == static_cast<py::ssize_t>(alphasize - 1));
            if (use_indellist)
                this->indellist = indellist;

            dist_matrix = py::array_t<double>({nseq, nseq});
            fmatsize = use_indellist ? (seqlen + 2) : (seqlen + 1);

            auto ptr = sm.mutable_unchecked<2>();
            maxscost = 0.0;
            if (norm == 4) {
                maxscost = 2 * indel;
            } else {
                for (int i = 0; i < alphasize; i++)
                    for (int j = i + 1; j < alphasize; j++)
                        if (ptr(i, j) > maxscost)
                            maxscost = ptr(i, j);
                maxscost = std::min(maxscost, 2 * indel);
            }

            timecost = expcost * maxscost;
            localcost = context;

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
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    // Convenience function (kept for external callers).
    inline double getIndel(int state, int prev, int next) {
        if (use_indellist) {
            auto ptr = indellist.unchecked<1>();
            int idx = indellist_0based ? (state - 1) : state;
            return ptr(idx);
        }
        auto ptr_sm = sm.unchecked<2>();
        return timecost + localcost * (ptr_sm(prev, state) + ptr_sm(next, state)) / 2.0;
    }

    double compute_distance(int is, int js, double* prev, double* curr) {
        try {
            auto ptr_len = seqlength.unchecked<1>();
            int m_full = ptr_len(is);
            int n_full = ptr_len(js);

            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_sm = sm.unchecked<2>();

            int m = m_full;
            int n = n_full;

            if (m == 0 && n == 0)
                return normalize_distance(0.0, 0.0, 0.0, 0.0, norm);
            if (m == 0) {
                double cost = double(n) * indel;
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, 0.0, double(n) * indel, norm);
            }
            if (n == 0) {
                double cost = double(m) * indel;
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, double(m) * indel, 0.0, norm);
            }

            // [OPT-1] Inline getIndel: pre-extract indellist data pointer for 1D case.
            const double* indel_data = use_indellist ? static_cast<const double*>(indellist.data()) : nullptr;

            auto indel_fn = [&](int state, int prev_s, int next_s) -> double {
                if (indel_data) {
                    int idx = indellist_0based ? (state - 1) : state;
                    return indel_data[idx];
                }
                return timecost + localcost * (ptr_sm(prev_s, state) + ptr_sm(next_s, state)) / 2.0;
            };

            int prefix = 0;
            int firststate = std::max(prefix - 1, 0);

            // First column F[i][0]: deletions from sequence is
            int prev_jstate = ptr_seq(js, prefix);
            int j_state = ptr_seq(js, firststate);
            prev[0] = 0.0;
            for (int ii = 1; ii <= m; ++ii) {
                prev[ii] = prev[ii - 1] + indel_fn(ptr_seq(is, ii - 1), prev_jstate, j_state);
            }

            // First row F[0][j]: insertions into sequence js
            int prev_istate = ptr_seq(is, prefix);
            int i_state = ptr_seq(is, firststate);
            curr[0] = 0.0;
            for (int jj = 1; jj <= n; ++jj) {
                curr[jj] = curr[jj - 1] + indel_fn(ptr_seq(js, jj - 1), prev_istate, i_state);
            }

            std::swap(prev, curr);
            // Now prev[0..n] = first row

            // Reset context for main DP loop
            prev_istate = ptr_seq(is, firststate);
            i_state = ptr_seq(is, prefix);
            prev_jstate = ptr_seq(js, prefix);
            j_state = ptr_seq(js, firststate);

            // =====================================================================
            // [OPT-2] Main DP loop with incremental curr[0].
            //
            // Context (prev_jstate, j_state) used for curr[0]:
            //   i=1: initial context (seq_j[0], seq_j[0])
            //   i≥2: stable context (seq_j[n-1], seq_j[n-1]) — set after i=1's j-loop
            // =====================================================================
            double cum_indel_stable = 0.0;
            int stable_prev_j = 0, stable_j = 0;

            for (int i = 1; i <= m; ++i) {
                int i_state_curr = ptr_seq(is, i - 1);

                if (i == 1) {
                    // i=1: 1 term, initial context
                    curr[0] = indel_fn(ptr_seq(is, 0), prev_jstate, j_state);
                } else if (i == 2) {
                    // i=2: recompute 2 terms with stable context
                    cum_indel_stable = indel_fn(ptr_seq(is, 0), stable_prev_j, stable_j)
                                     + indel_fn(ptr_seq(is, 1), stable_prev_j, stable_j);
                    curr[0] = cum_indel_stable;
                } else {
                    // i≥3: add 1 term
                    cum_indel_stable += indel_fn(ptr_seq(is, i - 1), stable_prev_j, stable_j);
                    curr[0] = cum_indel_stable;
                }

                for (int j = 1; j <= n; ++j) {
                    int j_state_curr = ptr_seq(js, j - 1);

                    double del_cost = prev[j] + indel_fn(i_state_curr, prev_jstate, j_state);
                    double ins_cost = curr[j - 1] + indel_fn(j_state_curr, prev_istate, i_state_curr);
                    double sub_cost = (i_state_curr == j_state_curr)
                        ? prev[j - 1] : prev[j - 1] + ptr_sm(i_state_curr, j_state_curr);

                    // [OPT-3] Manual 3-way min.
                    double best = del_cost;
                    if (ins_cost < best) best = ins_cost;
                    if (sub_cost < best) best = sub_cost;
                    curr[j] = best;

                    prev_jstate = j_state_curr;
                    j_state = (j < n_full) ? ptr_seq(js, j) : ptr_seq(js, j - 1);
                }

                // Save stable context after i=1's j-loop
                if (i == 1) {
                    stable_prev_j = prev_jstate;
                    stable_j = j_state;
                }

                prev_istate = i_state_curr;
                i_state = (i < m_full) ? ptr_seq(is, i) : ptr_seq(is, i - 1);
                std::swap(prev, curr);
            }

            double final_cost = prev[n];
            double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
            double ml = double(m) * indel;
            double nl = double(n) * indel;
            return normalize_distance(final_cost, maxpossiblecost, ml, nl, norm);

        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            return dp_utils::compute_all_distances(
                nseq, fmatsize, dist_matrix,
                [this](int i, int j, double* prev, double* curr) {
                    return this->compute_distance(i, j, prev, curr);
                });
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto buffer = refdist_matrix.mutable_unchecked<2>();
            #pragma omp parallel
            {
                double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
                double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize));
                #pragma omp for schedule(static)
                for (int rseq = rseq1; rseq < rseq2; rseq++) {
                    for (int is = 0; is < nseq; is++) {
                        buffer(is, rseq - rseq1) = (is != rseq)
                            ? compute_distance(is, rseq, prev, curr) : 0.0;
                    }
                }
                dp_utils::aligned_free_double(prev);
                dp_utils::aligned_free_double(curr);
            }
            return refdist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<double> sm;
    double indel;
    int norm;
    bool use_indellist = false;
    bool indellist_0based = false;
    py::array_t<double> indellist;
    int nseq;
    int seqlen;
    int alphasize;
    int fmatsize;
    py::array_t<int> seqlength;
    py::array_t<double> dist_matrix;
    double maxscost = 0.0;
    double timecost;
    double localcost;
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};
