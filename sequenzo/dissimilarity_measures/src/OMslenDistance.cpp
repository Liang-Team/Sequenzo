/*
 * OMslenDistance: Optimal Matching with spell-length sensitivity.
 *
 * Optimizations vs original:
 *   [OPT-1] Removed pseudo-SIMD (same rationale as OMspell).
 *   [OPT-2] Manual 3-way min replaces std::min({a,b,c}).
 *   [OPT-3] Inlined getIndel/getSubCost into compute_distance hot loop.
 *           Original created new unchecked<>() accessors per call (~O(m*n) calls
 *           in inner loop, each constructing proxy objects). Now reuses accessors
 *           created once at top of compute_distance.
 *
 * @Author  : Yuqi Liang 梁彧祺, Yapeng Wei 卫亚鹏
 * @File    : OMslenDistance.cpp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class OMslenDistance {
public:
    OMslenDistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, 
                   py::array_t<int> refseqS, py::array_t<double> seqdur, py::array_t<double> indellist,
                   int sublink, py::array_t<int> seqlength)
            : indel(indel), norm(norm), sublink(sublink) {

        py::print("[>] Starting Optimal Matching with spell length (OMslen)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->sm = sm;
            this->seqdur = seqdur;
            this->indellist = indellist;
            this->seqlength = seqlength;

            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            seqlen = seq_shape[1];
            alphasize = sm.shape()[0];

            dist_matrix = py::array_t<double>({nseq, nseq});
            fmatsize = seqlen + 1;

            auto ptr = sm.mutable_unchecked<2>();
            if (norm == 4) {
                maxscost = 2 * indel;
            } else {
                for (int i = 0; i < alphasize; i++)
                    for (int j = i + 1; j < alphasize; j++)
                        if (ptr(i, j) > maxscost)
                            maxscost = ptr(i, j);
                maxscost = std::min(maxscost, 2 * indel);
            }

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

    // Convenience functions (kept for external callers; hot path uses inlined versions).
    inline double getIndel(int seq_idx, int position, int state) {
        auto ptr_indel = indellist.unchecked<1>();
        auto ptr_dur = seqdur.unchecked<2>();
        return ptr_indel(state) * ptr_dur(seq_idx, position);
    }
    inline double getSubCost(int i_state, int j_state, int i_seq_idx, int i_pos, int j_seq_idx, int j_pos) {
        auto ptr_sm = sm.unchecked<2>();
        auto ptr_dur = seqdur.unchecked<2>();
        if (i_state == j_state) return 0.0;
        double base_cost = ptr_sm(i_state, j_state);
        double dur_i = ptr_dur(i_seq_idx, i_pos);
        double dur_j = ptr_dur(j_seq_idx, j_pos);
        if (sublink == 1) return base_cost * (dur_i + dur_j);
        else return base_cost * std::sqrt(dur_i * dur_j);
    }

    double compute_distance(int is, int js, double* prev, double* curr) {
        try {
            auto ptr_len = seqlength.unchecked<1>();
            int m_full = ptr_len(is);
            int n_full = ptr_len(js);
            int mSuf = m_full + 1, nSuf = n_full + 1;
            int prefix = 0;

            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_sm = sm.unchecked<2>();
            auto ptr_dur = seqdur.unchecked<2>();
            auto ptr_indel = indellist.unchecked<1>();

            // [OPT-3] Inline helpers using the accessors above.
            auto indel_cost = [&](int seq_idx, int position, int state) -> double {
                return ptr_indel(state) * ptr_dur(seq_idx, position);
            };
            auto sub_cost_fn = [&](int ist, int jst, int iseq, int ipos, int jseq, int jpos) -> double {
                if (ist == jst) return 0.0;
                double bc = ptr_sm(ist, jst);
                double di = ptr_dur(iseq, ipos);
                double dj = ptr_dur(jseq, jpos);
                return (sublink == 1) ? bc * (di + dj) : bc * std::sqrt(di * dj);
            };

            // Skipping common prefix
            int ii = 1, jj = 1;
            while (ii < mSuf && jj < nSuf && ptr_seq(is, ii - 1) == ptr_seq(js, jj - 1)) {
                ii++; jj++; prefix++;
            }
            // Skipping common suffix
            while (mSuf > ii && nSuf > jj && ptr_seq(is, mSuf - 2) == ptr_seq(js, nSuf - 2)) {
                mSuf--; nSuf--;
            }

            int m = mSuf - prefix;
            int n = nSuf - prefix;

            if (m == 0 && n == 0)
                return normalize_distance(0.0, 0.0, 0.0, 0.0, norm);
            if (m == 0) {
                double cost = 0.0;
                for (int j = prefix; j < n_full; j++)
                    cost += indel_cost(js, j, ptr_seq(js, j));
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, 0.0, cost, norm);
            }
            if (n == 0) {
                double cost = 0.0;
                for (int i = prefix; i < m_full; i++)
                    cost += indel_cost(is, i, ptr_seq(is, i));
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, cost, 0.0, norm);
            }

            // Initialize first row
            prev[0] = 0.0;
            for (int j = prefix; j < n_full; j++)
                prev[j - prefix + 1] = prev[j - prefix] + indel_cost(js, j, ptr_seq(js, j));

            // [OPT-1] Pure scalar DP — pseudo-SIMD removed.
            for (int i = prefix; i < m_full; i++) {
                int i_state = ptr_seq(is, i);
                double del_cost_i = indel_cost(is, i, i_state);

                curr[0] = prev[0] + del_cost_i;

                for (int j = prefix + 1; j < nSuf; j++) {
                    int jj_idx = j - 1;
                    int j_state = ptr_seq(js, jj_idx);

                    double delcost = prev[j - prefix] + del_cost_i;
                    double inscost = curr[j - 1 - prefix] + indel_cost(js, jj_idx, j_state);
                    double subcost = prev[j - 1 - prefix] + sub_cost_fn(i_state, j_state, is, i, js, jj_idx);

                    // [OPT-2] Manual 3-way min.
                    double best = delcost;
                    if (inscost < best) best = inscost;
                    if (subcost < best) best = subcost;
                    curr[j - prefix] = best;
                }
                std::swap(prev, curr);
            }

            double final_cost = prev[nSuf - 1 - prefix];

            // Normalization constants
            double maxpossiblecost = 0.0;
            for (int i = prefix; i < m_full; i++)
                maxpossiblecost += indel_cost(is, i, ptr_seq(is, i));
            for (int j = prefix; j < n_full; j++)
                maxpossiblecost += indel_cost(js, j, ptr_seq(js, j));
            maxpossiblecost = std::min(maxpossiblecost, std::abs(n - m) * indel + maxscost * std::min(m, n));

            double ml = 0.0, nl = 0.0;
            for (int i = prefix; i < m_full; i++)
                ml += indel_cost(is, i, ptr_seq(is, i));
            for (int j = prefix; j < n_full; j++)
                nl += indel_cost(js, j, ptr_seq(js, j));

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
    int nseq;
    int seqlen;
    int alphasize;
    int fmatsize;
    py::array_t<int> seqlength;
    py::array_t<double> dist_matrix;
    double maxscost = 0.0;
    py::array_t<double> seqdur;
    py::array_t<double> indellist;
    int sublink;
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};
