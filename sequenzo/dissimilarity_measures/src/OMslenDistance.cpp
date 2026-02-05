/*
 * OMslenDistance: Optimal Matching with spell-length sensitivity.
 *
 * Unlike standard OM (which uses constant indel costs) or OMspell (which uses
 * spell-level durations), OMslen uses position-level durations: each position
 * in the sequence has an associated duration weight derived from the spell length
 * at that position. This makes the distance sensitive to how long each state
 * persists at each position.
 *
 * Formula:
 *   The dynamic programming recurrence uses:
 *   - Indel cost at position i: indellist[state] * seqdur[seq_idx][i]
 *   - Substitution cost between states at positions i and j:
 *     * If link == "mean": scost * (seqdur[i] + seqdur[j])
 *     * If link == "gmean": scost * sqrt(seqdur[i] * seqdur[j])
 *
 * where:
 *   - seqdur: duration matrix (dur.mat) where each position stores the duration
 *     weight of the spell at that position, raised to power (-h).
 *   - indellist: vector of state-dependent indel costs.
 *   - scost: base substitution cost matrix.
 *   - h: exponential weight parameter (h >= 0); applied as dur.mat ^ (-h).
 *   - link: linkage method ("mean" or "gmean") for combining durations in substitution.
 *
 * The duration matrix (dur.mat) is built by expanding spell durations to position
 * level: if a spell of state A has duration d, then positions 1..d in that spell
 * all get duration weight d^(-h).
 *
 * Normalization:
 *   Supports standard OM normalizations: "none", "maxlength", "gmean", "maxdist", "YujianBo".
 *   Default (auto) is "YujianBo" for OMslen.
 *
 * Parameters:
 *   - link ("mean" or "gmean"): how to combine durations in substitution cost.
 *     "mean" uses arithmetic mean (scost * (dur_i + dur_j)), "gmean" uses geometric mean.
 *   - h (>= 0): exponential weight for duration; higher h gives more weight to longer spells.
 *     Applied as dur.mat ^ (-h), so h=0 means no duration weighting, h>0 penalizes longer spells.
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : OMslenDistance.cpp
 * @Time    : 2026/2/5 13:10
 * @Desc    : Optimal Matching with spell-length sensitivity (position-level duration weighting).
 *            References: 
 *            OMVIdistance.h from TraMineR package. https://github.com/cran/TraMineR/blob/master/src/OMVIdistance.h
 *            OMVIdistance.cpp from TraMineR package. https://github.com/cran/TraMineR/blob/master/src/OMVIdistance.cpp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <xsimd/xsimd.hpp>
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
            // ============================================
            // parameter : sequences, sm, seqdur, indellist
            // ============================================
            this->sequences = sequences;
            this->sm = sm;
            this->seqdur = seqdur;
            this->indellist = indellist;
            this->seqlength = seqlength;

            // ====================================================
            // initialize nseq, seqlen, dist_matrix, fmatsize
            // ====================================================
            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            seqlen = seq_shape[1];
            alphasize = sm.shape()[0];

            dist_matrix = py::array_t<double>({nseq, nseq});

            fmatsize = seqlen + 1;

            // ==================
            // initialize maxcost
            // ==================
            auto ptr = sm.mutable_unchecked<2>();

            if(norm == 4){
                maxscost = 2 * indel;
            }else{
                for(int i = 0; i < alphasize; i++){
                    for(int j = i+1; j < alphasize; j++){
                        if(ptr(i, j) > maxscost){
                            maxscost = ptr(i, j);
                        }
                    }
                }
                maxscost = std::min(maxscost, 2 * indel);
            }

            // about reference sequences :
            nans = nseq;

            rseq1 = refseqS.at(0);
            rseq2 = refseqS.at(1);
            if(rseq1 < rseq2){
                nseq = rseq1;
                nans = nseq * (rseq2 - rseq1);
            }else{
                rseq1 = rseq1 - 1;
            }
            refdist_matrix = py::array_t<double>({nseq, (rseq2-rseq1)});
        } catch (const std::exception& e) {
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    // Get indel cost for a state at a specific position
    // indel cost = indellist[state] * seqdur[seq_idx][position]
    inline double getIndel(int seq_idx, int position, int state) {
        auto ptr_indel = indellist.unchecked<1>();
        auto ptr_dur = seqdur.unchecked<2>();
        
        return ptr_indel(state) * ptr_dur(seq_idx, position);
    }

    // Get substitution cost adjusted by duration and sublink
    // sublink == 1 (mean): scost * (seqdur[i] + seqdur[j])
    // sublink == 0 (gmean): scost * sqrt(seqdur[i] * seqdur[j])
    inline double getSubCost(int i_state, int j_state, int i_seq_idx, int i_pos, int j_seq_idx, int j_pos) {
        auto ptr_sm = sm.unchecked<2>();
        auto ptr_dur = seqdur.unchecked<2>();
        
        if (i_state == j_state) {
            return 0.0;
        }
        
        double base_cost = ptr_sm(i_state, j_state);
        double dur_i = ptr_dur(i_seq_idx, i_pos);
        double dur_j = ptr_dur(j_seq_idx, j_pos);
        
        if (sublink == 1) {
            // mean: scost * (seqdur[i] + seqdur[j])
            return base_cost * (dur_i + dur_j);
        } else {
            // gmean: scost * sqrt(seqdur[i] * seqdur[j])
            return base_cost * std::sqrt(dur_i * dur_j);
        }
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

            // Skipping common prefix
            int ii = 1, jj = 1;
            while (ii < mSuf && jj < nSuf && ptr_seq(is, ii-1) == ptr_seq(js, jj-1)) {
                ii++; jj++; prefix++;
            }
            // Skipping common suffix
            while (mSuf > ii && nSuf > jj && ptr_seq(is, mSuf - 2) == ptr_seq(js, nSuf - 2)) {
                mSuf--; nSuf--;
            }

            int m = mSuf - prefix;
            int n = nSuf - prefix;

            // 预处理
            if (m == 0 && n == 0)
                return normalize_distance(0.0, 0.0, 0.0, 0.0, norm);
            if (m == 0) {
                double cost = 0.0;
                for (int j = prefix; j < n_full; j++) {
                    int state = ptr_seq(js, j);
                    cost += getIndel(js, j, state);
                }
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, 0.0, cost, norm);
            }
            if (n == 0) {
                double cost = 0.0;
                for (int i = prefix; i < m_full; i++) {
                    int state = ptr_seq(is, i);
                    cost += getIndel(is, i, state);
                }
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, cost, 0.0, norm);
            }

            using batch_t = xsimd::batch<double>;
            constexpr std::size_t B = batch_t::size;

            // Initialize first row (insertions into js)
            prev[0] = 0.0;
            for (int j = prefix; j < n_full; j++) {
                int state = ptr_seq(js, j);
                prev[j - prefix + 1] = prev[j - prefix] + getIndel(js, j, state);
            }

            // Main DP loop
            for (int i = prefix; i < m_full; i++) {
                int i_state = ptr_seq(is, i);
                double del_cost_i = getIndel(is, i, i_state);

                // First column: cumulative deletions
                curr[0] = prev[0] + del_cost_i;

                int j = prefix + 1;
                for (; j + (int)B <= nSuf; j += (int)B) {
                    const double* prev_ptr = prev + (j - prefix);
                    const double* prevm1_ptr = prev + (j - 1 - prefix);

                    batch_t prevj = batch_t::load_unaligned(prev_ptr);
                    batch_t prevjm1 = batch_t::load_unaligned(prevm1_ptr);

                    // substitution costs
                    alignas(64) double subs[B];
                    alignas(64) double ins[B];
                    for (std::size_t b = 0; b < B; ++b) {
                        int jj_idx = j + int(b) - 1;
                        int j_state = ptr_seq(js, jj_idx);
                        
                        subs[b] = getSubCost(i_state, j_state, is, i, js, jj_idx);
                        ins[b] = getIndel(js, jj_idx, j_state);
                    }
                    batch_t sub_batch = batch_t::load_unaligned(subs);
                    batch_t ins_batch = batch_t::load_unaligned(ins);

                    // Vectorize independent candidates: del and sub
                    batch_t cand_del = prevj + batch_t(del_cost_i);
                    batch_t cand_sub = prevjm1 + sub_batch;
                    batch_t vert = xsimd::min(cand_del, cand_sub);

                    // Sequential propagation for insert dependencies
                    double running_ins = curr[j - prefix - 1] + ins[0];
                    for (std::size_t b = 0; b < B; ++b) {
                        double v = vert.get(b);
                        double c = std::min(v, running_ins);
                        curr[j + int(b) - prefix] = c;
                        if (b + 1 < B) running_ins = c + ins[b + 1];
                    }
                }

                // Complete the tail section
                for (; j < nSuf; ++j) {
                    int jj_idx = j - 1;
                    int j_state = ptr_seq(js, jj_idx);
                    
                    double delcost = prev[j - prefix] + del_cost_i;
                    double inscost = curr[j - 1 - prefix] + getIndel(js, jj_idx, j_state);
                    double subcost = prev[j - 1 - prefix] + getSubCost(i_state, j_state, is, i, js, jj_idx);
                    
                    curr[j - prefix] = std::min({ delcost, inscost, subcost });
                }

                std::swap(prev, curr);
            }

            double final_cost = prev[nSuf - 1 - prefix];
            
            // Calculate max possible cost for normalization
            double maxpossiblecost = 0.0;
            for (int i = prefix; i < m_full; i++) {
                int state = ptr_seq(is, i);
                maxpossiblecost += getIndel(is, i, state);
            }
            for (int j = prefix; j < n_full; j++) {
                int state = ptr_seq(js, j);
                maxpossiblecost += getIndel(js, j, state);
            }
            maxpossiblecost = std::min(maxpossiblecost, std::abs(n - m) * indel + maxscost * std::min(m, n));
            
            double ml = 0.0, nl = 0.0;
            for (int i = prefix; i < m_full; i++) {
                int state = ptr_seq(is, i);
                ml += getIndel(is, i, state);
            }
            for (int j = prefix; j < n_full; j++) {
                int state = ptr_seq(js, j);
                nl += getIndel(js, j, state);
            }
            
            return normalize_distance(final_cost, maxpossiblecost, ml, nl, norm);

        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            return dp_utils::compute_all_distances(
                nseq,
                fmatsize,
                dist_matrix,
                [this](int i, int j, double* prev, double* curr) {
                    return this->compute_distance(i, j, prev, curr);
                }
            );
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
                for (int rseq = rseq1; rseq < rseq2; rseq ++) {
                    for (int is = 0; is < nseq; is ++) {
                        double cmpres = 0;
                        if(is != rseq){
                            cmpres = compute_distance(is, rseq, prev, curr);
                        }

                        buffer(is, rseq - rseq1) = cmpres;
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
    double maxscost;

    // OMslen specific parameters
    py::array_t<double> seqdur;      // Duration matrix (position-level durations)
    py::array_t<double> indellist;   // Vector of indel costs per state
    int sublink;                      // 1=mean, 0=gmean

    // about reference sequences :
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};
