/*
 * OMlocDistance: Optimal Matching with localized indel costs.
 *
 * Unlike standard OM (which uses constant indel costs), OMloc uses context-dependent
 * indel costs that depend on the previous and next states in the sequence.
 *
 * Formula for indel cost:
 *   indel(state, prev, next) = timecost + localcost * (scost[prev,state] + scost[next,state]) / 2
 *
 * where:
 *   - timecost = expcost * maxscost (time-dependent cost)
 *   - localcost = context (localization cost parameter)
 *   - scost: substitution cost matrix
 *   - prev: previous state in the sequence
 *   - next: next state in the sequence
 *
 * This makes the indel cost sensitive to the local context, penalizing insertions/deletions
 * more when they occur between states with high substitution costs.
 *
 * Normalization:
 *   Supports standard OM normalizations: "none", "maxlength", "gmean", "maxdist", "YujianBo".
 *   Default (auto) is "YujianBo" for OMloc.
 *
 * Parameters:
 *   - expcost (>= 0): exponential cost parameter for time-dependent component
 *   - context (>= 0): localization cost parameter for context-dependent component
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : OMlocDistance.cpp
 * @Time    : 2026/2/4 9:17
 * @Desc    : Optimal Matching with localized indel costs (context-dependent).
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
                  double expcost, double context)
            : indel(indel), norm(norm) {

        py::print("[>] Starting Optimal Matching with localized indel costs (OMloc)...");
        std::cout << std::flush;

        try {
            // =========================
            // parameter : sequences, sm
            // =========================
            this->sequences = sequences;
            this->sm = sm;
            this->seqlength = seqlength;

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
            maxscost = 0.0;
            for(int i = 0; i < alphasize; i++){
                for(int j = i+1; j < alphasize; j++){
                    if(ptr(i, j) > maxscost){
                        maxscost = ptr(i, j);
                    }
                }
            }
            maxscost = std::min(maxscost, 2 * indel);

            // Calculate timecost = expcost * maxscost (as in R code)
            timecost = expcost * maxscost;
            localcost = context;

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

    // Get context-dependent indel cost
    // Formula: timecost + localcost * (scost[prev,state] + scost[next,state]) / 2
    inline double getIndel(int state, int prev, int next) {
        auto ptr_sm = sm.unchecked<2>();
        double scost_prev = ptr_sm(prev, state);
        double scost_next = ptr_sm(next, state);
        return timecost + localcost * (scost_prev + scost_next) / 2.0;
    }

    double compute_distance(int is, int js, double* prev, double* curr) {
        try {
            auto ptr_len = seqlength.unchecked<1>();
            int m_full = ptr_len(is);
            int n_full = ptr_len(js);
            int prefix = 0;  // TraMineR OMVIdistance does NOT skip prefix/suffix

            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_sm = sm.unchecked<2>();

            int m = m_full;
            int n = n_full;

            // 预处理
            if (m == 0 && n == 0)
                return normalize_distance(0.0, 0.0, 0.0, 0.0, norm);
            if (m == 0) {
                // For empty sequence, use average indel cost
                double cost = double(n) * indel;
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, 0.0, double(n) * indel, norm);
            }
            if (n == 0) {
                double cost = double(m) * indel;
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, double(m) * indel, 0.0, norm);
            }

            // Initialize DP matrix: F[0][0] = 0
            // TraMineR uses column-major fmat; we use prev=row i-1, curr=row i
            int firststate = std::max(prefix - 1, 0);

            // First column F[i][0]: deletions from sequence is (stored for curr[0] each row)
            int prev_jstate = ptr_seq(js, prefix);
            int j_state = ptr_seq(js, firststate);
            prev[0] = 0.0;
            for (int ii = 1; ii <= m; ++ii) {
                int i_state_del = ptr_seq(is, ii - 1);
                prev[ii] = prev[ii - 1] + getIndel(i_state_del, prev_jstate, j_state);
            }
            // prev[0..m] now holds F[0][0], F[1][0], ..., F[m][0] (first column)

            // First row F[0][j]: insertions into sequence js
            int prev_istate = ptr_seq(is, prefix);
            int i_state = ptr_seq(is, firststate);
            curr[0] = 0.0;
            for (int jj = 1; jj <= n; ++jj) {
                int j_state_ins = ptr_seq(js, jj - 1);
                curr[jj] = curr[jj - 1] + getIndel(j_state_ins, prev_istate, i_state);
            }
            // curr[0..n] now holds F[0][0], F[0][1], ..., F[0][n] (first row)

            // Swap: prev = first row (F[0][*]), curr = first column (F[*][0])
            std::swap(prev, curr);
            // Now prev[0..n]=F[0][j], curr[0..m]=F[i][0]. curr gets overwritten in j-loop, so
            // we recompute F[i][0] each row (cumulative deletion cost).

            // Main DP loop: row by row. prev = row i-1, curr = row i
            prev_istate = ptr_seq(is, firststate);
            i_state = ptr_seq(is, prefix);
            prev_jstate = ptr_seq(js, prefix);
            j_state = ptr_seq(js, firststate);
            for (int i = 1; i <= m; ++i) {
                int i_state_curr = ptr_seq(is, i - 1);

                curr[0] = 0.0;
                for (int k = 0; k < i; ++k)
                    curr[0] += getIndel(ptr_seq(is, k), prev_jstate, j_state);
                for (int j = 1; j <= n; ++j) {
                    int j_state_curr = ptr_seq(js, j - 1);

                    double del_cost = prev[j] + getIndel(i_state_curr, prev_jstate, j_state);
                    double ins_cost = curr[j - 1] + getIndel(j_state_curr, prev_istate, i_state_curr);
                    double sub_cost = (i_state_curr == j_state_curr)
                        ? prev[j - 1] : prev[j - 1] + ptr_sm(i_state_curr, j_state_curr);

                    curr[j] = std::min({del_cost, ins_cost, sub_cost});

                    prev_jstate = j_state_curr;
                    j_state = (j < n_full) ? ptr_seq(js, j) : ptr_seq(js, j - 1);
                }
                prev_istate = i_state_curr;
                i_state = (i < m_full) ? ptr_seq(is, i) : ptr_seq(is, i - 1);
                std::swap(prev, curr);
            }

            double final_cost = prev[n];
            // For normalization, use standard OM formula
            // maxpossiblecost = |n-m|*indel + maxscost*min(m,n)
            // But for OMloc, indel varies, so we use average indel approximation
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
    double timecost;  // expcost * maxscost
    double localcost; // context

    // about reference sequences :
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};
