/*
 * OMtspellDistance: Optimal Matching with spell durations and token-dependent costs.
 *
 * Optimizations vs original:
 *   [OPT-1] Removed pseudo-SIMD (same rationale as OMspell).
 *   [OPT-2] Manual 3-way min replaces std::min({a,b,c}).
 *   [OPT-3] Cache dur_j, tok_j once per j-iteration.
 *
 * @Author  : Yuqi Liang 梁彧祺, Yapeng Wei 卫亚鹏
 * @File    : OMtspellDistance.cpp
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

class OMtspellDistance {
public:
    OMtspellDistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, py::array_t<int> refseqS,
                     double timecost, py::array_t<double> seqdur, py::array_t<double> indellist,
                     py::array_t<int> seqlength, py::array_t<double> tokdeplist, py::array_t<int> norm_seqlength)
            : indel(indel), norm(norm), timecost(timecost) {

        py::print("[>] Starting Optimal Matching with token-dependent spell (OMtspell)...");
        std::cout << std::flush;

        try {
            this->sequences = sequences;
            this->sm = sm;
            this->seqdur = seqdur;
            this->indellist = indellist;
            this->seqlength = seqlength;
            this->tokdeplist = tokdeplist;
            this->norm_seqlength = norm_seqlength;

            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            len = seq_shape[1];

            dist_matrix = py::array_t<double>({nseq, nseq});
            fmatsize = len + 1;

            auto sm_shape = sm.shape();
            alphasize = sm_shape[0];

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
            py::print("Error in OMtspell constructor: ", e.what());
            throw;
        }
    }

    double getIndel(int i, int j, int state) {
        auto ptr_indel = indellist.mutable_unchecked<1>();
        auto ptr_tok = tokdeplist.mutable_unchecked<1>();
        auto ptr_dur = seqdur.mutable_unchecked<2>();
        return ptr_indel(state) + timecost * ptr_tok(state) * ptr_dur(i, j);
    }

    double getSubCost(int i_state, int j_state, int i_x, int i_y, int j_x, int j_y) {
        auto ptr_dur = seqdur.mutable_unchecked<2>();
        auto ptr_tok = tokdeplist.mutable_unchecked<1>();
        if (i_state == j_state) {
            double diffdur = ptr_dur(i_x, i_y) - ptr_dur(j_x, j_y);
            return std::abs(timecost * diffdur * ptr_tok(i_state));
        }
        auto ptr_sm = sm.mutable_unchecked<2>();
        return ptr_sm(i_state, j_state) +
               (ptr_tok(i_state) * ptr_dur(i_x, i_y) + ptr_tok(j_state) * ptr_dur(j_x, j_y)) * timecost;
    }

    double compute_distance(int is, int js, double* prev, double* curr) {
        try {
            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_len = seqlength.unchecked<1>();
            auto ptr_norm_len = norm_seqlength.unchecked<1>();
            auto ptr_sm = sm.unchecked<2>();
            auto ptr_dur = seqdur.unchecked<2>();
            auto ptr_indel = indellist.unchecked<1>();
            auto ptr_tok = tokdeplist.unchecked<1>();

            int mm = ptr_len(is);
            int nn = ptr_len(js);
            int mSuf = mm + 1;
            int nSuf = nn + 1;

            prev[0] = 0;
            curr[0] = 0;

            for (int jj = 1; jj < nSuf; jj++) {
                int bj = ptr_seq(js, jj - 1);
                prev[jj] = prev[jj - 1] + (ptr_indel(bj) + timecost * ptr_tok(bj) * ptr_dur(js, jj - 1));
            }

            // [OPT-1] Pure scalar DP — pseudo-SIMD removed.
            for (int i = 1; i < mSuf; i++) {
                int i_state = ptr_seq(is, i - 1);
                double dur_i = ptr_dur(is, i - 1);
                double tok_i = ptr_tok(i_state);
                double del_cost_i = ptr_indel(i_state) + timecost * tok_i * dur_i;

                curr[0] = prev[0] + del_cost_i;

                for (int j = 1; j < nSuf; j++) {
                    int j_state = ptr_seq(js, j - 1);
                    double dur_j = ptr_dur(js, j - 1);   // [OPT-3]
                    double tok_j = ptr_tok(j_state);      // [OPT-3]

                    double del_cost = prev[j] + del_cost_i;
                    double ins_cost = curr[j - 1] + (ptr_indel(j_state) + timecost * tok_j * dur_j);
                    double sub_cost = prev[j - 1] + (
                        (i_state == j_state)
                        ? std::abs(timecost * (dur_i - dur_j) * tok_i)
                        : (ptr_sm(i_state, j_state) + (tok_i * dur_i + tok_j * dur_j) * timecost)
                    );

                    // [OPT-2] Manual 3-way min.
                    double best = del_cost;
                    if (ins_cost < best) best = ins_cost;
                    if (sub_cost < best) best = sub_cost;
                    curr[j] = best;
                }
                std::swap(prev, curr);
            }

            int mm_norm = ptr_norm_len(is);
            int nn_norm = ptr_norm_len(js);
            double maxpossiblecost = std::abs(nn_norm - mm_norm) * indel + maxscost * std::min(mm_norm, nn_norm);
            double ml = double(mm_norm) * indel;
            double nl = double(nn_norm) * indel;
            return normalize_distance(prev[nSuf - 1], maxpossiblecost, ml, nl, norm);
        } catch (const std::exception& e) {
            py::print("Error in OMtspell compute_distance: ", e.what());
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
            py::print("Error in OMtspell compute_all_distances: ", e.what());
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
            py::print("Error in OMtspell compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> sequences;
    py::array_t<int> seqlength;
    py::array_t<double> sm;
    double indel;
    int norm;
    int nseq;
    int len;
    int alphasize;
    int fmatsize;
    py::array_t<double> dist_matrix;
    double maxscost = 0.0;
    double timecost;
    py::array_t<double> seqdur;
    py::array_t<double> indellist;
    py::array_t<double> tokdeplist;
    py::array_t<int> norm_seqlength;
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};
