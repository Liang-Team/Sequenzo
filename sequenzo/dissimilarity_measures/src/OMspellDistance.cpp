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
#include <xsimd/xsimd.hpp>

namespace py = pybind11;

class OMspellDistance {
public:
    OMspellDistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, py::array_t<int> refseqS,
                    double timecost, py::array_t<double> seqdur, py::array_t<double> indellist, py::array_t<int> seqlength)
            : indel(indel), norm(norm), timecost(timecost) {

        py::print("[>] Starting Optimal Matching with spell(OMspell)...");
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
            // initialize nseq, seqlen, dist_matrix, fmatsize, fmat
            // ====================================================
            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            len = seq_shape[1];

            dist_matrix = py::array_t<double>({nseq, nseq});

            fmatsize = len + 1;

            // ====================
            // initialize alphasize
            // ====================
            auto sm_shape = sm.shape();
            alphasize = sm_shape[0];

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

    // 对齐分配函数 moved to dp_utils.h

    double getIndel(int i, int j, int state){
        auto ptr_indel = indellist.mutable_unchecked<1>();
        auto ptr_dur = seqdur.mutable_unchecked<2>();

        return ptr_indel(state) + timecost * ptr_dur(i, j);
    }

    double getSubCost(int i_state, int j_state, int i_x, int i_y, int j_x, int j_y){
        auto ptr_dur = seqdur.mutable_unchecked<2>();

        if(i_state == j_state){
            double diffdur = ptr_dur(i_x, i_y) - ptr_dur(j_x, j_y);

            return abs(timecost * diffdur);
        }else{
            auto ptr_sm = sm.mutable_unchecked<2>();

            return ptr_sm(i_state, j_state) +
                    (ptr_dur(i_x, i_y) + ptr_dur(j_x, j_y)) * timecost;
        }
    }

    double compute_distance(int is, int js, double* prev, double* curr) {
        try {
            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_len = seqlength.unchecked<1>();
            auto ptr_sm = sm.unchecked<2>();
            auto ptr_dur = seqdur.unchecked<2>();
            auto ptr_indel = indellist.unchecked<1>();

            int i_state = 0, j_state = 0;
            int mm = ptr_len(is);
            int nn = ptr_len(js);
            int mSuf = mm + 1;
            int nSuf = nn + 1;

            prev[0] = 0;
            curr[0] = 0;

            // initialize first row: cumulative insertions into js along columns
            for (int jj = 1; jj < nSuf; jj++) {
                int bj = ptr_seq(js, jj - 1);
                prev[jj] = prev[jj - 1] + (ptr_indel(bj) + timecost * ptr_dur(js, jj - 1));
            }

            using batch_t = xsimd::batch<double>;
            constexpr std::size_t B = batch_t::size;

            for (int i = 1; i < mSuf; i++) {
                i_state = ptr_seq(is, i - 1);
                // per-row deletion cost (depends only on i_state and i position)
                double dur_i = ptr_dur(is, i - 1);
                double del_cost_i = ptr_indel(i_state) + timecost * dur_i;

                // first column: cumulative deletions D[i][0] = D[i-1][0] + del_cost_i
                curr[0] = prev[0] + del_cost_i;

                int j = 1;
                for (; j + (int)B <= nSuf; j += (int)B) {
                    const double* prev_ptr = prev + j;
                    const double* prevm1_ptr = prev + (j - 1);

                    batch_t prevj = batch_t::load_unaligned(prev_ptr);
                    batch_t prevjm1 = batch_t::load_unaligned(prevm1_ptr);

                    alignas(64) double subs[B];
                    alignas(64) double ins[B];
                    for (std::size_t b = 0; b < B; ++b) {
                        int jj_idx = j + (int)b - 1;
                        int bj = ptr_seq(js, jj_idx);
                        double dur_j = ptr_dur(js, jj_idx);

                        if (i_state == bj) {
                            subs[b] = std::abs(timecost * (dur_i - dur_j));
                        } else {
                            subs[b] = ptr_sm(i_state, bj) + (dur_i + dur_j) * timecost;
                        }
                        ins[b] = ptr_indel(bj) + timecost * dur_j;
                    }

                    batch_t sub_batch = batch_t::load_unaligned(subs);
                    batch_t cand_del = prevj + batch_t(del_cost_i);
                    batch_t cand_sub = prevjm1 + sub_batch;
                    batch_t vert = xsimd::min(cand_del, cand_sub);

                    double running = curr[j - 1] + ins[0];
                    for (std::size_t b = 0; b < B; ++b) {
                        double v = vert.get(b);
                        double c = std::min(v, running);
                        curr[j + (int)b] = c;
                        if (b + 1 < B) running = c + ins[b + 1];
                    }
                }

                // tail scalar handling
                for (; j < nSuf; ++j) {
                    j_state = ptr_seq(js, j - 1);
                    double minimum = prev[j] + del_cost_i;
                    double j_indel = curr[j - 1] + (ptr_indel(j_state) + timecost * ptr_dur(js, j - 1));
                    double sub = prev[j - 1] + (
                        (i_state == j_state)
                        ? std::abs(timecost * (dur_i - ptr_dur(js, j - 1)))
                        : (ptr_sm(i_state, j_state) + (dur_i + ptr_dur(js, j - 1)) * timecost)
                    );
                    curr[j] = std::min({ minimum, j_indel, sub });
                }

                std::swap(prev, curr);
            }

            double maxpossiblecost = std::abs(nn - mm) * indel + maxscost * std::min(mm, nn);
            double ml = double(mm) * indel;
            double nl = double(nn) * indel;

            return normalize_distance(prev[nSuf - 1], maxpossiblecost, ml, nl, norm);
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
            py::print("Error in compute_all_distances: ", e.what());
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
    double maxscost;

    double timecost;
    py::array_t<double> seqdur;
    py::array_t<double> indellist;

    // about reference sequences :
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};