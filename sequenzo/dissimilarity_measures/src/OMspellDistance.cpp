#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <xsimd/xsimd.hpp>
#include "utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

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

    double getIndel(int i, int j, int state) {
        auto ptr_indel = indellist.mutable_unchecked<1>();
        auto ptr_dur = seqdur.mutable_unchecked<2>();

        xsimd::batch<double, xsimd::default_arch> state_vec(ptr_indel(state));
        xsimd::batch<double, xsimd::default_arch> timecost_vec(timecost);

        xsimd::batch<double, xsimd::default_arch> dur_vec(ptr_dur(i, j));
        xsimd::batch<double, xsimd::default_arch> result = state_vec + timecost_vec * dur_vec;

        return result.get(0);
    }


    double getSubCost(int i_state, int j_state, int i_x, int i_y, int j_x, int j_y) {
        auto ptr_dur = seqdur.mutable_unchecked<2>();

        if (i_state == j_state) {
            double diffdur = ptr_dur(i_x, i_y) - ptr_dur(j_x, j_y);
            return std::abs(timecost * diffdur);
        } else {
            auto ptr_sm = sm.mutable_unchecked<2>();

            double d1 = ptr_dur(i_x, i_y);
            double d2 = ptr_dur(j_x, j_y);

            xsimd::batch<double, xsimd::default_arch> d1_vec = xsimd::batch<double, xsimd::default_arch>::broadcast(d1);
            xsimd::batch<double, xsimd::default_arch> d2_vec = xsimd::batch<double, xsimd::default_arch>::broadcast(d2);
            xsimd::batch<double, xsimd::default_arch> cost = xsimd::batch<double, xsimd::default_arch>::broadcast(timecost);
            xsimd::batch<double, xsimd::default_arch> sum = (d1_vec + d2_vec) * cost;

            return ptr_sm(i_state, j_state) + sum.get(0);
        }
    }


    double compute_distance(int is, int js) {
        try {
            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_len = seqlength.unchecked<1>();

            int i_state = 0, j_state = 0;
            double maxpossiblecost;
            int mm = ptr_len(is);
            int nn = ptr_len(js);
            int mSuf = mm + 1, nSuf = nn + 1;

            std::vector<std::vector<double>> fmat(fmatsize, std::vector<double>(fmatsize, 0));

            fmat[0][0] = 0;

            for (int ii = 1; ii < mSuf; ii++) {
                i_state = ptr_seq(is, ii - 1);
                fmat[ii][0] = fmat[ii - 1][0] + getIndel(is, ii - 1, i_state);
            }

            for (int ii = 1; ii < nSuf; ii++) {
                j_state = ptr_seq(js, ii - 1);
                fmat[0][ii] = fmat[0][ii - 1] + getIndel(js, ii - 1, j_state);
            }

            for (int i = 1; i < mSuf; i++) {
                i_state = ptr_seq(is, i - 1);

                for (int j = 1; j < nSuf; j++) {
                    j_state = ptr_seq(js, j - 1);

                    xsimd::batch<double, xsimd::default_arch> minimum_batch = fmat[i - 1][j] + getIndel(is, i - 1, i_state);
                    xsimd::batch<double, xsimd::default_arch> j_indel_batch = fmat[i][j - 1] + getIndel(js, j - 1, j_state);
                    xsimd::batch<double, xsimd::default_arch> sub_batch = fmat[i - 1][j - 1] + getSubCost(i_state, j_state, is, i - 1, js, j - 1);

                    xsimd::batch<double> result = xsimd::min(xsimd::min(minimum_batch, j_indel_batch), sub_batch);
                    fmat[i][j] = result.get(0);
                }
            }

            maxpossiblecost = std::abs(nn - mm) * indel + maxscost * std::min(mm, nn);
            double ml = double(mm) * indel;
            double nl = double(nn) * indel;

            return normalize_distance(fmat[mSuf - 1][nSuf - 1], maxpossiblecost, ml, nl, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }


    py::array_t<double> compute_all_distances() {
        try {
            auto buffer = dist_matrix.mutable_unchecked<2>();

#pragma omp parallel for schedule(static)
            for (int i = 0; i < nseq; i++) {
                for (int j = i; j < nseq; j++) {
                    double dist = compute_distance(i, j);
                    buffer(i, j) = dist;
                    buffer(j, i) = dist;
                }
            }

            return dist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto buffer = refdist_matrix.mutable_unchecked<2>();

            double cmpres = 0;
#pragma omp parallel for schedule(static)
            for (int rseq = rseq1; rseq < rseq2; rseq ++) {
                for (int is = 0; is < nseq; is ++) {
                    if(is == rseq){
                        cmpres = 0;
                    }else{
                        cmpres = compute_distance(is, rseq);
                    }

                    buffer(is, rseq-rseq1) = cmpres;
                }
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