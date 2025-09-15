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
            : sequences(sequences), sm(sm), indel(indel), norm(norm), timecost(timecost),
              seqdur(seqdur), indellist(indellist), seqlength(seqlength) {

        py::print("[>] Starting Optimal Matching with spell(OMspell) SIMD optimized...");
        std::cout << std::flush;

        try {
            // initialize nseq, seqlen, dist_matrix, fmatsize
            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            len = seq_shape[1];

            dist_matrix = py::array_t<double>(std::vector<py::ssize_t>{nseq, nseq});
            fmatsize = len + 1;

            // SIMD optimization: pre-allocate memory
            prev_row.resize(fmatsize);
            curr_row.resize(fmatsize);

            // initialize alphasize
            auto sm_shape = sm.shape();
            alphasize = sm_shape[0];

            // initialize maxcost
            auto ptr = sm.mutable_unchecked<2>();
            maxscost = 0.0;

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

            // about reference sequences
            nans = nseq;
            rseq1 = refseqS.at(0);
            rseq2 = refseqS.at(1);
            if(rseq1 < rseq2){
                nseq = rseq1;
                nans = nseq * (rseq2 - rseq1);
            }else{
                rseq1 = rseq1 - 1;
            }
            refdist_matrix = py::array_t<double>(std::vector<py::ssize_t>{nseq, (rseq2-rseq1)});
        } catch (const std::exception& e) {
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    double getIndel(int i, int j, int state){
        auto ptr_indel = indellist.unchecked<1>();
        auto ptr_dur = seqdur.unchecked<2>();
        return ptr_indel(state) + timecost * ptr_dur(i, j);
    }

    double getSubCost(int i_state, int j_state, int i_x, int i_y, int j_x, int j_y){
        auto ptr_dur = seqdur.unchecked<2>();

        if(i_state == j_state){
            double diffdur = ptr_dur(i_x, i_y) - ptr_dur(j_x, j_y);
            return std::abs(timecost * diffdur);
        }else{
            auto ptr_sm = sm.unchecked<2>();
            return ptr_sm(i_state, j_state) +
                    (ptr_dur(i_x, i_y) + ptr_dur(j_x, j_y)) * timecost;
        }
    }

    // SIMD optimized distance calculation - FIXED VERSION
    double compute_distance(int is, int js) {
        try {
            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_len = seqlength.unchecked<1>();

            int mm = ptr_len(is);
            int nn = ptr_len(js);
            int mSuf = mm + 1, nSuf = nn + 1;

            // Use pre-allocated memory
            double* prev = prev_row.data();
            double* curr = curr_row.data();

            using batch_t = xsimd::batch<double>;
            constexpr std::size_t B = batch_t::size;

            // Initialize first row
            prev[0] = 0;
            for(int j = 1; j < nSuf; j++){
                int j_state = ptr_seq(js, j-1);
                prev[j] = prev[j-1] + getIndel(js, j-1, j_state);
            }

            // Process each row
            for(int i = 1; i < mSuf; i++){
                int i_state = ptr_seq(is, i-1);

                // Initialize current row first column
                curr[0] = prev[0] + getIndel(is, i-1, i_state);

                // Process column by column to maintain dependencies
                int j = 1;

                // CRITICAL FIX: Only batch when we have enough elements AND safe boundaries
                for(; j + (int)B <= nSuf; j += (int)B){
                    // Bounds check to prevent memory access issues
                    if(j - 1 < 0 || j + B - 1 >= nSuf) break;

                    // Pre-compute all costs for this batch
                    alignas(64) double del_costs[B];
                    alignas(64) double sub_costs[B];
                    alignas(64) double prev_vals[B];
                    alignas(64) double prev_m1_vals[B];

                    // Load and compute costs element by element for safety
                    for(std::size_t b = 0; b < B; ++b){
                        int jj = j + int(b);
                        if(jj >= nSuf) break; // Extra safety check

                        int j_state_local = ptr_seq(js, jj - 1);

                        // Load previous row values
                        prev_vals[b] = prev[jj];
                        prev_m1_vals[b] = prev[jj - 1];

                        // Compute costs
                        del_costs[b] = getIndel(js, jj - 1, j_state_local);
                        sub_costs[b] = getSubCost(i_state, j_state_local, is, i-1, js, jj-1);
                    }

                    // Load as batches
                    batch_t prev_batch = batch_t::load_unaligned(prev_vals);
                    batch_t prev_m1_batch = batch_t::load_unaligned(prev_m1_vals);
                    batch_t del_batch = batch_t::load_unaligned(del_costs);
                    batch_t sub_batch = batch_t::load_unaligned(sub_costs);

                    // Calculate deletion and substitution candidates
                    batch_t cand_del = prev_batch + del_batch;
                    batch_t cand_sub = prev_m1_batch + sub_batch;

                    // Store intermediate results
                    alignas(64) double del_vals[B];
                    alignas(64) double sub_vals[B];
                    cand_del.store_unaligned(del_vals);
                    cand_sub.store_unaligned(sub_vals);

                    // Process insertion serially (maintain left dependency)
                    for(std::size_t b = 0; b < B; ++b){
                        int jj = j + int(b);
                        if(jj >= nSuf) break;

                        // insertion cost: curr[j-1] + i_indel
                        double ins_cost = curr[jj - 1] + getIndel(is, i-1, i_state);

                        // Take minimum
                        curr[jj] = std::min({del_vals[b], ins_cost, sub_vals[b]});
                    }
                }

                // Process remaining scalar elements
                for(; j < nSuf; j++){
                    int j_state_local = ptr_seq(js, j-1);

                    double del_cost = prev[j] + getIndel(js, j-1, j_state_local);
                    double ins_cost = curr[j-1] + getIndel(is, i-1, i_state);
                    double sub_cost = prev[j-1] + getSubCost(i_state, j_state_local, is, i-1, js, j-1);

                    curr[j] = std::min({del_cost, ins_cost, sub_cost});
                }

                // Swap row pointers
                std::swap(prev, curr);
            }

            double final_cost = prev[nSuf-1];
            double maxpossiblecost = std::abs(nn - mm) * indel + maxscost * std::min(mm, nn);
            double ml = double(mm) * indel;
            double nl = double(nn) * indel;

            return normalize_distance(final_cost, maxpossiblecost, ml, nl, norm);

        } catch (const std::exception& e) {
            py::print("Error in compute_distance_simd: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            auto buffer = dist_matrix.mutable_unchecked<2>();

            #pragma omp parallel
            {
                // Each thread needs its own working memory
                thread_local std::vector<double> thread_prev(fmatsize);
                thread_local std::vector<double> thread_curr(fmatsize);

                #pragma omp for schedule(guided)
                for (int i = 0; i < nseq; i++) {
                    for (int j = i; j < nseq; j++) {
                        buffer(i, j) = compute_distance_simd_with_buffers(i, j, thread_prev.data(), thread_curr.data());
                    }
                }
            }

            #pragma omp parallel for schedule(static)
            for(int i = 0; i < nseq; i++) {
                for(int j = i+1; j < nseq; j++) {
                    buffer(j, i) = buffer(i, j);
                }
            }

            return dist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances_simd: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto buffer = refdist_matrix.mutable_unchecked<2>();

            #pragma omp parallel
            {
                thread_local std::vector<double> thread_prev(fmatsize);
                thread_local std::vector<double> thread_curr(fmatsize);

                #pragma omp for schedule(guided)
                for (int rseq = rseq1; rseq < rseq2; rseq ++) {
                    for (int is = 0; is < nseq; is ++) {
                        if(is == rseq){
                            buffer(is, rseq-rseq1) = 0;
                        }else{
                            buffer(is, rseq-rseq1) = compute_distance_simd_with_buffers(is, rseq, thread_prev.data(), thread_curr.data());
                        }
                    }
                }
            }

            return refdist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_refseq_distances: ", e.what());
            throw;
        }
    }

private:
    // Thread-safe version with provided buffers
    double compute_distance_simd_with_buffers(int is, int js, double* prev, double* curr) {
        try {
            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_len = seqlength.unchecked<1>();

            int mm = ptr_len(is);
            int nn = ptr_len(js);
            int mSuf = mm + 1, nSuf = nn + 1;

            using batch_t = xsimd::batch<double>;
            constexpr std::size_t B = batch_t::size;

            // Initialize first row
            prev[0] = 0;
            for(int j = 1; j < nSuf; j++){
                int j_state = ptr_seq(js, j-1);
                prev[j] = prev[j-1] + getIndel(js, j-1, j_state);
            }

            // Process each row
            for(int i = 1; i < mSuf; i++){
                int i_state = ptr_seq(is, i-1);
                curr[0] = prev[0] + getIndel(is, i-1, i_state);

                int j = 1;
                for(; j + (int)B <= nSuf; j += (int)B){
                    const double* prev_ptr = prev + j;
                    const double* prev_m1_ptr = prev + j - 1;

                    batch_t prevj = batch_t::load_unaligned(prev_ptr);
                    batch_t prevjm1 = batch_t::load_unaligned(prev_m1_ptr);

                    alignas(64) double del_costs[B];
                    alignas(64) double sub_costs[B];

                    for(std::size_t b = 0; b < B; ++b){
                        int jj = j + int(b);
                        int j_state_local = ptr_seq(js, jj - 1);
                        del_costs[b] = getIndel(js, jj - 1, j_state_local);
                        sub_costs[b] = getSubCost(i_state, j_state_local, is, i-1, js, jj-1);
                    }

                    batch_t del_batch = batch_t::load_unaligned(del_costs);
                    batch_t sub_batch = batch_t::load_unaligned(sub_costs);

                    batch_t cand_del = prevj + del_batch;
                    batch_t cand_sub = prevjm1 + sub_batch;

                    alignas(64) double del_vals[B];
                    alignas(64) double sub_vals[B];
                    cand_del.store_unaligned(del_vals);
                    cand_sub.store_unaligned(sub_vals);

                    for(std::size_t b = 0; b < B; ++b){
                        int jj = j + int(b);
                        double ins_cost = curr[jj - 1] + getIndel(is, i-1, i_state);
                        curr[jj] = std::min({del_vals[b], ins_cost, sub_vals[b]});
                    }
                }

                for(; j < nSuf; j++){
                    int j_state_local = ptr_seq(js, j-1);
                    double del_cost = prev[j] + getIndel(js, j-1, j_state_local);
                    double ins_cost = curr[j-1] + getIndel(is, i-1, i_state);
                    double sub_cost = prev[j-1] + getSubCost(i_state, j_state_local, is, i-1, js, j-1);
                    curr[j] = std::min({del_cost, ins_cost, sub_cost});
                }

                std::swap(prev, curr);
            }

            double final_cost = prev[nSuf-1];
            double maxpossiblecost = std::abs(nn - mm) * indel + maxscost * std::min(mm, nn);
            double ml = double(mm) * indel;
            double nl = double(nn) * indel;

            return normalize_distance(final_cost, maxpossiblecost, ml, nl, norm);

        } catch (const std::exception& e) {
            py::print("Error in compute_distance_simd_with_buffers: ", e.what());
            throw;
        }
    }

    // Member variables
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

    // about reference sequences
    int nans;
    int rseq1;
    int rseq2;
    py::array_t<double> refdist_matrix;

    // SIMD optimization related member variables
    std::vector<double> prev_row;
    std::vector<double> curr_row;
};