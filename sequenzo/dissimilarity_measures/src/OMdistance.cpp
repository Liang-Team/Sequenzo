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

class OMdistance {
public:
    OMdistance(py::array_t<int> sequences, py::array_t<double> sm, double indel, int norm, py::array_t<int> seqlength,py::array_t<int> refseqS)
            : indel(indel), norm(norm) {

        py::print("[>] Starting Optimal Matching(OM)...");
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
            if(norm == 4){
                maxscost = 2 * indel;
            }else{
                auto ptr = sm.mutable_unchecked<2>();
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

    double compute_distance(int is, int js, double* prev, double* curr) {
        try {
            auto ptr_len = seqlength.unchecked<1>();
            int m_full = ptr_len(is);
            int n_full = ptr_len(js);
            int mSuf = m_full + 1, nSuf = n_full + 1;
            int prefix = 0;

            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_sm = sm.unchecked<2>();

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
                double cost = double(n) * indel;
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, 0.0, double(n) * indel, norm);
            }
            if (n == 0) {
                double cost = double(m) * indel;
                double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
                return normalize_distance(cost, maxpossiblecost, double(m) * indel, 0.0, norm);
            }

            using batch_t = xsimd::batch<double>;
            constexpr std::size_t B = batch_t::size;

            #pragma omp simd
            for (int x = 0; x < n; ++x) prev[x] = double(x) * indel;

            for (int i = prefix + 1; i < mSuf; ++i) {
                curr[0] = indel * double(i - prefix);
                int ai = ptr_seq(is, i - 1);

                int j = prefix + 1;
                for (; j + (int)B <= nSuf; j += (int)B) {
                    // load prev[j .. j+B-1], prev[j-1 .. j+B-2]
                    const double* prev_ptr   = prev + (j - prefix);
                    const double* prevm1_ptr = prev + (j - 1 - prefix);

                    batch_t prevj   = batch_t::load_unaligned(prev_ptr);
                    batch_t prevjm1 = batch_t::load_unaligned(prevm1_ptr);

                    // substitution costs
                    alignas(64) double subs[B];
                    for (std::size_t b = 0; b < B; ++b) {
                        int jj_idx = j + int(b);
                        int bj = ptr_seq(js, jj_idx - 1);
                        subs[b] = (ai == bj) ? 0.0 : ptr_sm(ai, bj);
                    }
                    batch_t sub_batch = batch_t::load_unaligned(subs);

                    // Vectorize independent candidates: del and sub
                    batch_t cand_del = prevj + batch_t(indel);
                    batch_t cand_sub = prevjm1 + sub_batch;
                    batch_t vert = xsimd::min(cand_del, cand_sub);

                    // Sequential propagation for insert dependencies (low overhead)
                    double running_ins = curr[j - prefix - 1] + indel;
                    for (std::size_t b = 0; b < B; ++b) {
                        double v = vert.get(b);
                        double c = std::min(v, running_ins);
                        curr[j + int(b) - prefix] = c;
                        running_ins = c + indel;
                    }
                }

                // 补足尾部
                for (; j < nSuf; ++j) {
                    int bj = ptr_seq(js, j-1);
                    double subcost = (ai == bj) ? 0.0 : ptr_sm(ai, bj);
                    double delcost = prev[j - prefix] + indel;
                    double inscost = curr[j - 1 - prefix] + indel;
                    double subval  = prev[j - 1 - prefix] + subcost;
                    curr[j - prefix] = std::min({ delcost, inscost, subval });
                }

                std::swap(prev, curr);
            }

            double final_cost = prev[nSuf - 1 - prefix];
            double maxpossiblecost = std::abs(n - m) * indel + maxscost * std::min(m, n);
            double ml = double(m) * indel;
            double nl = double(n) * indel;
            return normalize_distance(final_cost, maxpossiblecost, ml, nl, norm);

        } catch (const std::exception& e) {
            py::print("Error in SIMD-batch compute_distance: ", e.what());
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

    // about reference sequences :
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};