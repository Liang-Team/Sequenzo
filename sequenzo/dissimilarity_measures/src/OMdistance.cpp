#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <xsimd/xsimd.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
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

    // 对齐分配函数
    #ifdef _WIN32
    inline double* aligned_alloc_double(size_t size, size_t align=64) {
        return reinterpret_cast<double*>(_aligned_malloc(size * sizeof(double), align));
    }
    inline void aligned_free_double(double* ptr) {
        _aligned_free(ptr);
    }
    #else
    inline double* aligned_alloc_double(size_t size, size_t align=64) {
        void* ptr = nullptr;
        if(posix_memalign(&ptr, align, size*sizeof(double)) != 0) throw std::bad_alloc();
        return reinterpret_cast<double*>(ptr);
    }
    inline void aligned_free_double(double* ptr) { free(ptr); }
    #endif

    double compute_distance(int is, int js, double* prev, double* curr) {
        try {
            auto ptr_len = seqlength.unchecked<1>();
            double maxpossiblecost;
            int m = ptr_len(is);
            int n = ptr_len(js);
            int mSuf = m+1, nSuf = n+1;
            int prefix = 0;

            auto ptr_seq = sequences.unchecked<2>();
            auto ptr_sm = sm.unchecked<2>();

            int i = 1;
            int j = 1;

            double ml = 0;
            double nl = 0;

//            Skipping common prefix
            while (i < mSuf && j < nSuf &&
                   ptr_seq(is, i-1) == ptr_seq(js, j-1)){
                i++; j++; prefix++;
            }

//            Skipping common suffix
            while (mSuf > i && nSuf > j
                   && ptr_seq(is, mSuf - 2) == ptr_seq(js, nSuf - 2)) {
                mSuf--; nSuf--;
            }

            m = mSuf - prefix;
            n = nSuf - prefix;

            #pragma omp simd
            for(i = 0; i < m; i ++)
                prev[i]  = i * indel;

            for(i = prefix+1; i < mSuf; i ++){
                curr[0] = indel * (i - prefix);

                for(int j = prefix+1; j < nSuf; j ++){
                    // Use SIMD batch processing to compute min and other operations
                    xsimd::batch<double> min_batch, j_indel_batch, sub_batch;

                    // Calculate the three values and perform min operation
                    min_batch = xsimd::batch<double>(prev[j-prefix] + indel);
                    j_indel_batch = xsimd::batch<double>(curr[j-1-prefix] + indel);
                    sub_batch = xsimd::batch<double>((ptr_seq(is, i-1) == ptr_seq(js, j-1)) ?
                                                     prev[j-1-prefix] :
                                                     (prev[j-1-prefix] + ptr_sm(ptr_seq(is, i-1), ptr_seq(js, j-1))));

                    // Store the result
                    xsimd::batch<double> result = xsimd::min(min_batch, j_indel_batch);
                    result = xsimd::min(result, sub_batch);
                    curr[j-prefix] = result.get(0);
                }

                std::swap(prev, curr);
            }

            maxpossiblecost = abs(n-m) * indel + maxscost * std::min(m, n);

            ml = double(m) * indel;
            nl = double(n) * indel;

            return normalize_distance(prev[nSuf-1-prefix], maxpossiblecost, ml, nl, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            auto buffer = dist_matrix.mutable_unchecked<2>();
            double* prev = aligned_alloc_double(fmatsize);
            double* curr = aligned_alloc_double(fmatsize);

            #pragma omp parallel
            {
                #pragma omp for schedule(guided)
                for (int i = 0; i < nseq; i++) {
                    for (int j = i; j < nseq; j++) {
                        buffer(i, j) = compute_distance(i, j, prev, curr);
                    }
                }
            }

            aligned_free_double(prev);
            aligned_free_double(curr);

            #pragma omp parallel for schedule(static)
            for(int i = 0; i < nseq; i++)
                for(int j = i+1; j < nseq; j++)
                    buffer(j, i) = buffer(i, j);

            return dist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_refseq_distances() {
        try {
            auto buffer = refdist_matrix.mutable_unchecked<2>();
            double* prev = aligned_alloc_double(fmatsize);
            double* curr = aligned_alloc_double(fmatsize);

            double cmpres = 0;
            #pragma omp parallel
            {
                #pragma omp for schedule(static)
                for (int rseq = rseq1; rseq < rseq2; rseq ++) {
                    for (int is = 0; is < nseq; is ++) {
                        if(is == rseq){
                            cmpres = 0;
                        }else{
                            cmpres = compute_distance(is, rseq, prev, curr);
                        }

                        buffer(is, rseq-rseq1) = cmpres;
                    }
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