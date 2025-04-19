#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class DHDdistance{
public:
    DHDdistance(py::array_t<int> sequences, py::array_t<double> sm, int norm, double maxdist, py::array_t<int> refseqS)
            : norm(norm), maxdist(maxdist){
        py::print("[>] Starting (Dynamic) Hamming Distance(DHD/HAM)...");
        std::cout << std::flush;

        try{
            this->sequences = sequences;
            this->sm = sm;

            auto seq_shape = sequences.shape();
            nseq = seq_shape[0];
            len = seq_shape[1];

            dist_matrix = py::array_t<double>({nseq, nseq});

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
        } catch (const std::exception& e){
            py::print("Error in constructor: ", e.what());
            throw ;
        }
    }

    double compute_distance(int is, int js) {
        try {
            int m = len;
            int n = len;
            int minimum = m;
            if(n < m) minimum = n;
            double cost = 0;

            auto ptr_sm = sm.unchecked<3>();
            auto ptr_seq = sequences.unchecked<2>();

            for(int i=0; i<minimum; i++){
                cost += ptr_sm(i, ptr_seq(is, i), ptr_seq(js, i));
            }

            return normalize_distance(cost, maxdist, maxdist, maxdist, norm);
        } catch (const std::exception& e) {
            py::print("Error in compute_distance: ", e.what());
            throw;
        }
    }

    py::array_t<double> compute_all_distances() {
        try {
            auto buffer = dist_matrix.mutable_unchecked<2>();

#pragma omp parallel for collapse(2) schedule(static)
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
#pragma omp parallel for collapse(2) schedule(static)
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
    py::array_t<double> sm;
    int norm;
    int nseq;
    int len;
    py::array_t<double> dist_matrix;
    double maxdist;

    py::array_t<int> refseqS;
    int nans = -1;
    int rseq1 = -1;
    int rseq2 = -1;
    py::array_t<double> refdist_matrix;
};