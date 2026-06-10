#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

class HAMdistance {
public:
    HAMdistance(py::array_t<int> sequences, py::array_t<double> sm, int norm, double maxdist, py::array_t<int> refseqS)
            : sequences(sequences), sm(sm), norm(norm), maxdist(maxdist) {
        py::print("[>] Starting Hamming Distance(HAM)...");
        std::cout << std::flush;

        auto seq_shape = sequences.shape();
        nseq = static_cast<int>(seq_shape[0]);
        len = static_cast<int>(seq_shape[1]);
        seq_ptr = sequences.data();
        sm_ptr = sm.data();
        sm_stride = static_cast<int>(sm.shape(1));

        rseq1 = refseqS.at(0);
        rseq2 = refseqS.at(1);
        if (rseq1 < rseq2) {
            nseq = rseq1;
        } else {
            rseq1 = rseq1 - 1;
        }
    }

    inline double state_cost(int left, int right) const {
        return left == right ? 0.0 : sm_ptr[static_cast<ptrdiff_t>(left) * sm_stride + right];
    }

    double compute_distance(int is, int js) const {
        const int* row_i = seq_ptr + static_cast<ptrdiff_t>(is) * len;
        const int* row_j = seq_ptr + static_cast<ptrdiff_t>(js) * len;
        double cost = 0.0;
        for (int k = 0; k < len; ++k) {
            cost += state_cost(row_i[k], row_j[k]);
        }
        return normalize_distance(cost, maxdist, maxdist, maxdist, norm);
    }

    py::array_t<double> compute_all_distances() {
        auto dist_matrix = py::array_t<double>({nseq, nseq});
        return dp_utils::compute_all_distances_simple(
            nseq,
            dist_matrix,
            [this](int i, int j) { return this->compute_distance(i, j); }
        );
    }

    py::array_t<double> compute_condensed_distances() {
        return dp_utils::compute_condensed_distances_simple(
            nseq,
            [this](int i, int j) { return this->compute_distance(i, j); }
        );
    }

    py::array_t<double> compute_refseq_distances() {
        auto refdist_matrix = py::array_t<double>({nseq, (rseq2 - rseq1)});
        return dp_utils::compute_refseq_distances_simple(
            nseq,
            rseq1,
            rseq2,
            refdist_matrix,
            [this](int is, int rseq) { return this->compute_distance(is, rseq); }
        );
    }

private:
    py::array_t<int> sequences;
    py::array_t<double> sm;
    int norm;
    int nseq;
    int len;
    double maxdist;
    const int* seq_ptr = nullptr;
    const double* sm_ptr = nullptr;
    int sm_stride = 0;
    int rseq1 = -1;
    int rseq2 = -1;
};
