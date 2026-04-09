/*
 * TWEDdistance: Time Warp Edit Distance (TraMineR-aligned).
 *
 * Optimizations vs original:
 *   [OPT-1] 2D fmat[m+1][n+1] → 2-row prev/curr + swap.
 *           TWED recurrence depends only on fmat[i-1][j-1], fmat[i-1][j], fmat[i][j-1]
 *           (previous row + current row left neighbor). Original allocated fmatsize_²
 *           doubles per thread (~7.5KB for L=30, ~37KB for L=50). Now 2*fmatsize_ (~0.5KB).
 *           Better cache locality for the inner loop.
 *   [OPT-2] Manual 3-way min replaces std::min({a,b,c}).
 *
 * @Author  : Yuqi Liang 梁彧祺, Yapeng Wei 卫亚鹏
 * @File    : TWEDdistance.cpp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <cmath>
#include <algorithm>
#include <vector>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class TWEDdistance {
public:
    TWEDdistance(py::array_t<int> sequences,
                 py::array_t<double> sm,
                 double indel,
                 int norm,
                 double nu,
                 double lambda,
                 py::array_t<int> seqlength,
                 py::array_t<int> refseq_id)
        : indel_(indel), norm_(norm), nu_(nu), lambda_(lambda) {
        sequences_ = sequences;
        sm_ = sm;
        seqlength_ = seqlength;

        auto sh = sequences.unchecked<2>();
        nseq_ = static_cast<int>(sh.shape(0));
        seqlen_ = static_cast<int>(sh.shape(1));
        alphasize_ = static_cast<int>(sm.shape()[0]) - 1;
        fmatsize_ = seqlen_ + 1;

        if (norm == 4) {
            maxscost_ = 2.0 * indel;
        } else {
            auto ptr = sm.unchecked<2>();
            maxscost_ = 0.0;
            for (int i = 0; i <= alphasize_; i++)
                for (int j = i + 1; j <= alphasize_; j++) {
                    double c = ptr(i, j);
                    if (c > maxscost_) maxscost_ = c;
                }
            maxscost_ = std::min(maxscost_, 2.0 * indel_);
        }

        rseq1_ = refseq_id.at(0);
        rseq2_ = refseq_id.at(1);
        if (rseq1_ < rseq2_) {
            nseq_ = rseq1_;
            refdist_matrix_ = py::array_t<double>(std::array<py::ssize_t, 2>{static_cast<py::ssize_t>(nseq_), static_cast<py::ssize_t>(rseq2_ - rseq1_)});
        } else {
            rseq1_ = rseq1_ - 1;
            refdist_matrix_ = py::array_t<double>(std::array<py::ssize_t, 2>{static_cast<py::ssize_t>(nseq_), 1});
        }
        dist_matrix_ = py::array_t<double>(std::array<py::ssize_t, 2>{static_cast<py::ssize_t>(nseq_), static_cast<py::ssize_t>(nseq_)});
    }

    // [OPT-1] Changed from (int, int, double* fmat) to 2-row prev/curr.
    double compute_distance(int is, int js, double* prev, double* curr) const {
        auto ptr_len = seqlength_.unchecked<1>();
        int m = ptr_len(is);
        int n = ptr_len(js);

        if (m == 0 && n == 0)
            return normalize_distance(0.0, 0.0, 0.0, 0.0, norm_);
        if (m == 0) {
            double cost = n * indel_;
            double maxcost = std::abs(n - m) * (nu_ + lambda_ + maxscost_) + 2.0 * (maxscost_ + nu_) * std::min(m, n);
            return normalize_distance(cost, maxcost, 0.0, n * indel_, norm_);
        }
        if (n == 0) {
            double cost = m * indel_;
            double maxcost = std::abs(n - m) * (nu_ + lambda_ + maxscost_) + 2.0 * (maxscost_ + nu_) * std::min(m, n);
            return normalize_distance(cost, maxcost, m * indel_, 0.0, norm_);
        }

        auto ptr_seq = sequences_.unchecked<2>();
        auto ptr_sm = sm_.unchecked<2>();
        const double inf = 1000.0 * (maxscost_ + nu_ + lambda_);

        // Initialize row 0: prev[j] = j * indel
        prev[0] = 0.0;
        for (int j = 1; j <= n; j++) prev[j] = j * indel_;

        for (int i = 1; i <= m; i++) {
            int i_state = ptr_seq(is, i - 1);
            int i_m1 = (i == 1) ? 0 : ptr_seq(is, i - 2);

            // Column 0 boundary
            curr[0] = i * indel_;

            for (int j = 1; j <= n; j++) {
                int j_state = ptr_seq(js, j - 1);
                int j_m1 = (j == 1) ? 0 : ptr_seq(js, j - 2);

                // Substitution: uses prev[j-1] = fmat[i-1][j-1]
                double cost;
                if ((i_state == j_state) && (i_m1 == j_m1)) {
                    cost = 0.0;
                } else {
                    cost = ptr_sm(i_state, j_state) + ptr_sm(i_m1, j_m1);
                }
                double sub = prev[j - 1] + cost + 2.0 * nu_ * std::abs(i - j);

                // i_warp: uses curr[j-1] = fmat[i][j-1] (already computed)
                double i_warp = inf;
                if (j > 1) {
                    i_warp = curr[j - 1] + ptr_sm(j_state, j_m1) + nu_ + lambda_;
                } else if (i > 1) {
                    sub = inf;
                }

                // j_warp: uses prev[j] = fmat[i-1][j]
                double j_warp = inf;
                if (i > 1) {
                    j_warp = prev[j] + ptr_sm(i_state, i_m1) + nu_ + lambda_;
                } else if (j > 1) {
                    sub = inf;
                }

                // [OPT-2] Manual 3-way min.
                double best = sub;
                if (i_warp < best) best = i_warp;
                if (j_warp < best) best = j_warp;
                curr[j] = best;
            }
            std::swap(prev, curr);
        }

        double raw = prev[n];
        double maxpossiblecost = std::abs(n - m) * (nu_ + lambda_ + maxscost_) + 2.0 * (maxscost_ + nu_) * std::min(static_cast<double>(m), static_cast<double>(n));
        double ml = m * indel_;
        double nl = n * indel_;
        return normalize_distance(raw, maxpossiblecost, ml, nl, norm_);
    }

    // [OPT-1] Allocate 2 rows instead of fmatsize_² per thread.
    py::array_t<double> compute_all_distances() {
        auto buffer = dist_matrix_.mutable_unchecked<2>();

#pragma omp parallel
        {
            double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize_));
            double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize_));

#pragma omp for schedule(static)
            for (int i = 0; i < nseq_; i++) {
                buffer(i, i) = 0.0;
                for (int j = i + 1; j < nseq_; j++) {
                    double d = compute_distance(i, j, prev, curr);
                    buffer(i, j) = d;
                    buffer(j, i) = d;
                }
            }
            dp_utils::aligned_free_double(prev);
            dp_utils::aligned_free_double(curr);
        }
        return dist_matrix_;
    }

    py::array_t<double> compute_refseq_distances() {
        auto buffer = refdist_matrix_.mutable_unchecked<2>();

#pragma omp parallel
        {
            double* prev = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize_));
            double* curr = dp_utils::aligned_alloc_double(static_cast<size_t>(fmatsize_));

#pragma omp for schedule(static)
            for (int rseq = rseq1_; rseq < rseq2_; rseq++) {
                for (int is = 0; is < nseq_; is++) {
                    double d = (is == rseq) ? 0.0 : compute_distance(is, rseq, prev, curr);
                    buffer(is, rseq - rseq1_) = d;
                }
            }
            dp_utils::aligned_free_double(prev);
            dp_utils::aligned_free_double(curr);
        }
        return refdist_matrix_;
    }

private:
    py::array_t<int> sequences_;
    py::array_t<double> sm_;
    py::array_t<int> seqlength_;
    double indel_;
    int norm_;
    double nu_;
    double lambda_;
    int nseq_;
    int seqlen_;
    int alphasize_;
    int fmatsize_;
    double maxscost_ = 0.0;
    int rseq1_;
    int rseq2_;
    py::array_t<double> dist_matrix_;
    py::array_t<double> refdist_matrix_;
};
