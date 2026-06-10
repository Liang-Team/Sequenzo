/*
 * TWEDdistance: Time Warp Edit Distance (TraMineR-aligned).
 *
 * @Author  : Yuqi Liang 梁彧祺, Yapeng Wei 卫亚鹏
 * @File    : TWEDdistance.cpp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
#include "utils.h"
#include "dp_utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

static inline double normalize_twed_distance(double rawdist, double maxdist, double l1, double l2, int norm) {
    if (norm == 4) {
        if (std::fabs(rawdist) < EPS) return 0.0;
        return std::fabs(maxdist) < EPS ? 1.0 : (2.0 * rawdist) / (rawdist + maxdist);
    }
    return normalize_distance(rawdist, maxdist, l1, l2, norm);
}

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

        const int ref_a = refseq_id.at(0);
        const int ref_b = refseq_id.at(1);
        if (ref_a < 0 && ref_b < 0) {
            rseq1_ = 0;
            rseq2_ = 0;
            has_refseq_ = false;
        } else if (ref_a < ref_b) {
            rseq1_ = ref_a;
            rseq2_ = ref_b;
            nseq_ = rseq1_;
            has_refseq_ = true;
            refdist_matrix_ = py::array_t<double>(std::array<py::ssize_t, 2>{static_cast<py::ssize_t>(nseq_), static_cast<py::ssize_t>(rseq2_ - rseq1_)});
        } else {
            rseq1_ = ref_a - 1;
            rseq2_ = ref_b;
            has_refseq_ = true;
            refdist_matrix_ = py::array_t<double>(std::array<py::ssize_t, 2>{static_cast<py::ssize_t>(nseq_), 1});
        }
    }

    // Uses 2-row rolling DP buffers shared within each worker thread.
    double compute_distance(int is, int js, double* prev, double* curr) const {
        auto ptr_len = seqlength_.unchecked<1>();
        int m = ptr_len(is);
        int n = ptr_len(js);

        if (m == 0 && n == 0)
            return normalize_twed_distance(0.0, 0.0, 0.0, 0.0, norm_);
        if (m == 0) {
            double cost = n * indel_;
            double maxcost = std::abs(n - m) * (nu_ + lambda_ + maxscost_) + 2.0 * (maxscost_ + nu_) * std::min(m, n);
            return normalize_twed_distance(cost, maxcost, 0.0, n * indel_, norm_);
        }
        if (n == 0) {
            double cost = m * indel_;
            double maxcost = std::abs(n - m) * (nu_ + lambda_ + maxscost_) + 2.0 * (maxscost_ + nu_) * std::min(m, n);
            return normalize_twed_distance(cost, maxcost, m * indel_, 0.0, norm_);
        }

        auto ptr_seq = sequences_.unchecked<2>();
        auto ptr_sm = sm_.unchecked<2>();
        const double inf = std::numeric_limits<double>::infinity();

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
        return normalize_twed_distance(raw, maxpossiblecost, ml, nl, norm_);
    }

    // Allocate 2 rows instead of fmatsize_² per worker thread.
    py::array_t<double> compute_all_distances() {
        py::array_t<double> dist_matrix(std::array<py::ssize_t, 2>{static_cast<py::ssize_t>(nseq_), static_cast<py::ssize_t>(nseq_)});
        return dp_utils::compute_all_distances(
            nseq_,
            fmatsize_,
            dist_matrix,
            [this](int i, int j, double* prev, double* curr) {
                return this->compute_distance(i, j, prev, curr);
            }
        );
    }

    py::array_t<double> compute_condensed_distances() {
        return dp_utils::compute_condensed_distances(
            nseq_,
            fmatsize_,
            [this](int i, int j, double* prev, double* curr) {
                return this->compute_distance(i, j, prev, curr);
            }
        );
    }

    py::array_t<double> compute_refseq_distances() {
        if (!has_refseq_) {
            throw std::runtime_error("TWED refseq distances requested without a refseq configuration.");
        }
        return dp_utils::compute_refseq_distances_buffered(
            nseq_,
            rseq1_,
            rseq2_,
            fmatsize_,
            refdist_matrix_,
            [this](int is, int rseq, double* prev, double* curr) {
                return this->compute_distance(is, rseq, prev, curr);
            }
        );
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
    bool has_refseq_ = false;
    py::array_t<double> refdist_matrix_;
};
