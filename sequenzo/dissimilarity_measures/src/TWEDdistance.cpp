/*
 * TWEDdistance: Time Warp Edit Distance (TraMineR-aligned).
 *
 * Implements TWED as in Marteau (2009). Uses substitution cost matrix sm,
 * stiffness nu (> 0), and gap penalty lambda (h). Does not strip common
 * prefix/suffix (TraMineR TWED keeps them). Normalization: same as OM (none,
 * maxlength, gmean, maxdist, YujianBo).
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : TWEDdistance.py
 * @Time    : 2026/02/07 15:07
 * @Desc    : 
 * Reference: TraMineR src/TWEDdistance.cpp, seqdist.R method="TWED".
 * Cross-check: tests/dissimilarity_measures/new_measures/test_dissimilarity_measures_traminer.py (Part 3 TWED) vs tests/dissimilarity_measures/twed/traminer_twed_reference.R.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <cmath>
#include <algorithm>
#include <vector>
#include "utils.h"
#include "dp_utils.h"

namespace py = pybind11;

class TWEDdistance {
public:
    /*
     * Constructor.
     * - sequences: (nseq, maxlen) int, state codes 1..alphasize; 0 reserved for dummy.
     * - sm: (alphasize+1, alphasize+1) substitution cost; sm(0,0)=0, sm(0,k)=sm(k,0)=indel for dummy.
     * - indel: used for empty sequences and for sm(0,:)/sm(:,0).
     * - norm: 0=none, 1=maxlength, 2=gmean, 3=maxdist, 4=YujianBo.
     * - nu: stiffness parameter (must be > 0).
     * - lambda: gap penalty (h in R, >= 0).
     * - seqlength: (nseq,) valid length per sequence.
     * - refseq_id: [rseq1, rseq2].
     */
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
        alphasize_ = static_cast<int>(sm.shape()[0]) - 1;  // sm is (alphasize+1) x (alphasize+1)
        fmatsize_ = seqlen_ + 1;

        // maxscost for normalization (YujianBo uses 2*indel; else max over sm)
        if (norm == 4) {
            maxscost_ = 2.0 * indel;
        } else {
            auto ptr = sm.unchecked<2>();
            maxscost_ = 0.0;
            for (int i = 0; i <= alphasize_; i++) {
                for (int j = i + 1; j <= alphasize_; j++) {
                    double c = ptr(i, j);
                    if (c > maxscost_) maxscost_ = c;
                }
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

    double compute_distance(int is, int js, double* fmat) const {
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

        // Initialize boundaries: fmat(0,0)=0, fmat(i,0)=i*indel, fmat(0,j)=j*indel
        fmat[0] = 0.0;
        for (int i = 1; i <= m; i++) fmat[i * fmatsize_] = i * indel_;
        for (int j = 1; j <= n; j++) fmat[j] = j * indel_;

        for (int i = 1; i <= m; i++) {
            int i_state = ptr_seq(is, i - 1);
            int i_m1 = (i == 1) ? 0 : ptr_seq(is, i - 2);
            for (int j = 1; j <= n; j++) {
                int j_state = ptr_seq(js, j - 1);
                int j_m1 = (j == 1) ? 0 : ptr_seq(js, j - 2);

                double cost;
                if ((i_state == j_state) && (i_m1 == j_m1)) {
                    cost = 0.0;
                } else {
                    cost = ptr_sm(i_state, j_state) + ptr_sm(i_m1, j_m1);
                }
                double sub = fmat[(i - 1) * fmatsize_ + (j - 1)] + cost + 2.0 * nu_ * std::abs(i - j);

                double i_warp = inf;
                if (j > 1) {
                    i_warp = fmat[i * fmatsize_ + (j - 1)] + ptr_sm(j_state, j_m1) + nu_ + lambda_;
                } else if (i > 1) {
                    sub = inf;
                }

                double j_warp = inf;
                if (i > 1) {
                    j_warp = fmat[(i - 1) * fmatsize_ + j] + ptr_sm(i_state, i_m1) + nu_ + lambda_;
                } else if (j > 1) {
                    sub = inf;
                }

                double minimum = std::min({sub, i_warp, j_warp});
                fmat[i * fmatsize_ + j] = minimum;
            }
        }

        double raw = fmat[m * fmatsize_ + n];
        double maxpossiblecost = std::abs(n - m) * (nu_ + lambda_ + maxscost_) + 2.0 * (maxscost_ + nu_) * std::min(static_cast<double>(m), static_cast<double>(n));
        double ml = m * indel_;
        double nl = n * indel_;
        return normalize_distance(raw, maxpossiblecost, ml, nl, norm_);
    }

    py::array_t<double> compute_all_distances() {
        auto buffer = dist_matrix_.mutable_unchecked<2>();
        std::vector<double> fmat(static_cast<size_t>(fmatsize_) * fmatsize_);

#pragma omp parallel
        {
            std::vector<double> fmat_local(static_cast<size_t>(fmatsize_) * fmatsize_);
            double* pf = fmat_local.data();

#pragma omp for schedule(static)
            for (int i = 0; i < nseq_; i++) {
                for (int j = i; j < nseq_; j++) {
                    double d = compute_distance(i, j, pf);
                    buffer(i, j) = d;
                    buffer(j, i) = d;
                }
            }
        }
        return dist_matrix_;
    }

    py::array_t<double> compute_refseq_distances() {
        auto buffer = refdist_matrix_.mutable_unchecked<2>();
        std::vector<double> fmat(static_cast<size_t>(fmatsize_) * fmatsize_);

#pragma omp parallel
        {
            std::vector<double> fmat_local(static_cast<size_t>(fmatsize_) * fmatsize_);
            double* pf = fmat_local.data();

#pragma omp for schedule(static)
            for (int rseq = rseq1_; rseq < rseq2_; rseq++) {
                for (int is = 0; is < nseq_; is++) {
                    double d = (is == rseq) ? 0.0 : compute_distance(is, rseq, pf);
                    buffer(is, rseq - rseq1_) = d;
                }
            }
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
    double maxscost_;
    int rseq1_;
    int rseq2_;
    py::array_t<double> dist_matrix_;
    py::array_t<double> refdist_matrix_;
};
