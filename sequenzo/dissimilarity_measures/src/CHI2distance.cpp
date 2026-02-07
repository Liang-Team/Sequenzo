/*
 * CHI2distance: Chi-square / Euclidean sequence distance (TraMineR-aligned).
 *
 * Computes d(i,j) = sqrt(sum_c (allmat(i,c)-allmat(j,c))^2 / pdotj(c)),
 * optionally scaled by 1/sqrt(n_breaks). Matches TraMineR chisq.cpp (tmrChisq / tmrChisqRef).
 * Python builds allmat and pdotj; this module does the distance loops only.
 *
 * @Author  : Yuqi Liang 梁彧祺
 * @File    : CHI2distance.py
 * @Time    : 2026/02/05 23:27
 * @Desc    : C++ port of TraMineR src/chisq.cpp for CHI2 and EUCLID methods.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <cmath>

namespace py = pybind11;

class CHI2distance {
public:
    /*
     * Constructor.
     * - allmat: (n, n_cols) row-major, proportion table per (sequence, column).
     * - pdotj: (n_cols,) marginal weights for each column (denominator); must be > 0 where used.
     * - norm_factor: 1.0 or 1.0/sqrt(n_breaks) to apply to all distances.
     * - refseq_id: [rseq1, rseq2]. If rseq1 < rseq2: ref distances set1 vs set2 (n1=rseq1, n2=rseq2-rseq1).
     *             If rseq1 >= rseq2: single reference at row index rseq1.
     */
    CHI2distance(py::array_t<double> allmat,
                 py::array_t<double> pdotj,
                 double norm_factor,
                 py::array_t<int> refseq_id) : norm_factor_(norm_factor) {
        allmat_ = allmat;
        pdotj_ = pdotj;

        auto sh = allmat.unchecked<2>();
        n_ = static_cast<int>(sh.shape(0));
        n_cols_ = static_cast<int>(sh.shape(1));

        rseq1_ = refseq_id.at(0);
        rseq2_ = refseq_id.at(1);
    }

    /* Raw chi-square distance between row i and row j (no norm_factor). */
    double compute_distance(int i, int j) const {
        auto am = allmat_.unchecked<2>();
        auto pj = pdotj_.unchecked<1>();
        double sum = 0.0;
        for (int c = 0; c < n_cols_; c++) {
            double d = am(i, c) - am(j, c);
            double m = pj(c);
            if (m > 0.0)
                sum += (d * d) / m;
        }
        return std::sqrt(sum);
    }

    /* Pairwise: fill full (n, n) symmetric matrix and scale by norm_factor. */
    py::array_t<double> compute_all_distances() {
        py::array_t<double> out({n_, n_});
        auto buf = out.mutable_unchecked<2>();
        for (int i = 0; i < n_; i++) {
            buf(i, i) = 0.0;
            for (int j = i + 1; j < n_; j++) {
                double d = compute_distance(i, j) * norm_factor_;
                buf(i, j) = d;
                buf(j, i) = d;
            }
        }
        return out;
    }

    /*
     * Reference-sequence distances.
     * If rseq1_ < rseq2_: return (n1, n2) with n1=rseq1_, n2=rseq2_-rseq1_ (set1 vs set2).
     * Else: return (n_) vector of distances to reference row rseq1_.
     */
    py::array_t<double> compute_refseq_distances() {
        if (rseq1_ < rseq2_) {
            int n1 = rseq1_;
            int n2 = rseq2_ - rseq1_;
            py::array_t<double> out({n1, n2});
            auto buf = out.mutable_unchecked<2>();
            for (int rseq = 0; rseq < n2; rseq++) {
                int ref_row = n1 + rseq;
                for (int i = 0; i < n1; i++) {
                    buf(i, rseq) = compute_distance(i, ref_row) * norm_factor_;
                }
            }
            return out;
        } else {
            int ref_row = rseq1_;
            py::array_t<double> out(std::array<py::ssize_t, 1>{static_cast<py::ssize_t>(n_)});
            auto buf = out.mutable_unchecked<1>();
            for (int i = 0; i < n_; i++) {
                buf(i) = compute_distance(i, ref_row) * norm_factor_;
            }
            return out;
        }
    }

private:
    py::array_t<double> allmat_;
    py::array_t<double> pdotj_;
    double norm_factor_;
    int n_;
    int n_cols_;
    int rseq1_;
    int rseq2_;
};
