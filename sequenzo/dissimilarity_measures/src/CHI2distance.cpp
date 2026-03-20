/*
 * CHI2distance: Chi-square / Euclidean sequence distance.
 *
 * Optimizations vs original:
 * 1. Raw pointer cache (eliminates pybind11 unchecked<>() accessor per compute_distance call)
 * 2. Pre-computed 1/pdotj array (replaces division with multiplication in inner loop;
 *    division ~20 cycles vs multiplication ~4 cycles)
 * 3. OpenMP parallelization (original compute_all_distances was fully sequential;
 *    for EUCLID n=10000 L=30 this is ~50M pairs x 900 columns = 45B ops)
 * 4. Removed try/catch from hot path
 *
 * Note: EUCLID uses this same class. The Python layer builds allmat/pdotj differently
 * for EUCLID (uniform marginals) vs CHI2 (data-dependent marginals), but the C++
 * distance computation is identical.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <cmath>
#include <vector>
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class CHI2distance {
public:
    CHI2distance(py::array_t<double> allmat,
                 py::array_t<double> pdotj,
                 double norm_factor,
                 py::array_t<int> refseq_id) : norm_factor_(norm_factor) {
        allmat_ = allmat;
        pdotj_ = pdotj;

        n_ = static_cast<int>(allmat.shape(0));
        n_cols_ = static_cast<int>(allmat.shape(1));

        rseq1_ = refseq_id.at(0);
        rseq2_ = refseq_id.at(1);

        // [OPT-1] Cache raw pointer for allmat
        am_ptr_ = allmat.data();

        // [OPT-2] Pre-compute reciprocals: replace d*d/pdotj[c] with d*d*inv[c]
        const double* pj = pdotj.data();
        inv_pdotj_.resize(n_cols_);
        for (int c = 0; c < n_cols_; c++) {
            inv_pdotj_[c] = (pj[c] > 0.0) ? (1.0 / pj[c]) : 0.0;
        }
        inv_ptr_ = inv_pdotj_.data();
    }

    inline double compute_distance(int i, int j) const {
        const double* row_i = am_ptr_ + static_cast<ptrdiff_t>(i) * n_cols_;
        const double* row_j = am_ptr_ + static_cast<ptrdiff_t>(j) * n_cols_;
        const double* inv_pj = inv_ptr_;
        const int nc = n_cols_;

        double sum = 0.0;
        for (int c = 0; c < nc; c++) {
            double d = row_i[c] - row_j[c];
            sum += d * d * inv_pj[c];
        }
        return std::sqrt(sum);
    }

    // [OPT-3] Added OpenMP. Original was fully sequential loop.
    // For EUCLID n=10000 L=30: 50M pairs, ~900 columns each = ~45B operations.
    // Single-thread ~40s. With 8 cores: ~5-8s.
    py::array_t<double> compute_all_distances() {
        py::array_t<double> out({n_, n_});
        auto buf = out.mutable_unchecked<2>();
        const double nf = norm_factor_;

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < n_; i++) {
            buf(i, i) = 0.0;
            for (int j = i + 1; j < n_; j++) {
                double d = compute_distance(i, j) * nf;
                buf(i, j) = d;
                buf(j, i) = d;
            }
        }

        return out;
    }

    py::array_t<double> compute_refseq_distances() {
        const double nf = norm_factor_;

        if (rseq1_ < rseq2_) {
            int n1 = rseq1_;
            int n2 = rseq2_ - rseq1_;
            py::array_t<double> out({n1, n2});
            auto buf = out.mutable_unchecked<2>();

            #pragma omp parallel for schedule(dynamic, 4)
            for (int rseq = 0; rseq < n2; rseq++) {
                int ref_row = n1 + rseq;
                for (int i = 0; i < n1; i++) {
                    buf(i, rseq) = compute_distance(i, ref_row) * nf;
                }
            }

            return out;
        } else {
            int ref_row = rseq1_;
            py::array_t<double> out(std::array<py::ssize_t, 1>{static_cast<py::ssize_t>(n_)});
            auto buf = out.mutable_unchecked<1>();

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_; i++) {
                buf(i) = compute_distance(i, ref_row) * nf;
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

    const double* am_ptr_ = nullptr;
    std::vector<double> inv_pdotj_;
    const double* inv_ptr_ = nullptr;
};
