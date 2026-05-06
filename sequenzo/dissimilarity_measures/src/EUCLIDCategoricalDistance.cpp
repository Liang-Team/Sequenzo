/*
 * EUCLIDCategoricalDistance: exact fast path for default categorical EUCLID.
 *
 * For complete categorical sequences with step=1 and no custom breaks, the
 * CHI2/EUCLID construction produces one one-hot state vector per time point.
 * The Euclidean distance is therefore a square-root transform of the Hamming
 * mismatch count:
 *   norm="auto": sqrt(mismatches / L)
 *   norm="none": sqrt(2 * mismatches)
 *
 * This avoids the dense L*S one-hot representation used by CHI2distance.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class EUCLIDCategoricalDistance {
public:
    EUCLIDCategoricalDistance(py::array_t<int> sequences,
                              bool norm_auto,
                              py::array_t<int> refseqS)
            : sequences_(sequences),
              norm_auto_(norm_auto) {
        py::print("[>] Starting categorical EUCLID fast path...");
        std::cout << std::flush;

        auto seq_shape = sequences_.shape();
        nseq_ = static_cast<int>(seq_shape[0]);
        len_ = static_cast<int>(seq_shape[1]);
        seq_ptr_ = sequences_.data();

        dist_matrix_ = py::array_t<double>({nseq_, nseq_});

        rseq1_ = refseqS.at(0);
        rseq2_ = refseqS.at(1);
        if (rseq1_ >= 0 && rseq2_ >= 0) {
            if (rseq1_ < rseq2_) {
                ref_nseq_ = rseq1_;
            } else {
                ref_nseq_ = nseq_;
                rseq1_ = rseq1_ - 1;
            }
            refdist_matrix_ = py::array_t<double>({ref_nseq_, (rseq2_ - rseq1_)});
        }
    }

    inline double compute_distance(int is, int js) const {
        const int* row_i = seq_ptr_ + static_cast<ptrdiff_t>(is) * len_;
        const int* row_j = seq_ptr_ + static_cast<ptrdiff_t>(js) * len_;
        int mismatches = 0;
        for (int k = 0; k < len_; ++k) {
            mismatches += (row_i[k] != row_j[k]);
        }

        if (norm_auto_) {
            return len_ > 0 ? std::sqrt(static_cast<double>(mismatches) / static_cast<double>(len_)) : 0.0;
        }
        return std::sqrt(2.0 * static_cast<double>(mismatches));
    }

    py::array_t<double> compute_all_distances() {
        auto buffer = dist_matrix_.mutable_unchecked<2>();

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < nseq_; ++i) {
            buffer(i, i) = 0.0;
            for (int j = i + 1; j < nseq_; ++j) {
                buffer(i, j) = compute_distance(i, j);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nseq_; ++i) {
            for (int j = i + 1; j < nseq_; ++j) {
                buffer(j, i) = buffer(i, j);
            }
        }

        return dist_matrix_;
    }

    py::array_t<double> compute_refseq_distances() {
        if (rseq1_ < 0 || rseq2_ < 0) {
            throw std::runtime_error("refseq distances require a valid refseq range.");
        }

        auto buffer = refdist_matrix_.mutable_unchecked<2>();

        #pragma omp parallel for schedule(dynamic, 4)
        for (int rseq = rseq1_; rseq < rseq2_; ++rseq) {
            for (int is = 0; is < ref_nseq_; ++is) {
                buffer(is, rseq - rseq1_) = (is == rseq) ? 0.0 : compute_distance(is, rseq);
            }
        }

        return refdist_matrix_;
    }

private:
    py::array_t<int> sequences_;
    bool norm_auto_;
    int nseq_ = 0;
    int len_ = 0;
    const int* seq_ptr_ = nullptr;
    py::array_t<double> dist_matrix_;

    int rseq1_ = -1;
    int rseq2_ = -1;
    int ref_nseq_ = 0;
    py::array_t<double> refdist_matrix_;
};
