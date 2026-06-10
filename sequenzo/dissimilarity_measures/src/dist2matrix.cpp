#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

py::array_t<double> unique_condensed_to_condensed(
    int nseq,
    py::array_t<int> seqdata_didxs,
    py::array_t<double> unique_condensed,
    int nunique
) {
    if (nseq < 0 || nunique < 0) {
        throw std::invalid_argument("nseq and nunique must be non-negative.");
    }
    if (seqdata_didxs.ndim() != 1 || seqdata_didxs.shape(0) < nseq) {
        throw std::invalid_argument("seqdata_didxs must be a one-dimensional array with length at least nseq.");
    }
    const long long unique_len = static_cast<long long>(nunique) * (nunique - 1) / 2;
    if (unique_condensed.ndim() != 1 || unique_condensed.shape(0) != unique_len) {
        throw std::invalid_argument("unique_condensed has an unexpected length.");
    }

    const long long condensed_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
    auto condensed = py::array_t<double>(condensed_len);
    const int* idxs = seqdata_didxs.data();
    const double* unique = unique_condensed.data();
    double* out = condensed.mutable_data();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nseq - 1; ++i) {
        const int left = idxs[i];
        if (left < 0 || left >= nunique) {
            continue;
        }
        const long long row_start = static_cast<long long>(i) * (2 * nseq - i - 1) / 2;
        for (int j = i + 1; j < nseq; ++j) {
            const int right = idxs[j];
            const long long out_pos = row_start + (j - i - 1);
            if (right < 0 || right >= nunique) {
                out[out_pos] = std::numeric_limits<double>::quiet_NaN();
            } else if (left == right) {
                out[out_pos] = 0.0;
            } else {
                const int u = left < right ? left : right;
                const int v = left < right ? right : left;
                const long long in_pos = static_cast<long long>(nunique) * u
                    - (static_cast<long long>(u) * (u + 1)) / 2
                    + (v - u - 1);
                out[out_pos] = unique[in_pos];
            }
        }
    }

    for (int i = 0; i < nseq; ++i) {
        const int idx = idxs[i];
        if (idx < 0 || idx >= nunique) {
            throw std::out_of_range("seqdata_didxs contains an out-of-range unique-sequence index.");
        }
    }

    return condensed;
}

class dist2matrix {
public:
    dist2matrix(int nseq, py::array_t<int> seqdata_didxs, py::array_t<double> dist_dseqs_num)
            : nseq(nseq) {

        py::print("[>] Computing all pairwise distances...");
        std::cout << std::flush;

        try {
            this->seqdata_didxs = seqdata_didxs;
            this->dist_dseqs_num = dist_dseqs_num;
        } catch (const std::exception& e) {
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    // Expand unique-sequence distances back to original sequence order.
    py::array_t<double> padding_matrix() {
        dist_matrix = py::array_t<double>({nseq, nseq});
        const int* idxs = seqdata_didxs.data();
        const double* dist = dist_dseqs_num.data();
        double* out = dist_matrix.mutable_data();
        const int nunique = static_cast<int>(dist_dseqs_num.shape(0));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nseq; ++i) {
            const int ui = idxs[i];
            const double* dist_row = dist + static_cast<ptrdiff_t>(ui) * nunique;
            double* out_row = out + static_cast<ptrdiff_t>(i) * nseq;
            for (int j = i; j < nseq; ++j) {
                out_row[j] = dist_row[idxs[j]];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nseq; ++i) {
            for (int j = i + 1; j < nseq; ++j) {
                out[static_cast<ptrdiff_t>(j) * nseq + i] = out[static_cast<ptrdiff_t>(i) * nseq + j];
            }
        }

        return dist_matrix;
    }

    py::array_t<double> padding_condensed() {
        const int* idxs = seqdata_didxs.data();
        const double* dist = dist_dseqs_num.data();
        const int nunique = static_cast<int>(dist_dseqs_num.shape(0));

        const long long condensed_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
        auto condensed = py::array_t<double>(condensed_len);
        auto* out_ptr = static_cast<double*>(condensed.request().ptr);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nseq; ++i) {
            const int ui = idxs[i];
            const double* dist_row = dist + static_cast<ptrdiff_t>(ui) * nunique;
            const long long row_start = static_cast<long long>(i) * (2 * nseq - i - 1) / 2;
            for (int j = i + 1; j < nseq; ++j) {
                out_ptr[row_start + (j - i - 1)] = dist_row[idxs[j]];
            }
        }

        return condensed;
    }

private:
    py::array_t<int> seqdata_didxs;
    py::array_t<double> dist_dseqs_num;
    int nseq = 0;
    py::array_t<double> dist_matrix;
};
