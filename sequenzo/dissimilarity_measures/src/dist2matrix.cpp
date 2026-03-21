#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#ifdef _OPENMP
    #include <omp.h>
#endif

namespace py = pybind11;

class dist2matrix {
public:
    dist2matrix(int nseq, py::array_t<int> seqdata_didxs, py::array_t<double> dist_dseqs_num)
            : nseq(nseq) {

        py::print("[>] Computing all pairwise distances...");
        std::cout << std::flush;

        try {
            this->seqdata_didxs = seqdata_didxs;
            this->dist_dseqs_num = dist_dseqs_num;

            dist_matrix = py::array_t<double>({nseq, nseq});
        } catch (const std::exception& e) {
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    // [OPT-12] Raw pointer expansion replaces pybind11 unchecked<>() accessors.
    // For n=10000, this loop runs ~50M iterations. Each accessor call has overhead
    // from stride computation. Raw pointers with pre-computed row offsets eliminate this.
    //
    // Hoisting dist_row outside the j-loop means we access the unique distance matrix
    // row-sequentially within each i-iteration, which is cache-friendly.
    py::array_t<double> padding_matrix() {
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
