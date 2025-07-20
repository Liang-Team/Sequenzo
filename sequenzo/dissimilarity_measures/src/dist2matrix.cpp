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
    dist2matrix(int nseq_, py::array_t<int> seqdata_didxs_, py::array_t<double> dist_dseqs_num_)
        : nseq(nseq_), seqdata_didxs(seqdata_didxs_), dist_dseqs_num(dist_dseqs_num_) {

        py::print("[>] Computing all pairwise distances...");
        std::cout << std::flush;

        try {
            dist_matrix = py::array_t<double>({nseq, nseq});
        } catch (const std::exception& e) {
            py::print("Error in constructor: ", e.what());
            throw;
        }
    }

    py::array_t<double> padding_matrix() {
        try {
            auto idxs_buf = seqdata_didxs.unchecked<1>();
            auto dist_buf = dist_dseqs_num.unchecked<2>();
            auto buffer = dist_matrix.mutable_unchecked<2>();

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nseq; ++i) {
                for (int j = i; j < nseq; ++j) {
                    buffer(i, j) = dist_buf(idxs_buf(i), idxs_buf(j));
                }
            }

            for (int i = 0; i < nseq; ++i) {
                for (int j = 0; j < i; ++j) {
                    buffer(i, j) = buffer(j, i);
                }
            }

            return dist_matrix;

        } catch (const std::exception& e) {
            py::print("Error in padding_matrix: ", e.what());
            throw;
        }
    }

private:
    int nseq;
    py::array_t<int> seqdata_didxs;
    py::array_t<double> dist_dseqs_num;
    py::array_t<double> dist_matrix;
};