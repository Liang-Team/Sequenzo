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

    py::array_t<double> padding_matrix() {
        try {
            auto buffer = dist_matrix.mutable_unchecked<2>();

            #pragma omp for schedule(static)
            for(int i=0; i < nseq; i++){
                for(int j=i; j < nseq; j++){
                    int idx_i = seqdata_didxs.at(i);
                    int idx_j = seqdata_didxs.at(j);

                    buffer(i, j) = dist_dseqs_num.at(idx_i, idx_j);
                    buffer(j, i) = buffer(i, j);
                }
            }

            return dist_matrix;
        } catch (const std::exception& e) {
            py::print("Error in compute_all_distances: ", e.what());
            throw;
        }
    }

private:
    py::array_t<int> seqdata_didxs;
    py::array_t<double> dist_dseqs_num;
    int nseq = 0;

    py::array_t<double> dist_matrix;
};