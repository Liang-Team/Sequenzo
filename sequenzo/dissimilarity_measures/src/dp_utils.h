#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <cstdlib>
#include <new>

namespace dp_utils {

// Cross-platform aligned allocation for double buffers
#ifdef _WIN32
inline double* aligned_alloc_double(size_t size, size_t align = 64) {
    return reinterpret_cast<double*>(_aligned_malloc(size * sizeof(double), align));
}
inline void aligned_free_double(double* ptr) {
    _aligned_free(ptr);
}
#else
inline double* aligned_alloc_double(size_t size, size_t align = 64) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, align, size * sizeof(double)) != 0) throw std::bad_alloc();
    return reinterpret_cast<double*>(ptr);
}
inline void aligned_free_double(double* ptr) { free(ptr); }
#endif

// Generic pairwise symmetric computation helper
// ComputeFn signature: double(int i, int j, double* prev, double* curr)
template <typename ComputeFn>
inline pybind11::array_t<double> compute_all_distances(
    int nseq,
    int fmatsize,
    pybind11::array_t<double>& dist_matrix,
    ComputeFn&& compute_fn
) {
    auto buffer = dist_matrix.mutable_unchecked<2>();

    #pragma omp parallel
    {
        double* prev = aligned_alloc_double(static_cast<size_t>(fmatsize));
        double* curr = aligned_alloc_double(static_cast<size_t>(fmatsize));

        #pragma omp for schedule(static)
        for (int i = 0; i < nseq; i++) {
            for (int j = i; j < nseq; j++) {
                buffer(i, j) = compute_fn(i, j, prev, curr);
            }
        }

        aligned_free_double(prev);
        aligned_free_double(curr);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nseq; i++) {
        for (int j = i + 1; j < nseq; j++) {
            buffer(j, i) = buffer(i, j);
        }
    }

    return dist_matrix;
}

// Generic pairwise symmetric computation helper (no buffers)
// ComputeFn signature: double(int i, int j)
template <typename ComputeFn>
inline pybind11::array_t<double> compute_all_distances_simple(
    int nseq,
    pybind11::array_t<double>& dist_matrix,
    ComputeFn&& compute_fn
) {
    auto buffer = dist_matrix.mutable_unchecked<2>();

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < nseq; i++) {
            for (int j = i; j < nseq; j++) {
                buffer(i, j) = compute_fn(i, j);
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nseq; ++i) {
        for (int j = i + 1; j < nseq; ++j) {
            buffer(j, i) = buffer(i, j);
        }
    }

    return dist_matrix;
}

// Generic reference-sequence computation helper (no buffers)
// ComputeFn signature: double(int is, int rseq)
template <typename ComputeFn>
inline pybind11::array_t<double> compute_refseq_distances_simple(
    int nseq,
    int rseq1,
    int rseq2,
    pybind11::array_t<double>& refdist_matrix,
    ComputeFn&& compute_fn
) {
    auto buffer = refdist_matrix.mutable_unchecked<2>();

    #pragma omp parallel
    {
        #pragma omp for schedule(guided)
        for (int rseq = rseq1; rseq < rseq2; rseq++) {
            for (int is = 0; is < nseq; is++) {
                buffer(is, rseq - rseq1) = (is == rseq) ? 0.0 : compute_fn(is, rseq);
            }
        }
    }

    return refdist_matrix;
}

// Generic reference-sequence computation helper (with DP buffers)
// ComputeFn signature: double(int is, int rseq, double* prev, double* curr)
template <typename ComputeFn>
inline pybind11::array_t<double> compute_refseq_distances_buffered(
    int nseq,
    int rseq1,
    int rseq2,
    int fmatsize,
    pybind11::array_t<double>& refdist_matrix,
    ComputeFn&& compute_fn
) {
    auto buffer = refdist_matrix.mutable_unchecked<2>();

    #pragma omp parallel
    {
        double* prev = aligned_alloc_double(static_cast<size_t>(fmatsize));
        double* curr = aligned_alloc_double(static_cast<size_t>(fmatsize));

        #pragma omp for schedule(static)
        for (int rseq = rseq1; rseq < rseq2; rseq++) {
            for (int is = 0; is < nseq; is++) {
                double cmpres = 0.0;
                if (is != rseq) {
                    cmpres = compute_fn(is, rseq, prev, curr);
                }
                buffer(is, rseq - rseq1) = cmpres;
            }
        }

        aligned_free_double(prev);
        aligned_free_double(curr);
    }

    return refdist_matrix;
}

} // namespace dp_utils


