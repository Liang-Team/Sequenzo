#pragma once

// MSVC uses __restrict; GCC/Clang use __restrict__
#ifdef _MSC_VER
  #define RESTRICT __restrict
#else
  #define RESTRICT __restrict__
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <atomic>
#include <cstdlib>
#include <cstddef>
#include <exception>
#include <memory>
#include <new>

namespace dp_utils {

#ifdef _WIN32
inline double* aligned_alloc_double(size_t size, size_t align = 64) {
    double* ptr = reinterpret_cast<double*>(_aligned_malloc(size * sizeof(double), align));
    if (ptr == nullptr) throw std::bad_alloc();
    return ptr;
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

struct aligned_double_deleter {
    void operator()(double* ptr) const noexcept { aligned_free_double(ptr); }
};

using aligned_double_ptr = std::unique_ptr<double, aligned_double_deleter>;

inline aligned_double_ptr make_aligned_double_buffer(size_t size) {
    return aligned_double_ptr(aligned_alloc_double(size));
}

inline void record_first_exception(std::exception_ptr& first_exception, std::atomic<bool>& has_exception) {
    if (!has_exception.exchange(true)) {
        #pragma omp critical(dp_utils_exception)
        {
            if (!first_exception) {
                first_exception = std::current_exception();
            }
        }
    }
}

// Dynamic scheduling balances triangular pairwise computation.
// Row 0 computes nseq-1 pairs, row nseq-1 computes 0 pairs.
// schedule(static) gives thread 0 the heaviest rows; dynamic rebalances.
//
// Skip diagonal: d(i,i) = 0 always. This avoids one full DP computation
// per unique sequence.
template <typename ComputeFn>
inline pybind11::array_t<double> compute_all_distances(
    int nseq,
    int fmatsize,
    pybind11::array_t<double>& dist_matrix,
    ComputeFn&& compute_fn
) {
    auto buffer = dist_matrix.mutable_unchecked<2>();
    std::exception_ptr first_exception;
    std::atomic<bool> has_exception(false);

    #pragma omp parallel
    {
        aligned_double_ptr prev;
        aligned_double_ptr curr;
        bool ready = true;

        try {
            prev = make_aligned_double_buffer(static_cast<size_t>(fmatsize));
            curr = make_aligned_double_buffer(static_cast<size_t>(fmatsize));
        } catch (...) {
            ready = false;
            record_first_exception(first_exception, has_exception);
        }

        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < nseq; i++) {
            if (!ready || has_exception.load()) continue;
            buffer(i, i) = 0.0;
            for (int j = i + 1; j < nseq; j++) {
                if (has_exception.load()) break;
                try {
                    buffer(i, j) = compute_fn(i, j, prev.get(), curr.get());
                } catch (...) {
                    record_first_exception(first_exception, has_exception);
                    break;
                }
            }
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }

    // Mirror upper triangle to lower
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nseq; i++) {
        for (int j = i + 1; j < nseq; j++) {
            buffer(j, i) = buffer(i, j);
        }
    }

    return dist_matrix;
}

inline long long condensed_index(int nseq, int i, int j) {
    return static_cast<long long>(nseq) * i
        - (static_cast<long long>(i) * (i + 1)) / 2
        + (j - i - 1);
}

template <typename ComputeFn>
inline pybind11::array_t<double> compute_condensed_distances(
    int nseq,
    int fmatsize,
    ComputeFn&& compute_fn
) {
    const long long condensed_len = static_cast<long long>(nseq) * (nseq - 1) / 2;
    pybind11::array_t<double> condensed(
        std::array<pybind11::ssize_t, 1>{static_cast<pybind11::ssize_t>(condensed_len)}
    );
    auto buffer = condensed.mutable_unchecked<1>();
    std::exception_ptr first_exception;
    std::atomic<bool> has_exception(false);

    #pragma omp parallel
    {
        aligned_double_ptr prev;
        aligned_double_ptr curr;
        bool ready = true;

        try {
            prev = make_aligned_double_buffer(static_cast<size_t>(fmatsize));
            curr = make_aligned_double_buffer(static_cast<size_t>(fmatsize));
        } catch (...) {
            ready = false;
            record_first_exception(first_exception, has_exception);
        }

        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < nseq - 1; i++) {
            if (!ready || has_exception.load()) continue;
            for (int j = i + 1; j < nseq; j++) {
                if (has_exception.load()) break;
                try {
                    buffer(condensed_index(nseq, i, j)) = compute_fn(i, j, prev.get(), curr.get());
                } catch (...) {
                    record_first_exception(first_exception, has_exception);
                    break;
                }
            }
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }

    return condensed;
}

template <typename ComputeFn>
inline pybind11::array_t<double> compute_all_distances_simple(
    int nseq,
    pybind11::array_t<double>& dist_matrix,
    ComputeFn&& compute_fn
) {
    auto buffer = dist_matrix.mutable_unchecked<2>();

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < nseq; i++) {
            buffer(i, i) = 0.0;
            for (int j = i + 1; j < nseq; j++) {
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
        #pragma omp for schedule(dynamic, 4)
        for (int rseq = rseq1; rseq < rseq2; rseq++) {
            for (int is = 0; is < nseq; is++) {
                buffer(is, rseq - rseq1) = (is == rseq) ? 0.0 : compute_fn(is, rseq);
            }
        }
    }

    return refdist_matrix;
}

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
    const int nref = rseq2 - rseq1;
    const long long total = static_cast<long long>(nseq) * static_cast<long long>(nref);
    std::exception_ptr first_exception;
    std::atomic<bool> has_exception(false);

    #pragma omp parallel
    {
        aligned_double_ptr prev;
        aligned_double_ptr curr;
        bool ready = true;

        try {
            prev = make_aligned_double_buffer(static_cast<size_t>(fmatsize));
            curr = make_aligned_double_buffer(static_cast<size_t>(fmatsize));
        } catch (...) {
            ready = false;
            record_first_exception(first_exception, has_exception);
        }

        #pragma omp for schedule(dynamic, 16)
        for (long long idx = 0; idx < total; idx++) {
            if (!ready || has_exception.load()) continue;
            const int is = static_cast<int>(idx / nref);
            const int col = static_cast<int>(idx % nref);
            const int rseq = rseq1 + col;
            if (is == rseq) {
                buffer(is, col) = 0.0;
                continue;
            }
            try {
                buffer(is, col) = compute_fn(is, rseq, prev.get(), curr.get());
            } catch (...) {
                record_first_exception(first_exception, has_exception);
            }
        }
    }

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }

    return refdist_matrix;
}

} // namespace dp_utils
