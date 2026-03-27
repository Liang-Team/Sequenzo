#pragma once

#include <cstddef>
#include <string>
#include <vector>

// pybind11 is needed only for the vector_to_pyarray utility functions.
// All core prep/check functions use std::ptrdiff_t and are Python-independent.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct EuclideanCheckResult {
    bool compatible = true;
    double violation_rate = 0.0;
    double neg_energy_ratio = 0.0;
    int sample_n = 0;
};

enum WarningFlags : int {
    WARN_NONE = 0,
    WARN_NONFINITE = 1 << 0,
    WARN_NEGATIVE = 1 << 1,
    WARN_SYMMETRIZED = 1 << 2,
    WARN_WARD_NON_EUCLIDEAN = 1 << 3,
};

struct PreparedMatrixData {
    std::vector<double> full;
    std::vector<double> condensed;
    bool had_nonfinite = false;
    bool had_negative = false;
    bool was_symmetrized = false;
    double replacement_value = 0.0;
    std::ptrdiff_t n = 0;
    int warning_flags = WARN_NONE;
};

PreparedMatrixData prepare_distance_matrix_impl(
    const double* in_ptr,
    std::ptrdiff_t n,
    bool enforce_symmetry,
    double rtol,
    double atol,
    double replacement_quantile
);

struct PreparedCondensedData {
    std::vector<double> condensed;
    bool had_nonfinite = false;
    bool had_negative = false;
    double replacement_value = 0.0;
    std::ptrdiff_t n = 0;
    int warning_flags = WARN_NONE;
};

// Fast path: validate/clean a condensed distance array directly.
// Skips full-matrix construction and symmetry checks entirely.
PreparedCondensedData prepare_distance_condensed_impl(
    const double* in_ptr,
    std::ptrdiff_t condensed_len,
    std::ptrdiff_t n,
    double replacement_quantile
);

// Fast path: extract condensed directly from a square distance matrix in a
// single upper-triangle pass.  Avoids allocating a full N×N copy.
// Applies NaN/Inf replacement and negative-value clamping inline.
// Does NOT check symmetry or retain the full matrix.
PreparedCondensedData prepare_matrix_to_condensed_fast(
    const double* in_ptr,
    std::ptrdiff_t n,
    double replacement_quantile
);

// Fused full-path extraction: reads the upper triangle of an N×N matrix in a
// single pass, performs symmetrization, validation (NaN/Inf/negative), and
// writes the condensed array directly.  Never allocates a full N×N copy.
// check_symmetry: if true, detects asymmetry and sets WARN_SYMMETRIZED.
// Always symmetrizes inline (average of (i,j) and (j,i)) regardless of flag.
PreparedCondensedData prepare_matrix_to_condensed_fused(
    const double* in_ptr,
    std::ptrdiff_t n,
    double replacement_quantile,
    bool check_symmetry,
    double rtol,
    double atol
);

// Pure-C++ eigenvalue check (no NumPy dependency).
// Operates on a symmetric s×s distance sub-matrix (row-major).
EuclideanCheckResult check_euclidean_compatibility_pure(
    const double* matrix_ptr,
    std::ptrdiff_t n,
    const std::string& method
);

// Utility: move a std::vector<double> into a NumPy array with zero-copy.
py::array_t<double> vector_to_pyarray_2d(std::vector<double>&& data, py::ssize_t rows, py::ssize_t cols);
py::array_t<double> vector_to_pyarray_1d(std::vector<double>&& data);
