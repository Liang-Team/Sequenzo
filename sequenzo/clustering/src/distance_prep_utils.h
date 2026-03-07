#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

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
    ssize_t n = 0;
    int warning_flags = WARN_NONE;
};

PreparedMatrixData prepare_distance_matrix_impl(
    const double* in_ptr,
    ssize_t n,
    bool enforce_symmetry,
    double rtol,
    double atol,
    double replacement_quantile
);

EuclideanCheckResult check_euclidean_compatibility_impl(
    const double* matrix_ptr,
    ssize_t n,
    const std::string& method
);

py::array_t<double> vector_to_pyarray_2d(std::vector<double>&& data, ssize_t rows, ssize_t cols);
py::array_t<double> vector_to_pyarray_1d(std::vector<double>&& data);
