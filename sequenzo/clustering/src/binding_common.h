#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

void validate_square_matrix(const py::buffer_info& matrix_buf, const char* msg);
void validate_vector_length(ssize_t actual, ssize_t expected, const char* msg);
void validate_condensed_size(ssize_t actual, int n, const char* msg);

py::dict build_asw_result(py::array_t<double> asw_i, py::array_t<double> asw_w);
py::dict stats_vector_to_dict(const std::vector<double>& stats);

bool is_nan_ieee(double x);
