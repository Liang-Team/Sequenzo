#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

void validate_linkage(py::buffer_info& linkage_buf, int n);
void compute_labels_from_linkage(
    const double* linkage_ptr,
    int n,
    int nclusters,
    int* labels_out
);
