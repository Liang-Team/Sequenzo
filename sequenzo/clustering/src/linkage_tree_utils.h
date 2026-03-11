#pragma once

#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

void validate_linkage(py::buffer_info& linkage_buf, int n);
void compute_labels_from_linkage(
    const double* linkage_ptr,
    int n,
    int nclusters,
    int* labels_out
);

struct ClusterResultData {
    std::vector<int> labels;             // n elements, 1-based
    std::vector<int> cluster_ids;        // sorted unique cluster ids
    std::vector<int> counts;
    std::vector<double> percentages;
    std::vector<double> weight_sums;
    std::vector<double> weight_percentages;
};

// Combined cutree + distribution in one call.
ClusterResultData compute_cluster_results(
    const double* linkage_ptr,
    int n,
    int num_clusters,
    const double* weights
);
