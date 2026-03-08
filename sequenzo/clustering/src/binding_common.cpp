#include "binding_common.h"
#include "cluster_quality.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

void validate_square_matrix(const py::buffer_info& matrix_buf, const char* msg) {
    if (matrix_buf.ndim != 2 || matrix_buf.shape[0] != matrix_buf.shape[1]) {
        throw std::runtime_error(msg);
    }
}

void validate_vector_length(py::ssize_t actual, py::ssize_t expected, const char* msg) {
    if (actual != expected) {
        throw std::runtime_error(msg);
    }
}

void validate_condensed_size(py::ssize_t actual, int n, const char* msg) {
    const int expected = n * (n - 1) / 2;
    if (actual != expected) {
        throw std::runtime_error(msg);
    }
}

py::dict build_asw_result(py::array_t<double> asw_i, py::array_t<double> asw_w) {
    py::dict result;
    result["asw_individual"] = asw_i;
    result["asw_weighted"] = asw_w;
    return result;
}

py::dict stats_vector_to_dict(const std::vector<double>& stats) {
    py::dict result;
    result["PBC"] = stats[ClusterQualHPG];
    result["HG"] = stats[ClusterQualHG];
    result["HGSD"] = stats[ClusterQualHGSD];
    result["ASW"] = stats[ClusterQualASWi];
    result["ASWw"] = stats[ClusterQualASWw];
    result["CH"] = stats[ClusterQualF];
    result["R2"] = stats[ClusterQualR];
    result["CHsq"] = stats[ClusterQualF2];
    result["R2sq"] = stats[ClusterQualR2];
    result["HC"] = stats[ClusterQualHC];
    return result;
}

bool is_nan_ieee(double x) {
    uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(double));
    const uint64_t exp_mask = 0x7ff0000000000000ULL;
    const uint64_t mantissa_mask = 0x000fffffffffffffULL;
    return ((bits & exp_mask) == exp_mask) && ((bits & mantissa_mask) != 0ULL);
}
