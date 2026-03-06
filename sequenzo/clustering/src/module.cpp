#include "PAM.cpp"
#include "KMedoid.cpp"
#include "PAMonce.cpp"
#include "weightedinertia.cpp"
#include "cluster_quality.cpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// ============================================================================
// Shared constants and lightweight helpers
// ============================================================================

constexpr double kWardTol = 1e-10;
constexpr double kWardViolationThreshold = 0.1;
constexpr int kWardSampleCap = 50;
constexpr int kEigenCheckCap = 100;

class DisjointSet {
public:
    explicit DisjointSet(int n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int x) {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) {
            return;
        }
        if (rank_[ra] < rank_[rb]) {
            parent_[ra] = rb;
        } else if (rank_[ra] > rank_[rb]) {
            parent_[rb] = ra;
        } else {
            parent_[rb] = ra;
            rank_[ra] += 1;
        }
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

void validate_square_matrix(const py::buffer_info& matrix_buf, const char* msg) {
    if (matrix_buf.ndim != 2 || matrix_buf.shape[0] != matrix_buf.shape[1]) {
        throw std::runtime_error(msg);
    }
}

void validate_vector_length(ssize_t actual, ssize_t expected, const char* msg) {
    if (actual != expected) {
        throw std::runtime_error(msg);
    }
}

void validate_condensed_size(ssize_t actual, int n, const char* msg) {
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

void validate_linkage(py::buffer_info& linkage_buf, int n) {
    if (linkage_buf.ndim != 2 || linkage_buf.shape[1] != 4) {
        throw std::runtime_error("Linkage matrix must have shape (n-1, 4)");
    }
    if (linkage_buf.shape[0] != n - 1) {
        throw std::runtime_error("Linkage matrix row count must be n-1");
    }
}

void compute_labels_from_linkage(
    const double* linkage_ptr,
    int n,
    int nclusters,
    int* labels_out
) {
    if (nclusters < 1 || nclusters > n) {
        throw std::runtime_error("nclusters must be in [1, n]");
    }

    if (n == 1) {
        labels_out[0] = 1;
        return;
    }

    const int merges_to_apply = n - nclusters;
    DisjointSet dsu(n);
    std::vector<int> node_rep(2 * n - 1, -1);
    for (int i = 0; i < n; ++i) {
        node_rep[i] = i;
    }

    for (int step = 0; step < merges_to_apply; ++step) {
        const int a = static_cast<int>(linkage_ptr[step * 4 + 0]);
        const int b = static_cast<int>(linkage_ptr[step * 4 + 1]);

        if (a < 0 || a >= (n + step) || b < 0 || b >= (n + step)) {
            throw std::runtime_error("Invalid linkage node index encountered");
        }
        if (node_rep[a] < 0 || node_rep[b] < 0) {
            throw std::runtime_error("Invalid linkage topology encountered");
        }

        dsu.unite(node_rep[a], node_rep[b]);
        node_rep[n + step] = dsu.find(node_rep[a]);
    }

    std::unordered_map<int, int> root_to_label;
    root_to_label.reserve(static_cast<size_t>(nclusters));
    int next_label = 1;

    for (int i = 0; i < n; ++i) {
        const int root = dsu.find(i);
        auto it = root_to_label.find(root);
        if (it == root_to_label.end()) {
            root_to_label.emplace(root, next_label);
            labels_out[i] = next_label;
            next_label += 1;
        } else {
            labels_out[i] = it->second;
        }
    }
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

double percentile_linear(std::vector<double>& values, double q) {
    if (values.empty()) {
        throw std::runtime_error("Cannot compute percentile on empty values");
    }
    if (q <= 0.0) {
        return *std::min_element(values.begin(), values.end());
    }
    if (q >= 1.0) {
        return *std::max_element(values.begin(), values.end());
    }

    std::sort(values.begin(), values.end());
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }
    const double alpha = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

bool is_finite_ieee(double x) {
    uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(double));
    // Exponent all-ones => Inf or NaN.
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

bool is_nan_ieee(double x) {
    uint64_t bits = 0;
    std::memcpy(&bits, &x, sizeof(double));
    const uint64_t exp_mask = 0x7ff0000000000000ULL;
    const uint64_t mantissa_mask = 0x000fffffffffffffULL;
    return ((bits & exp_mask) == exp_mask) && ((bits & mantissa_mask) != 0ULL);
}

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
) {
    PreparedMatrixData out;
    out.n = n;
    const ssize_t nn = n * n;
    out.full.assign(in_ptr, in_ptr + nn);

    std::vector<double> finite_vals;
    finite_vals.reserve(static_cast<size_t>(nn));

    for (ssize_t i = 0; i < nn; ++i) {
        const double v = out.full[static_cast<size_t>(i)];
        if (is_finite_ieee(v)) {
            finite_vals.push_back(v);
            if (v < 0.0) {
                out.had_negative = true;
            }
        } else {
            out.had_nonfinite = true;
        }
    }

    if (out.had_nonfinite) {
        out.warning_flags |= WARN_NONFINITE;
        if (!finite_vals.empty()) {
            out.replacement_value = percentile_linear(finite_vals, replacement_quantile);
        } else {
            out.replacement_value = 1.0;
        }
        for (ssize_t i = 0; i < nn; ++i) {
            if (!is_finite_ieee(out.full[static_cast<size_t>(i)])) {
                out.full[static_cast<size_t>(i)] = out.replacement_value;
            }
        }
    }

    for (ssize_t i = 0; i < n; ++i) {
        out.full[static_cast<size_t>(i * n + i)] = 0.0;
    }
    if (out.had_negative) {
        out.warning_flags |= WARN_NEGATIVE;
        for (ssize_t i = 0; i < nn; ++i) {
            if (out.full[static_cast<size_t>(i)] < 0.0) {
                out.full[static_cast<size_t>(i)] = 0.0;
            }
        }
    }

    if (enforce_symmetry) {
        bool is_symmetric = true;
        for (ssize_t i = 0; i < n && is_symmetric; ++i) {
            for (ssize_t j = i + 1; j < n; ++j) {
                const double a = out.full[static_cast<size_t>(i * n + j)];
                const double b = out.full[static_cast<size_t>(j * n + i)];
                const double tol = atol + rtol * std::abs(b);
                if (std::abs(a - b) > tol) {
                    is_symmetric = false;
                    break;
                }
            }
        }
        if (!is_symmetric) {
            out.was_symmetrized = true;
            out.warning_flags |= WARN_SYMMETRIZED;
            for (ssize_t i = 0; i < n; ++i) {
                for (ssize_t j = i + 1; j < n; ++j) {
                    const double avg = 0.5 * (
                        out.full[static_cast<size_t>(i * n + j)] +
                        out.full[static_cast<size_t>(j * n + i)]
                    );
                    out.full[static_cast<size_t>(i * n + j)] = avg;
                    out.full[static_cast<size_t>(j * n + i)] = avg;
                }
            }
        }
    }

    const ssize_t condensed_size = n * (n - 1) / 2;
    out.condensed.resize(static_cast<size_t>(condensed_size));
    ssize_t idx = 0;
    for (ssize_t i = 0; i < n; ++i) {
        for (ssize_t j = i + 1; j < n; ++j) {
            out.condensed[static_cast<size_t>(idx++)] = out.full[static_cast<size_t>(i * n + j)];
        }
    }
    return out;
}

EuclideanCheckResult check_euclidean_compatibility_impl(
    const double* matrix_ptr,
    ssize_t n,
    const std::string& method
) {
    EuclideanCheckResult out;
    const std::string m = method;
    if (m != "ward" && m != "ward_d" && m != "ward_d2") {
        out.compatible = true;
        return out;
    }

    const int sample_size = static_cast<int>(std::min<ssize_t>(kWardSampleCap, n));
    std::vector<int> indices(static_cast<size_t>(n));
    std::iota(indices.begin(), indices.end(), 0);
    if (n > sample_size) {
        std::mt19937 gen(5489U + static_cast<uint32_t>(n));
        std::shuffle(indices.begin(), indices.end(), gen);
        indices.resize(static_cast<size_t>(sample_size));
    }
    out.sample_n = static_cast<int>(indices.size());
    const int s = out.sample_n;

    std::vector<double> sample(static_cast<size_t>(s * s), 0.0);
    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s; ++j) {
            sample[static_cast<size_t>(i * s + j)] =
                matrix_ptr[static_cast<size_t>(indices[static_cast<size_t>(i)] * n + indices[static_cast<size_t>(j)])];
        }
    }

    long long violations = 0;
    long long total_checks = 0;
    for (int i = 0; i < s; ++i) {
        for (int j = i + 1; j < s; ++j) {
            for (int k = j + 1; k < s; ++k) {
                const double dij = sample[static_cast<size_t>(i * s + j)];
                const double dik = sample[static_cast<size_t>(i * s + k)];
                const double djk = sample[static_cast<size_t>(j * s + k)];
                if (dik > dij + djk + kWardTol || dij > dik + djk + kWardTol || djk > dij + dik + kWardTol) {
                    ++violations;
                }
                ++total_checks;
            }
        }
    }
    if (total_checks > 0) {
        out.violation_rate = static_cast<double>(violations) / static_cast<double>(total_checks);
        if (out.violation_rate > kWardViolationThreshold) {
            out.compatible = false;
            return out;
        }
    }

    // Eigen-based check for small matrices (matching Python heuristic).
    if (s <= kEigenCheckCap) {
        try {
            auto sample_arr = py::array_t<double>({s, s});
            auto sample_buf = sample_arr.request();
            auto* sample_ptr = static_cast<double*>(sample_buf.ptr);
            std::copy(sample.begin(), sample.end(), sample_ptr);

            py::module_ np = py::module_::import("numpy");
            py::module_ linalg = py::module_::import("numpy.linalg");
            py::object H = np.attr("eye")(s) - (np.attr("ones")(py::make_tuple(s, s)) / py::float_(static_cast<double>(s)));
            py::object sq = np.attr("square")(sample_arr);
            py::object B = py::float_(-0.5) * H.attr("__matmul__")(sq.attr("__matmul__")(H));
            py::array eigenvals = linalg.attr("eigvalsh")(B).cast<py::array>();
            auto eig_buf = eigenvals.request();
            auto* eig_ptr = static_cast<double*>(eig_buf.ptr);

            double neg_energy = 0.0;
            double total_energy = 0.0;
            for (ssize_t i = 0; i < eig_buf.size; ++i) {
                const double ev = eig_ptr[i];
                if (ev < -kWardTol) {
                    neg_energy += -ev;
                }
                total_energy += std::abs(ev);
            }
            if (total_energy > 0.0) {
                out.neg_energy_ratio = neg_energy / total_energy;
                if (out.neg_energy_ratio > kWardViolationThreshold) {
                    out.compatible = false;
                    return out;
                }
            }
        } catch (const py::error_already_set&) {
            // Keep compatibility as-is if eig computation fails, same as Python behavior.
        }
    }

    out.compatible = true;
    return out;
}

}  // namespace

PYBIND11_MODULE(clustering_c_code, m) {
    // =========================================================================
    // Legacy clustering engines
    // =========================================================================
    auto register_legacy_engines = [&m]() {
    py::class_<PAM>(m, "PAM")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &PAM::runclusterloop);

    py::class_<KMedoid>(m, "KMedoid")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &KMedoid::runclusterloop);

    py::class_<PAMonce>(m, "PAMonce")
            .def(py::init<int, py::array_t<double>, py::array_t<int>, int, py::array_t<double>>())
            .def("runclusterloop", &PAMonce::runclusterloop);

    py::class_<weightedinertia>(m, "weightedinertia")
            .def(py::init<py::array_t<double>, py::array_t<int>, py::array_t<double>>())
            .def("tmrWeightedInertiaContrib", &weightedinertia::tmrWeightedInertiaContrib);
    };
    register_legacy_engines();

    // =========================================================================
    // Cluster quality metrics
    // =========================================================================
    auto register_cluster_quality_apis = [&m]() {
    m.def("cluster_quality", [](py::array_t<double> diss_matrix, 
                               py::array_t<int> cluster_labels,
                               py::array_t<double> weights,
                               int nclusters) -> py::dict {
        auto diss_buf = diss_matrix.request();
        auto cluster_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        validate_square_matrix(diss_buf, "Distance matrix must be square");
        
        int n = diss_buf.shape[0];
        validate_vector_length(cluster_buf.size, n, "Cluster labels and weights must have same length as matrix dimension");
        validate_vector_length(weights_buf.size, n, "Cluster labels and weights must have same length as matrix dimension");
        
        double* diss_ptr = static_cast<double*>(diss_buf.ptr);
        int* cluster_ptr = static_cast<int*>(cluster_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        // Prepare output arrays
        std::vector<double> stats(ClusterQualNumStat);
        std::vector<double> asw(2 * nclusters);
        
        // Create Kendall tree for caching
        KendallTree kendall;
        
        // Call core function
        clusterquality(diss_ptr, cluster_ptr, weights_ptr, n, 
                      stats.data(), nclusters, asw.data(), kendall);
        
        // Clean up Kendall tree
        finalizeKendall(kendall);
        
        py::dict result = stats_vector_to_dict(stats);
        
        // Convert ASW array to numpy array
        auto asw_array = py::array_t<double>(2 * nclusters);
        auto asw_buf = asw_array.request();
        double* asw_out = static_cast<double*>(asw_buf.ptr);
        std::copy(asw.begin(), asw.end(), asw_out);
        result["cluster_asw"] = asw_array;
        
        return result;
    }, "Compute cluster quality indicators for full distance matrix");

    m.def("cluster_quality_condensed", [](py::array_t<double> diss_condensed,
                                         py::array_t<int> cluster_labels,
                                         py::array_t<double> weights,
                                         int n, int nclusters) -> py::dict {
        auto diss_buf = diss_condensed.request();
        auto cluster_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        validate_condensed_size(diss_buf.size, n, "Condensed distance array size mismatch");
        validate_vector_length(cluster_buf.size, n, "Cluster labels and weights must have length n");
        validate_vector_length(weights_buf.size, n, "Cluster labels and weights must have length n");
        
        double* diss_ptr = static_cast<double*>(diss_buf.ptr);
        int* cluster_ptr = static_cast<int*>(cluster_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        // Prepare output arrays
        std::vector<double> stats(ClusterQualNumStat);
        std::vector<double> asw(2 * nclusters);
        
        // Create Kendall tree for caching
        KendallTree kendall;
        
        // Call core function
        clusterquality_dist(diss_ptr, cluster_ptr, weights_ptr, n,
                           stats.data(), nclusters, asw.data(), kendall);
        
        // Clean up Kendall tree
        finalizeKendall(kendall);
        
        // Return results as dictionary
        py::dict result = stats_vector_to_dict(stats);
        
        // Convert ASW array to numpy array
        auto asw_array = py::array_t<double>(2 * nclusters);
        auto asw_buf = asw_array.request();
        double* asw_out = static_cast<double*>(asw_buf.ptr);
        std::copy(asw.begin(), asw.end(), asw_out);
        result["cluster_asw"] = asw_array;
        
        return result;
    }, "Compute cluster quality indicators for condensed distance array");

    m.def("individual_asw", [](py::array_t<double> diss_matrix,
                              py::array_t<int> cluster_labels,
                              py::array_t<double> weights,
                              int nclusters) -> py::dict {
        auto diss_buf = diss_matrix.request();
        auto cluster_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        validate_square_matrix(diss_buf, "Distance matrix must be square");
        
        int n = diss_buf.shape[0];
        validate_vector_length(cluster_buf.size, n, "Cluster labels and weights must have same length as matrix dimension");
        validate_vector_length(weights_buf.size, n, "Cluster labels and weights must have same length as matrix dimension");
        
        double* diss_ptr = static_cast<double*>(diss_buf.ptr);
        int* cluster_ptr = static_cast<int*>(cluster_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        // Prepare output arrays
        auto asw_i = py::array_t<double>(n);
        auto asw_w = py::array_t<double>(n);
        
        auto asw_i_buf = asw_i.request();
        auto asw_w_buf = asw_w.request();
        
        double* asw_i_ptr = static_cast<double*>(asw_i_buf.ptr);
        double* asw_w_ptr = static_cast<double*>(asw_w_buf.ptr);
        
        // Call core function
        indiv_asw(diss_ptr, cluster_ptr, weights_ptr, n, nclusters, asw_i_ptr, asw_w_ptr);
        
        return build_asw_result(asw_i, asw_w);
    }, "Compute individual ASW scores for all samples");

    m.def("individual_asw_condensed", [](py::array_t<double> diss_condensed,
                                        py::array_t<int> cluster_labels,
                                        py::array_t<double> weights,
                                        int n, int nclusters) -> py::dict {
        auto diss_buf = diss_condensed.request();
        auto cluster_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        validate_condensed_size(diss_buf.size, n, "Condensed distance array size mismatch");
        validate_vector_length(cluster_buf.size, n, "Cluster labels and weights must have length n");
        validate_vector_length(weights_buf.size, n, "Cluster labels and weights must have length n");
        
        double* diss_ptr = static_cast<double*>(diss_buf.ptr);
        int* cluster_ptr = static_cast<int*>(cluster_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        // Prepare output arrays
        auto asw_i = py::array_t<double>(n);
        auto asw_w = py::array_t<double>(n);
        
        auto asw_i_buf = asw_i.request();
        auto asw_w_buf = asw_w.request();
        
        double* asw_i_ptr = static_cast<double*>(asw_i_buf.ptr);
        double* asw_w_ptr = static_cast<double*>(asw_w_buf.ptr);
        
        // Call core function
        indiv_asw_dist(diss_ptr, cluster_ptr, weights_ptr, n, nclusters, asw_i_ptr, asw_w_ptr);
        
        return build_asw_result(asw_i, asw_w);
    }, "Compute individual ASW scores for condensed distance array");
    };
    register_cluster_quality_apis();

    // =========================================================================
    // Linkage tree operations
    // =========================================================================
    auto register_linkage_and_distribution_apis = [&m]() {
    m.def("cutree_maxclust", [](py::array_t<double> linkage_matrix,
                                int n,
                                int nclusters) -> py::array_t<int> {
        auto linkage_buf = linkage_matrix.request();
        validate_linkage(linkage_buf, n);

        auto labels = py::array_t<int>(n);
        auto labels_buf = labels.request();
        auto* linkage_ptr = static_cast<double*>(linkage_buf.ptr);
        auto* labels_ptr = static_cast<int*>(labels_buf.ptr);

        compute_labels_from_linkage(linkage_ptr, n, nclusters, labels_ptr);
        return labels;
    }, "Cut linkage tree into exactly nclusters clusters (1-based labels).");

    m.def("cutree_maxclust_all", [](py::array_t<double> linkage_matrix,
                                    int n,
                                    int k_min,
                                    int k_max) -> py::array_t<int> {
        auto linkage_buf = linkage_matrix.request();
        validate_linkage(linkage_buf, n);

        if (k_min < 1 || k_max < k_min || k_max > n) {
            throw std::runtime_error("Require 1 <= k_min <= k_max <= n");
        }

        const int k_count = k_max - k_min + 1;
        auto labels = py::array_t<int>({k_count, n});
        auto labels_buf = labels.request();
        auto* linkage_ptr = static_cast<double*>(linkage_buf.ptr);
        auto* labels_ptr = static_cast<int*>(labels_buf.ptr);

        for (int idx = 0; idx < k_count; ++idx) {
            const int k = k_min + idx;
            compute_labels_from_linkage(linkage_ptr, n, k, labels_ptr + idx * n);
        }
        return labels;
    }, "Cut linkage tree for all k in [k_min, k_max].");

    m.def("cluster_quality_over_k_condensed", [](py::array_t<double> diss_condensed,
                                                 py::array_t<double> linkage_matrix,
                                                 py::array_t<double> weights,
                                                 int n,
                                                 int k_min,
                                                 int k_max) -> py::dict {
        auto diss_buf = diss_condensed.request();
        auto linkage_buf = linkage_matrix.request();
        auto weights_buf = weights.request();

        validate_linkage(linkage_buf, n);
        validate_condensed_size(diss_buf.size, n, "Condensed distance array size mismatch");
        validate_vector_length(weights_buf.size, n, "Weights must have length n");
        if (k_min < 2 || k_max < k_min || k_max > n) {
            throw std::runtime_error("Require 2 <= k_min <= k_max <= n");
        }

        const int k_count = k_max - k_min + 1;
        auto* diss_ptr = static_cast<double*>(diss_buf.ptr);
        auto* linkage_ptr = static_cast<double*>(linkage_buf.ptr);
        auto* weights_ptr = static_cast<double*>(weights_buf.ptr);

        std::vector<int> labels(static_cast<size_t>(n), 1);
        std::vector<double> stats(ClusterQualNumStat);

        auto pbc = py::array_t<double>(k_count);
        auto hg = py::array_t<double>(k_count);
        auto hgsd = py::array_t<double>(k_count);
        auto asw = py::array_t<double>(k_count);
        auto asww = py::array_t<double>(k_count);
        auto ch = py::array_t<double>(k_count);
        auto r2 = py::array_t<double>(k_count);
        auto chsq = py::array_t<double>(k_count);
        auto r2sq = py::array_t<double>(k_count);
        auto hc = py::array_t<double>(k_count);

        auto pbc_buf = pbc.request();
        auto hg_buf = hg.request();
        auto hgsd_buf = hgsd.request();
        auto asw_buf = asw.request();
        auto asww_buf = asww.request();
        auto ch_buf = ch.request();
        auto r2_buf = r2.request();
        auto chsq_buf = chsq.request();
        auto r2sq_buf = r2sq.request();
        auto hc_buf = hc.request();

        auto* pbc_ptr = static_cast<double*>(pbc_buf.ptr);
        auto* hg_ptr = static_cast<double*>(hg_buf.ptr);
        auto* hgsd_ptr = static_cast<double*>(hgsd_buf.ptr);
        auto* asw_ptr = static_cast<double*>(asw_buf.ptr);
        auto* asww_ptr = static_cast<double*>(asww_buf.ptr);
        auto* ch_ptr = static_cast<double*>(ch_buf.ptr);
        auto* r2_ptr = static_cast<double*>(r2_buf.ptr);
        auto* chsq_ptr = static_cast<double*>(chsq_buf.ptr);
        auto* r2sq_ptr = static_cast<double*>(r2sq_buf.ptr);
        auto* hc_ptr = static_cast<double*>(hc_buf.ptr);

        for (int idx = 0; idx < k_count; ++idx) {
            const int k = k_min + idx;
            compute_labels_from_linkage(linkage_ptr, n, k, labels.data());

            std::vector<double> asw_cluster(2 * k);
            // clusterquality_dist expects a fresh Kendall tree state per run.
            KendallTree kendall;
            clusterquality_dist(
                diss_ptr,
                labels.data(),
                weights_ptr,
                n,
                stats.data(),
                k,
                asw_cluster.data(),
                kendall
            );
            finalizeKendall(kendall);

            pbc_ptr[idx] = stats[ClusterQualHPG];
            hg_ptr[idx] = stats[ClusterQualHG];
            hgsd_ptr[idx] = stats[ClusterQualHGSD];
            asw_ptr[idx] = stats[ClusterQualASWi];
            asww_ptr[idx] = stats[ClusterQualASWw];
            ch_ptr[idx] = stats[ClusterQualF];
            r2_ptr[idx] = stats[ClusterQualR];
            chsq_ptr[idx] = stats[ClusterQualF2];
            r2sq_ptr[idx] = stats[ClusterQualR2];
            hc_ptr[idx] = stats[ClusterQualHC];
        }

        py::dict result;
        result["PBC"] = pbc;
        result["HG"] = hg;
        result["HGSD"] = hgsd;
        result["ASW"] = asw;
        result["ASWw"] = asww;
        result["CH"] = ch;
        result["R2"] = r2;
        result["CHsq"] = chsq;
        result["R2sq"] = r2sq;
        result["HC"] = hc;
        return result;
    }, "Compute cluster quality metrics for all k in [k_min, k_max] from condensed distances and linkage.");

    m.def("cluster_distribution_from_labels", [](py::array_t<int> cluster_labels,
                                                 py::array_t<double> weights) -> py::dict {
        auto labels_buf = cluster_labels.request();
        auto weights_buf = weights.request();
        if (labels_buf.ndim != 1 || weights_buf.ndim != 1) {
            throw std::runtime_error("cluster_labels and weights must be 1D arrays");
        }
        if (labels_buf.size != weights_buf.size) {
            throw std::runtime_error("cluster_labels and weights must have the same length");
        }

        const int n = static_cast<int>(labels_buf.size);
        auto* labels_ptr = static_cast<int*>(labels_buf.ptr);
        auto* weights_ptr = static_cast<double*>(weights_buf.ptr);

        std::unordered_map<int, int> count_map;
        std::unordered_map<int, double> weight_map;
        count_map.reserve(static_cast<size_t>(n));
        weight_map.reserve(static_cast<size_t>(n));

        double total_weight = 0.0;
        for (int i = 0; i < n; ++i) {
            const int cid = labels_ptr[i];
            if (cid <= 0) {
                throw std::runtime_error("cluster labels must be positive integers");
            }
            count_map[cid] += 1;
            weight_map[cid] += weights_ptr[i];
            total_weight += weights_ptr[i];
        }

        std::vector<int> cluster_ids;
        cluster_ids.reserve(count_map.size());
        for (const auto& kv : count_map) {
            cluster_ids.push_back(kv.first);
        }
        std::sort(cluster_ids.begin(), cluster_ids.end());

        const ssize_t m = static_cast<ssize_t>(cluster_ids.size());
        auto out_cluster = py::array_t<int>(m);
        auto out_count = py::array_t<int>(m);
        auto out_pct = py::array_t<double>(m);
        auto out_weight_sum = py::array_t<double>(m);
        auto out_weight_pct = py::array_t<double>(m);

        auto* cluster_ptr = static_cast<int*>(out_cluster.request().ptr);
        auto* count_ptr = static_cast<int*>(out_count.request().ptr);
        auto* pct_ptr = static_cast<double*>(out_pct.request().ptr);
        auto* weight_sum_ptr = static_cast<double*>(out_weight_sum.request().ptr);
        auto* weight_pct_ptr = static_cast<double*>(out_weight_pct.request().ptr);

        for (ssize_t i = 0; i < m; ++i) {
            const int cid = cluster_ids[i];
            const int cnt = count_map[cid];
            const double wsum = weight_map[cid];
            cluster_ptr[i] = cid;
            count_ptr[i] = cnt;
            pct_ptr[i] = (n > 0) ? (100.0 * static_cast<double>(cnt) / static_cast<double>(n)) : 0.0;
            weight_sum_ptr[i] = wsum;
            weight_pct_ptr[i] = (total_weight > 0.0) ? (100.0 * wsum / total_weight) : 0.0;
        }

        py::dict result;
        result["Cluster"] = out_cluster;
        result["Count"] = out_count;
        result["Percentage"] = out_pct;
        result["Weight_Sum"] = out_weight_sum;
        result["Weight_Percentage"] = out_weight_pct;
        return result;
    }, "Compute cluster distribution (count and weight stats) from labels.");
    };
    register_linkage_and_distribution_apis();

    // =========================================================================
    // Distance matrix preparation and compatibility checks
    // =========================================================================
    auto register_distance_prep_apis = [&m]() {
    m.def("prepare_distance_matrix", [](py::array_t<double, py::array::c_style | py::array::forcecast> matrix,
                                        bool enforce_symmetry,
                                        double rtol,
                                        double atol,
                                        double replacement_quantile) -> py::dict {
        auto in_buf = matrix.request();
        if (in_buf.ndim != 2 || in_buf.shape[0] != in_buf.shape[1]) {
            throw std::runtime_error("Distance matrix must be square");
        }
        const ssize_t n = in_buf.shape[0];
        auto* in_ptr = static_cast<double*>(in_buf.ptr);
        PreparedMatrixData prep = prepare_distance_matrix_impl(
            in_ptr, n, enforce_symmetry, rtol, atol, replacement_quantile
        );

        auto full = py::array_t<double>({n, n});
        auto* full_ptr = static_cast<double*>(full.request().ptr);
        std::copy(prep.full.begin(), prep.full.end(), full_ptr);

        auto condensed = py::array_t<double>(n * (n - 1) / 2);
        auto* cond_ptr = static_cast<double*>(condensed.request().ptr);
        std::copy(prep.condensed.begin(), prep.condensed.end(), cond_ptr);

        py::dict result;
        result["full_matrix"] = full;
        result["condensed_matrix"] = condensed;
        result["had_nonfinite"] = prep.had_nonfinite;
        result["had_negative"] = prep.had_negative;
        result["was_symmetrized"] = prep.was_symmetrized;
        result["replacement_value"] = prep.replacement_value;
        result["warning_flags"] = prep.warning_flags;
        return result;
    },
    py::arg("matrix"),
    py::arg("enforce_symmetry") = true,
    py::arg("rtol") = 1e-5,
    py::arg("atol") = 1e-8,
    py::arg("replacement_quantile") = 0.95,
    "Clean/validate/symmetrize square distance matrix and return condensed form.");

    m.def("check_euclidean_compatibility", [](py::array_t<double, py::array::c_style | py::array::forcecast> matrix,
                                              const std::string& method) -> py::dict {
        auto buf = matrix.request();
        if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
            throw std::runtime_error("Distance matrix must be square");
        }
        const ssize_t n = buf.shape[0];
        auto* ptr = static_cast<double*>(buf.ptr);
        EuclideanCheckResult r = check_euclidean_compatibility_impl(ptr, n, method);

        py::dict out;
        out["compatible"] = r.compatible;
        out["violation_rate"] = r.violation_rate;
        out["neg_energy_ratio"] = r.neg_energy_ratio;
        out["sample_n"] = r.sample_n;
        return out;
    }, "Heuristic Euclidean compatibility check for Ward methods.");

    m.def("prepare_distance_matrix_and_check_ward", [](py::array_t<double, py::array::c_style | py::array::forcecast> matrix,
                                                       const std::string& method,
                                                       bool enforce_symmetry,
                                                       double rtol,
                                                       double atol,
                                                       double replacement_quantile) -> py::dict {
        auto in_buf = matrix.request();
        if (in_buf.ndim != 2 || in_buf.shape[0] != in_buf.shape[1]) {
            throw std::runtime_error("Distance matrix must be square");
        }
        const ssize_t n = in_buf.shape[0];
        auto* in_ptr = static_cast<double*>(in_buf.ptr);

        PreparedMatrixData prep = prepare_distance_matrix_impl(
            in_ptr, n, enforce_symmetry, rtol, atol, replacement_quantile
        );
        EuclideanCheckResult eu = check_euclidean_compatibility_impl(
            prep.full.data(), n, method
        );

        auto full = py::array_t<double>({n, n});
        auto* full_ptr = static_cast<double*>(full.request().ptr);
        std::copy(prep.full.begin(), prep.full.end(), full_ptr);

        auto condensed = py::array_t<double>(n * (n - 1) / 2);
        auto* cond_ptr = static_cast<double*>(condensed.request().ptr);
        std::copy(prep.condensed.begin(), prep.condensed.end(), cond_ptr);

        py::dict result;
        result["full_matrix"] = full;
        result["condensed_matrix"] = condensed;
        result["had_nonfinite"] = prep.had_nonfinite;
        result["had_negative"] = prep.had_negative;
        result["was_symmetrized"] = prep.was_symmetrized;
        result["replacement_value"] = prep.replacement_value;
        int warning_flags = prep.warning_flags;
        if (!eu.compatible && (method == "ward" || method == "ward_d" || method == "ward_d2")) {
            warning_flags |= WARN_WARD_NON_EUCLIDEAN;
        }
        result["warning_flags"] = warning_flags;
        result["compatible"] = eu.compatible;
        result["violation_rate"] = eu.violation_rate;
        result["neg_energy_ratio"] = eu.neg_energy_ratio;
        result["sample_n"] = eu.sample_n;
        return result;
    },
    py::arg("matrix"),
    py::arg("method"),
    py::arg("enforce_symmetry") = true,
    py::arg("rtol") = 1e-5,
    py::arg("atol") = 1e-8,
    py::arg("replacement_quantile") = 0.95,
    "Prepare distance matrix and run Ward Euclidean-compatibility check in one call.");
    };
    register_distance_prep_apis();

    // =========================================================================
    // CQI post-processing helpers
    // =========================================================================
    auto register_cqi_postprocess_apis = [&m]() {
    m.def("cluster_quality_summary", [](py::dict metric_values,
                                        py::list metric_order,
                                        int k_start) -> py::dict {
        const ssize_t m = static_cast<ssize_t>(py::len(metric_order));
        auto opt_clusters = py::array_t<double>(m);
        auto raw_values = py::array_t<double>(m);
        auto z_values = py::array_t<double>(m);
        auto* opt_ptr = static_cast<double*>(opt_clusters.request().ptr);
        auto* raw_ptr = static_cast<double*>(raw_values.request().ptr);
        auto* z_ptr = static_cast<double*>(z_values.request().ptr);

        for (ssize_t i = 0; i < m; ++i) {
            const std::string metric = py::cast<std::string>(metric_order[i]);
            if (!metric_values.contains(py::str(metric))) {
                opt_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                raw_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                z_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>(
                metric_values[py::str(metric)]
            );
            auto buf = arr.request();
            auto* ptr = static_cast<double*>(buf.ptr);
            const ssize_t n = buf.size;

            double sum = 0.0;
            double sumsq = 0.0;
            ssize_t cnt = 0;
            double best_val = -std::numeric_limits<double>::infinity();
            ssize_t best_idx = -1;

            for (ssize_t j = 0; j < n; ++j) {
                const double v = ptr[j];
                if (is_nan_ieee(v)) {
                    continue;
                }
                sum += v;
                sumsq += v * v;
                cnt += 1;
                if (best_idx < 0 || v > best_val) {
                    best_val = v;
                    best_idx = j;
                }
            }

            if (best_idx < 0 || cnt == 0) {
                opt_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                raw_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                z_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            const double mean = sum / static_cast<double>(cnt);
            const double var = std::max(0.0, (sumsq / static_cast<double>(cnt)) - mean * mean);
            const double stddev = std::sqrt(var);
            const double raw = ptr[best_idx];
            const double z = (stddev > 0.0) ? ((raw - mean) / stddev) : raw;

            opt_ptr[i] = static_cast<double>(k_start + best_idx);
            raw_ptr[i] = raw;
            z_ptr[i] = z;
        }

        py::dict out;
        out["Opt. Clusters"] = opt_clusters;
        out["Raw Value"] = raw_values;
        out["Z-Score Norm."] = z_values;
        return out;
    }, py::arg("metric_values"), py::arg("metric_order"), py::arg("k_start") = 2,
    "Compute CQI summary (optimal k, raw value, z-score at optimum) from metric arrays.");

    m.def("cluster_quality_range_table", [](py::dict metric_values,
                                            py::list metric_order) -> py::dict {
        const ssize_t n_metrics = static_cast<ssize_t>(py::len(metric_order));
        ssize_t n_rows = -1;
        std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> arrays;
        arrays.reserve(static_cast<size_t>(n_metrics));

        for (ssize_t i = 0; i < n_metrics; ++i) {
            const std::string metric = py::cast<std::string>(metric_order[i]);
            py::array_t<double, py::array::c_style | py::array::forcecast> arr;
            if (metric_values.contains(py::str(metric))) {
                arr = py::array_t<double, py::array::c_style | py::array::forcecast>(
                    metric_values[py::str(metric)]
                );
            } else {
                arr = py::array_t<double>(0);
            }
            auto buf = arr.request();
            if (n_rows < 0) {
                n_rows = buf.size;
            } else if (buf.size != n_rows) {
                throw std::runtime_error("Inconsistent metric lengths detected");
            }
            arrays.push_back(arr);
        }

        if (n_rows < 0) {
            n_rows = 0;
        }

        auto values = py::array_t<double>({n_rows, n_metrics});
        auto* out_ptr = static_cast<double*>(values.request().ptr);
        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (ssize_t r = 0; r < n_rows; ++r) {
            for (ssize_t c = 0; c < n_metrics; ++c) {
                out_ptr[r * n_metrics + c] = nan;
            }
        }

        for (ssize_t c = 0; c < n_metrics; ++c) {
            auto buf = arrays[static_cast<size_t>(c)].request();
            auto* ptr = static_cast<double*>(buf.ptr);
            for (ssize_t r = 0; r < n_rows; ++r) {
                out_ptr[r * n_metrics + c] = ptr[r];
            }
        }

        py::dict out;
        out["values"] = values;
        out["n_rows"] = n_rows;
        return out;
    }, py::arg("metric_values"), py::arg("metric_order"),
    "Build metrics-by-cluster range table values matrix.");
    };
    register_cqi_postprocess_apis();
}