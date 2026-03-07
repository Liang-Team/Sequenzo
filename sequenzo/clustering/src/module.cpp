#include "PAM.cpp"
#include "KMedoid.cpp"
#include "PAMonce.cpp"
#include "weightedinertia.cpp"
#include "cluster_quality.cpp"
#include "binding_common.cpp"
#include "linkage_tree_utils.cpp"
#include "distance_prep_utils.cpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

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
                                        double replacement_quantile,
                                        bool include_full_matrix) -> py::dict {
        auto in_buf = matrix.request();
        if (in_buf.ndim != 2 || in_buf.shape[0] != in_buf.shape[1]) {
            throw std::runtime_error("Distance matrix must be square");
        }
        const ssize_t n = in_buf.shape[0];
        auto* in_ptr = static_cast<double*>(in_buf.ptr);
        PreparedMatrixData prep = prepare_distance_matrix_impl(
            in_ptr, n, enforce_symmetry, rtol, atol, replacement_quantile
        );

        auto condensed = vector_to_pyarray_1d(std::move(prep.condensed));

        py::dict result;
        if (include_full_matrix) {
            result["full_matrix"] = vector_to_pyarray_2d(std::move(prep.full), n, n);
        }
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
    py::arg("include_full_matrix") = true,
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
                                                       double replacement_quantile,
                                                       bool include_full_matrix,
                                                       bool run_ward_check) -> py::dict {
        auto in_buf = matrix.request();
        if (in_buf.ndim != 2 || in_buf.shape[0] != in_buf.shape[1]) {
            throw std::runtime_error("Distance matrix must be square");
        }
        const ssize_t n = in_buf.shape[0];
        auto* in_ptr = static_cast<double*>(in_buf.ptr);

        PreparedMatrixData prep = prepare_distance_matrix_impl(
            in_ptr, n, enforce_symmetry, rtol, atol, replacement_quantile
        );
        EuclideanCheckResult eu;
        if (run_ward_check) {
            eu = check_euclidean_compatibility_impl(prep.full.data(), n, method);
        }

        auto condensed = vector_to_pyarray_1d(std::move(prep.condensed));

        py::dict result;
        if (include_full_matrix) {
            result["full_matrix"] = vector_to_pyarray_2d(std::move(prep.full), n, n);
        }
        result["condensed_matrix"] = condensed;
        result["had_nonfinite"] = prep.had_nonfinite;
        result["had_negative"] = prep.had_negative;
        result["was_symmetrized"] = prep.was_symmetrized;
        result["replacement_value"] = prep.replacement_value;
        int warning_flags = prep.warning_flags;
        if (run_ward_check && !eu.compatible && (method == "ward" || method == "ward_d" || method == "ward_d2")) {
            warning_flags |= WARN_WARD_NON_EUCLIDEAN;
        }
        result["warning_flags"] = warning_flags;
        if (run_ward_check) {
            result["compatible"] = py::bool_(eu.compatible);
            result["violation_rate"] = py::float_(eu.violation_rate);
            result["neg_energy_ratio"] = py::float_(eu.neg_energy_ratio);
            result["sample_n"] = py::int_(eu.sample_n);
        } else {
            result["compatible"] = py::none();
            result["violation_rate"] = py::none();
            result["neg_energy_ratio"] = py::none();
            result["sample_n"] = py::none();
        }
        return result;
    },
    py::arg("matrix"),
    py::arg("method"),
    py::arg("enforce_symmetry") = true,
    py::arg("rtol") = 1e-5,
    py::arg("atol") = 1e-8,
    py::arg("replacement_quantile") = 0.95,
    py::arg("include_full_matrix") = true,
    py::arg("run_ward_check") = true,
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