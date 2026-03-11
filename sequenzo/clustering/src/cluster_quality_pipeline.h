#pragma once

#include <string>
#include <vector>

static constexpr int CQ_NUM_METRICS = 10;

struct ClusterQualityPipelineResult {
    // Linkage data (only populated for the from-matrix path)
    std::vector<double> linkage_matrix;   // (N-1)*4, flat row-major
    std::vector<double> condensed_matrix; // N*(N-1)/2
    std::vector<double> full_matrix;      // N*N
    int warning_flags = 0;
    bool euclidean_compatible = true;

    // CQI score arrays (each length k_count)
    std::vector<double> pbc, hg, hgsd, asw, asww;
    std::vector<double> ch, r2, chsq, r2sq, hc;

    // CQI summary (each length CQ_NUM_METRICS)
    std::vector<double> opt_clusters;
    std::vector<double> raw_values;
    std::vector<double> z_scores;

    // Range table: k_count rows x CQ_NUM_METRICS cols, row-major
    std::vector<double> range_table;

    int k_min = 2;
    int k_count = 0;
};

// Path A: Cluster instance already provides linkage + condensed + weights.
// Computes CQI scores + summary + range table in one call.
ClusterQualityPipelineResult cluster_quality_from_cluster_data(
    const double* condensed, int condensed_len,
    const double* linkage, int n,
    const double* weights,
    int k_min, int k_max
);

// Path B: Raw distance matrix. Performs full pipeline:
//   matrix prep -> linkage -> CQI computation -> summary -> range table.
ClusterQualityPipelineResult cluster_quality_from_matrix(
    const double* matrix, int N,
    const std::string& method,
    int max_clusters,
    const double* weights
);
