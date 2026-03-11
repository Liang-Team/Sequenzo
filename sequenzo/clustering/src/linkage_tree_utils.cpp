#include "linkage_tree_utils.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

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

}  // namespace

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

ClusterResultData compute_cluster_results(
    const double* linkage_ptr,
    int n,
    int num_clusters,
    const double* weights
) {
    ClusterResultData result;
    result.labels.resize(static_cast<size_t>(n));

    compute_labels_from_linkage(linkage_ptr, n, num_clusters, result.labels.data());

    // Accumulate counts and weight sums per cluster.
    std::unordered_map<int, int> count_map;
    std::unordered_map<int, double> weight_map;
    count_map.reserve(static_cast<size_t>(num_clusters));
    weight_map.reserve(static_cast<size_t>(num_clusters));

    double total_weight = 0.0;
    for (int i = 0; i < n; ++i) {
        const int cid = result.labels[i];
        count_map[cid] += 1;
        weight_map[cid] += weights[i];
        total_weight += weights[i];
    }

    // Sort cluster ids.
    result.cluster_ids.reserve(count_map.size());
    for (const auto& kv : count_map) {
        result.cluster_ids.push_back(kv.first);
    }
    std::sort(result.cluster_ids.begin(), result.cluster_ids.end());

    const auto m = static_cast<int>(result.cluster_ids.size());
    result.counts.resize(static_cast<size_t>(m));
    result.percentages.resize(static_cast<size_t>(m));
    result.weight_sums.resize(static_cast<size_t>(m));
    result.weight_percentages.resize(static_cast<size_t>(m));

    for (int i = 0; i < m; ++i) {
        const int cid = result.cluster_ids[i];
        const int cnt = count_map[cid];
        const double wsum = weight_map[cid];
        result.counts[i] = cnt;
        result.percentages[i] = (n > 0)
            ? (100.0 * static_cast<double>(cnt) / static_cast<double>(n))
            : 0.0;
        result.weight_sums[i] = wsum;
        result.weight_percentages[i] = (total_weight > 0.0)
            ? (100.0 * wsum / total_weight)
            : 0.0;
    }

    return result;
}
