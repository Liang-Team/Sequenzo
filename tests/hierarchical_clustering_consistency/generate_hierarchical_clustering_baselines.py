"""
Generate hierarchical clustering reference results for consistency checks.

Run before/after refactors to compare ClusterQuality and ClusterResults outputs.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sequenzo import SequenceData, get_distance_matrix, load_dataset
from sequenzo.clustering.hierarchical_clustering import Cluster, ClusterQuality, ClusterResults

REFERENCE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "reference_results")
os.makedirs(REFERENCE_RESULTS_DIR, exist_ok=True)


def save_reference_result(prefix, cq, cr, method, ks=(3, 5, 10)):
    """Save all relevant outputs for one dataset + method combination."""
    data = {}
    for metric, values in cq.scores.items():
        data[f"scores_{metric}"] = np.array(values, dtype=np.float64)

    cqi_table = cq.get_cqi_table()
    data["cqi_opt_clusters"] = cqi_table["Opt. Clusters"].values.astype(np.float64)
    data["cqi_raw_value"] = cqi_table["Raw Value"].values.astype(np.float64)
    data["cqi_zscore"] = cqi_table["Z-Score Norm."].values.astype(np.float64)

    range_table = cq.get_cluster_range_table()
    data["range_table"] = range_table.values.astype(np.float64)

    for k in ks:
        if k > cq.max_clusters:
            continue
        membership = cr.get_cluster_memberships(k)
        data[f"labels_k{k}"] = membership["Cluster"].values.astype(np.int32)

        dist = cr.get_cluster_distribution(k, weighted=True)
        data[f"dist_cluster_k{k}"] = dist["Cluster"].values.astype(np.int32)
        data[f"dist_count_k{k}"] = dist["Count"].values.astype(np.int32)
        data[f"dist_pct_k{k}"] = dist["Percentage"].values.astype(np.float64)
        data[f"dist_wsum_k{k}"] = dist["Weight_Sum"].values.astype(np.float64)
        data[f"dist_wpct_k{k}"] = dist["Weight_Percentage"].values.astype(np.float64)

    path = os.path.join(REFERENCE_RESULTS_DIR, f"{prefix}_{method}.npz")
    np.savez(path, **data)
    print(f"  Saved: {path} ({len(data)} arrays)")


def run_co2_dataset(method):
    print(f"\n--- country_co2, method={method} ---")
    df = load_dataset("country_co2_emissions_global_deciles")
    time_list = list(df.columns)[1:]
    states = ["D1 (Very Low)", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10 (Very High)"]
    seq = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=seq, method="OM", sm="TRATE", indel="auto")

    cluster = Cluster(om, seq.ids, clustering_method=method)
    cq = ClusterQuality(cluster, max_clusters=15)
    cq.compute_cluster_quality_scores()
    cr = ClusterResults(cluster)
    save_reference_result("co2", cq, cr, method, ks=(3, 5, 10))

    cq_direct = ClusterQuality(om, max_clusters=15, clustering_method=method)
    cq_direct.compute_cluster_quality_scores()
    direct_data = {}
    for metric, values in cq_direct.scores.items():
        direct_data[f"scores_{metric}"] = np.array(values, dtype=np.float64)
    path = os.path.join(REFERENCE_RESULTS_DIR, f"co2_direct_{method}.npz")
    np.savez(path, **direct_data)
    print(f"  Saved direct: {path}")


def run_co2_weighted(method):
    print(f"\n--- country_co2 weighted, method={method} ---")
    df = load_dataset("country_co2_emissions_global_deciles")
    time_list = list(df.columns)[1:]
    states = ["D1 (Very Low)", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10 (Very High)"]
    seq = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=seq, method="OM", sm="TRATE", indel="auto")

    n = om.shape[0] if isinstance(om, np.ndarray) else om.values.shape[0]
    rng = np.random.RandomState(42)
    weights = rng.uniform(0.5, 2.0, size=n)

    cluster = Cluster(om, seq.ids, clustering_method=method, weights=weights)
    cq = ClusterQuality(cluster, max_clusters=15)
    cq.compute_cluster_quality_scores()
    cr = ClusterResults(cluster)
    save_reference_result("co2_weighted", cq, cr, method, ks=(3, 5))


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    for m in ("ward_d", "ward_d2"):
        run_co2_dataset(m)
    run_co2_weighted("ward_d2")

    print("\n=== All hierarchical clustering reference results saved ===")
    for f in sorted(os.listdir(REFERENCE_RESULTS_DIR)):
        print(f"  {f}")
