"""
Verify hierarchical clustering outputs against reference results.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sequenzo import SequenceData, get_distance_matrix, load_dataset
from sequenzo.clustering.hierarchical_clustering import Cluster, ClusterQuality, ClusterResults

REFERENCE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "reference_results")
METRIC_ORDER = ["PBC", "HG", "HGSD", "ASW", "ASWw", "CH", "R2", "CHsq", "R2sq", "HC"]


def load_reference_result(name):
    path = os.path.join(REFERENCE_RESULTS_DIR, f"{name}.npz")
    return dict(np.load(path))


def verify_cqi_scores(reference_data, cq, tag):
    ok = True
    for metric in METRIC_ORDER:
        key = f"scores_{metric}"
        expected = reference_data[key]
        actual = np.array(cq.scores[metric], dtype=np.float64)
        if not np.allclose(expected, actual, atol=1e-10, equal_nan=True):
            print(f"  FAIL [{tag}] scores.{metric}: max diff = {np.nanmax(np.abs(expected - actual))}")
            ok = False
    return ok


def verify_cqi_table(reference_data, cq, tag):
    table = cq.get_cqi_table()
    ok = True
    for col, key in [
        ("Opt. Clusters", "cqi_opt_clusters"),
        ("Raw Value", "cqi_raw_value"),
        ("Z-Score Norm.", "cqi_zscore"),
    ]:
        expected = reference_data[key]
        actual = table[col].values.astype(np.float64)
        if not np.allclose(expected, actual, atol=1e-10, equal_nan=True):
            print(f"  FAIL [{tag}] cqi_table.{col}: max diff = {np.nanmax(np.abs(expected - actual))}")
            ok = False
    return ok


def verify_range_table(reference_data, cq, tag):
    table = cq.get_cluster_range_table()
    expected = reference_data["range_table"]
    actual = table.values.astype(np.float64)
    if not np.allclose(expected, actual, atol=1e-10, equal_nan=True):
        print(f"  FAIL [{tag}] range_table: max diff = {np.nanmax(np.abs(expected - actual))}")
        return False
    return True


def verify_cluster_results(reference_data, cr, tag, ks=(3, 5, 10)):
    ok = True
    for k in ks:
        key = f"labels_k{k}"
        if key not in reference_data:
            continue
        membership = cr.get_cluster_memberships(k)
        actual_labels = membership["Cluster"].values.astype(np.int32)
        expected_labels = reference_data[key]
        if not np.array_equal(expected_labels, actual_labels):
            print(f"  FAIL [{tag}] labels k={k}")
            ok = False

        dist = cr.get_cluster_distribution(k, weighted=True)
        for col, suffix in [
            ("Cluster", "cluster"),
            ("Count", "count"),
            ("Percentage", "pct"),
            ("Weight_Sum", "wsum"),
            ("Weight_Percentage", "wpct"),
        ]:
            exp_key = f"dist_{suffix}_k{k}"
            expected = reference_data[exp_key]
            actual = dist[col].values
            if col in ("Cluster", "Count"):
                actual = actual.astype(np.int32)
                if not np.array_equal(expected, actual):
                    print(f"  FAIL [{tag}] dist.{col} k={k}")
                    ok = False
            else:
                actual = actual.astype(np.float64)
                expected = expected.astype(np.float64)
                if not np.allclose(expected, actual, atol=1e-10, equal_nan=True):
                    print(f"  FAIL [{tag}] dist.{col} k={k}: max diff = {np.nanmax(np.abs(expected - actual))}")
                    ok = False
    return ok


def run_verification(method):
    print(f"\n=== Verifying method={method} ===")
    df = load_dataset("country_co2_emissions_global_deciles")
    time_list = list(df.columns)[1:]
    states = ["D1 (Very Low)", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10 (Very High)"]
    seq = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=seq, method="OM", sm="TRATE", indel="auto")

    cluster = Cluster(om, seq.ids, clustering_method=method)
    cq = ClusterQuality(cluster, max_clusters=15)
    cq.compute_cluster_quality_scores()
    cr = ClusterResults(cluster)

    reference_data = load_reference_result(f"co2_{method}")
    tag = f"co2_{method}"

    all_ok = True
    all_ok &= verify_cqi_scores(reference_data, cq, tag)
    all_ok &= verify_cqi_table(reference_data, cq, tag)
    all_ok &= verify_range_table(reference_data, cq, tag)
    all_ok &= verify_cluster_results(reference_data, cr, tag)

    if all_ok:
        print(f"  PASS: all checks for {tag}")

    print(f"\n  --- Verifying direct matrix path for {method} ---")
    cq_direct = ClusterQuality(om, max_clusters=15, clustering_method=method)
    reference_direct = load_reference_result(f"co2_direct_{method}")
    tag_d = f"co2_direct_{method}"
    direct_ok = True
    for metric in METRIC_ORDER:
        key = f"scores_{metric}"
        expected = reference_direct[key]
        actual = np.array(cq_direct.scores[metric], dtype=np.float64)
        if not np.allclose(expected, actual, atol=1e-10, equal_nan=True):
            print(f"  FAIL [{tag_d}] scores.{metric}: max diff = {np.nanmax(np.abs(expected - actual))}")
            direct_ok = False
    if direct_ok:
        print(f"  PASS: all checks for {tag_d}")
    all_ok &= direct_ok
    return all_ok


def run_weighted_verification():
    print("\n=== Verifying weighted ward_d2 ===")
    df = load_dataset("country_co2_emissions_global_deciles")
    time_list = list(df.columns)[1:]
    states = ["D1 (Very Low)", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10 (Very High)"]
    seq = SequenceData(df, time=time_list, id_col="country", states=states, labels=states)
    om = get_distance_matrix(seqdata=seq, method="OM", sm="TRATE", indel="auto")

    n = om.shape[0] if isinstance(om, np.ndarray) else om.values.shape[0]
    rng = np.random.RandomState(42)
    weights = rng.uniform(0.5, 2.0, size=n)

    cluster = Cluster(om, seq.ids, clustering_method="ward_d2", weights=weights)
    cq = ClusterQuality(cluster, max_clusters=15)
    cr = ClusterResults(cluster)

    reference_data = load_reference_result("co2_weighted_ward_d2")
    tag = "co2_weighted_ward_d2"

    all_ok = True
    all_ok &= verify_cqi_scores(reference_data, cq, tag)
    all_ok &= verify_cqi_table(reference_data, cq, tag)
    all_ok &= verify_range_table(reference_data, cq, tag)
    all_ok &= verify_cluster_results(reference_data, cr, tag, ks=(3, 5))

    if all_ok:
        print(f"  PASS: all checks for {tag}")
    return all_ok


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    all_pass = True
    for m in ("ward_d", "ward_d2"):
        all_pass &= run_verification(m)
    all_pass &= run_weighted_verification()

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL VERIFICATIONS PASSED")
    else:
        print("SOME VERIFICATIONS FAILED")
    sys.exit(0 if all_pass else 1)
