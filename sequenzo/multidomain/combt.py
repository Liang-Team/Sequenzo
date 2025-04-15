"""
@Author  : Yuqi Liang 梁彧祺
@File    : combt_utils.py
@Time    : 15/04/2025 21:30
@Desc    : Modular utility functions for CombT (Combined Typology) strategy.
           Split into reusable components to give users control over distance calculation, clustering, and label merging.
"""
import numpy as np
import pandas as pd
from sequenzo.dissimilarity_measures import get_distance_matrix


def compute_domain_distances(sequence_objects, method_params) -> list[np.ndarray]:
    """
    Compute dissimilarity matrices for multiple domains.

    Parameters:
    - sequence_objects: List of SequenceData instances.
    - method_params: List of dicts, each with parameters for get_distance_matrix().

    Returns:
    - List of distance matrices (NumPy arrays), one per domain.
    """
    if method_params is None or len(method_params) != len(sequence_objects):
        raise ValueError("[CombT] Number of method_params must match number of domains.")

    distances = []
    for seq, params in zip(sequence_objects, method_params):
        diss = get_distance_matrix(seqdata=seq, **params)
        distances.append(diss)
    return distances


def assemble_combined_typology(cluster_labels: list[np.ndarray], ids: np.ndarray, sep: str = "+") -> pd.Series:
    """
    Assemble a combined typology label from domain-specific cluster labels.

    Parameters:
    - cluster_labels: List of label arrays, each from a domain (must be same length).
    - ids: Entity IDs to attach as Series index.
    - sep: Separator between domain cluster labels.

    Returns:
    - Pandas Series with combined typology labels.
    """
    n = len(cluster_labels[0])
    assert all(len(cl) == n for cl in cluster_labels), "[CombT] Cluster label arrays must have the same length."

    combined = [sep.join(str(cl[i]) for cl in cluster_labels) for i in range(n)]
    return pd.Series(combined, index=ids, name="CombT")


def get_combt_membership_table(ids: np.ndarray,
                               cluster_labels: list[np.ndarray],
                               combined_typology: pd.Series,
                               domain_names: list[str] = None) -> pd.DataFrame:
    """
    Build a full membership table with domain-level cluster assignments and final combined typology.

    Parameters:
    - ids: Entity IDs (used as index).
    - cluster_labels: List of cluster arrays (1D NumPy arrays).
    - combined_typology: Series of combined typology labels.
    - domain_names: Optional list of domain names to label the cluster columns.

    Returns:
    - Pandas DataFrame with domain clusters + combined typology.
    """
    df = pd.DataFrame(index=ids)

    if domain_names is None:
        domain_names = [f"Domain_{i + 1}" for i in range(len(cluster_labels))]

    for name, labels in zip(domain_names, cluster_labels):
        df[f"{name}_Cluster"] = labels

    df["CombT"] = combined_typology
    return df


# ===== EXAMPLE USAGE =====
if __name__ == '__main__':
    from sequenzo import *
    from sequenzo.clustering.hierarchical_clustering import Cluster

    left_df = load_dataset('biofam_left_domain')
    children_df = load_dataset('biofam_child_domain')
    married_df = load_dataset('biofam_married_domain')

    time_cols = [col for col in children_df.columns if col.startswith("age_")]

    seq_left = SequenceData(left_df, time_type="age", time=time_cols, states=[0, 1],
                            labels=["At home", "Left home"], id_col="id")
    seq_child = SequenceData(children_df, time_type="age", time=time_cols, states=[0, 1],
                             labels=["No child", "Child"], id_col="id")
    seq_marr = SequenceData(married_df, time_type="age", time=time_cols, states=[0, 1],
                            labels=["Not married", "Married"], id_col="id")

    domains = [seq_left, seq_child, seq_marr]
    method_params = [
        {"method": "OM", "sm": "TRATE", "indel": "auto"},
        {"method": "OM", "sm": "CONSTANT", "indel": "auto"},
        {"method": "OM", "sm": "CONSTANT", "indel": 1},
    ]

    diss_matrices = compute_domain_distances(domains, method_params)

    cluster_labels = []
    for diss, seq in zip(diss_matrices, domains):
        clus = Cluster(matrix=diss, entity_ids=seq.ids)
        labels = clus.get_cluster_labels(num_clusters=4)
        cluster_labels.append(labels)

    combt_series = assemble_combined_typology(cluster_labels, ids=domains[0].ids)
    membership_df = get_combt_membership_table(domains[0].ids, cluster_labels, combt_series,
                                               domain_names=["Left", "Child", "Married"])

    print(membership_df.head())
