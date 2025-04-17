"""
@Author  : Yuqi Liang 梁彧祺
@File    : combt.py
@Time    : 15/04/2025 21:30
@Desc    : Modular utility functions for CombT (Combined Typology) strategy.
           Split into reusable components to give users control over distance calculation, clustering, and label merging.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 设置后端：非交互式（避免 show() 崩溃）
import matplotlib.pyplot as plt

import seaborn as sns
from sequenzo.dissimilarity_measures import get_distance_matrix
from sequenzo.clustering.hierarchical_clustering import Cluster, ClusterQuality
from sequenzo.visualization.utils.utils import save_and_show_results


def compute_domain_distances(sequence_objects, method_params) -> list[np.ndarray]:
    if method_params is None or len(method_params) != len(sequence_objects):
        raise ValueError("[CombT] Number of method_params must match number of domains.")

    distances = []
    for seq, params in zip(sequence_objects, method_params):
        diss = get_distance_matrix(seqdata=seq, **params)
        distances.append(diss)
    return distances


def assemble_combined_typology(cluster_labels: list[np.ndarray], ids: np.ndarray, sep: str = "+") -> pd.Series:
    n = len(cluster_labels[0])
    assert all(len(cl) == n for cl in cluster_labels), "[CombT] Cluster label arrays must have the same length."
    combined = [sep.join(str(cl[i]) for cl in cluster_labels) for i in range(n)]
    return pd.Series(combined, index=ids, name="CombT")

def get_combt_membership_table(ids: np.ndarray,
                                cluster_labels: list[np.ndarray],
                                combined_typology: pd.Series,
                                domain_names: list[str] = None) -> pd.DataFrame:
    df = pd.DataFrame(index=ids)
    if domain_names is None:
        domain_names = [f"Domain_{i + 1}" for i in range(len(cluster_labels))]
    for name, labels in zip(domain_names, cluster_labels):
        df[f"{name}_Cluster"] = labels
    df["CombT"] = combined_typology
    return df


def get_interactive_combined_typology(domains, method_params, domain_names=None, norm="zscore"):
    diss_matrices = compute_domain_distances(domains, method_params)

    cluster_labels = []
    ids = domains[0].ids

    if domain_names is None:
        domain_names = [f"Domain_{i + 1}" for i in range(len(domains))]

    for i, (diss, seq) in enumerate(zip(diss_matrices, domains)):
        print(f"\n[>] Processing domain: {domain_names[i]}")
        clus = Cluster(matrix=diss, entity_ids=seq.ids)
        quality = ClusterQuality(clus)
        quality.compute_cluster_quality_scores()
        quality.plot_combined_scores(norm=norm,
                                     title=f"Cluster Quality - {domain_names[i]}",
                                     save_as=f"Cluster Quality - {domain_names[i]}")

        while True:
            try:
                print(f"Cluster Quality - {domain_names[i]}.png has been saved. Please check it and then come back.\n")
                k = int(input(f"[?] Enter number of clusters for domain '{domain_names[i]}': "))
                labels = clus.get_cluster_labels(num_clusters=k)
                cluster_labels.append(labels)
                break
            except Exception as e:
                print(f"[!] Invalid input: {e}. Please try again.")

    combt_series = assemble_combined_typology(cluster_labels, ids=ids)
    membership_df = get_combt_membership_table(ids, cluster_labels, combt_series, domain_names)

    print("\n[>] Combined Typology Membership Table Preview:")
    print(membership_df.reset_index().rename(columns={"index": "id"}).head())

    membership_df.reset_index().rename(columns={"index": "id"}).to_csv("combt_membership_table.csv", index=False)
    print("\n[>] combt_membership_table.csv has been saved.")

    # Output frequency and proportion table
    freq_table = membership_df["CombT"].value_counts().reset_index()
    freq_table.columns = ["CombT", "Frequency"]
    freq_table["Proportion (%)"] = (freq_table["Frequency"] / freq_table["Frequency"].sum() * 100).round(2)

    print("\n[>] CombT Frequency Table:")
    print(freq_table)

    # Optional bar plot
    plt.figure(figsize=(10, 5))
    sns.barplot(data=freq_table, x="CombT", y="Proportion (%)", color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Frequency of Combined Typologies")
    plt.xlabel("CombT")
    plt.ylabel("Frequency")
    plt.tight_layout()
    save_and_show_results(save_as="Frequency of Combined Typologies")
    print("\nFrequency of Combined Typologies.png has been saved.")

    return membership_df


if __name__ == '__main__':
    from sequenzo import *

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

    membership_df = get_interactive_combined_typology(domains, method_params, domain_names=["Left", "Child", "Married"])