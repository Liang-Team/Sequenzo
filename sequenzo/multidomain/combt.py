"""
@Author  : Yuqi Liang 梁彧祺
@File    : combt.py
@Time    : 15/04/2025 21:30
@Desc    : Modular utility functions for CombT (Combined Typology) strategy.
           Split into reusable components to give users control over distance calculation, clustering, and label merging.

           raw ASW - 不低于0.5 也能接受 ， 6
"""
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
import os
# 如果没有 DISPLAY 环境变量（说明是服务器/终端），才设置为 Agg
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

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
    freq_table.to_csv("freq_table.csv", index=False)
    print("\n[>] freq_table.csv has been saved.")

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

    return diss_matrices, membership_df


def compute_silhouette_score(diss_matrix, labels):
    """
    Wrapper to compute silhouette with precomputed distance.
    """
    return silhouette_score(diss_matrix, labels, metric='precomputed')


def merge_sparse_combt_types(distance_matrix,
                              labels,
                              min_size=30,  # For sample size about 2,000
                              verbose=True):
    """
    Merge sparse CombT labels based on silhouette score threshold (ASW >= 0.5 strategy).

    Parameters:
    - distance_matrix: np.ndarray or pd.DataFrame, full square dissimilarity matrix.
    - labels: array-like of original CombT string labels.
    - min_size: int, minimum samples per allowed group.
    - verbose: print steps or not.

    Returns:
    - new_labels: numpy array of updated labels after merging.
    """
    labels = np.array(labels)
    label_counts = pd.Series(labels).value_counts()
    original_labels = sorted(label_counts.index.tolist())

    label_map = {label: f"C{i+1}" for i, label in enumerate(original_labels)}
    numeric_labels = np.array([label_map[l] for l in labels])
    reverse_map = {v: k for k, v in label_map.items()}

    unique_labels = np.unique(numeric_labels)
    current_score = compute_silhouette_score(distance_matrix, numeric_labels)

    while current_score >= 0.5:
        counts = Counter(numeric_labels)
        small_clusters = [lab for lab, cnt in counts.items() if cnt < min_size]
        if not small_clusters:
            break

        merged = False
        for small in small_clusters:
            other_labels = [lab for lab in np.unique(numeric_labels) if lab != small]

            best_score = -np.inf
            best_target = None

            for target in other_labels:
                temp_labels = numeric_labels.copy()
                temp_labels[temp_labels == small] = target
                try:
                    score = compute_silhouette_score(distance_matrix, temp_labels)
                    if score > best_score:
                        best_score = score
                        best_target = target
                except Exception:
                    continue

            if best_score >= 0.5:
                numeric_labels[numeric_labels == small] = best_target
                current_score = best_score
                if verbose:
                    print(f"[+] Merged {small} → {best_target} | New ASW: {current_score:.4f}")
                merged = True
                break

        if not merged:
            break

    # Convert back to readable labels
    merged_map = {old: reverse_map[old] for old in np.unique(numeric_labels)}
    new_combined = [merged_map[l] for l in numeric_labels]

    original_cluster_count = len(set(labels))
    final_cluster_count = len(set(new_combined))

    print(f"\n[>] CombT clusters before merging: {original_cluster_count}")
    print(f"[>] CombT clusters after merging: {final_cluster_count}")

    return np.array(new_combined)


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

    # TODO：一定要让大家理解这个order很重要 domains = [seq_left, seq_child, seq_marr] 和下面的domains 和 domain_names
    diss_matrices, membership_df = get_interactive_combined_typology(domains, method_params, domain_names=["Left", "Child", "Married"])

    dat_matrix = compute_dat_distance_matrix(domains, method_params=method_params)

    # CombT as the label
    labels = membership_df["CombT"].values

    # Merge sparse clusters
    # 也要让大家理解，要看他们的 proportion
    merged_labels = merge_sparse_combt_types(dat_matrix, labels, min_size=50)

    # Update the membership table
    membership_df["CombT_Merged"] = merged_labels

    membership_df.reset_index().rename(columns={"index": "id"}).to_csv("combt_membership_table.csv", index=False)
    print("\n[>] combt_membership_table.csv has been saved.")