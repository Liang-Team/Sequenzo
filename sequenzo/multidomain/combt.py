"""
@Author  : Yuqi Liang 梁彧祺
@File    : combt.py
@Time    : 15/04/2025 21:30
@Desc    : Modular utility functions for CombT (Combined Typology) strategy.
           Split into reusable components to give users control over distance calculation, clustering, and label merging.
"""
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
import os

# Set to Agg backend if no DISPLAY environment variable (server/terminal)
# if os.environ.get("DISPLAY", "") == "":
#     matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

from sequenzo.dissimilarity_measures import get_distance_matrix
from sequenzo.clustering.hierarchical_clustering import Cluster, ClusterQuality
from sequenzo.visualization.utils.utils import save_and_show_results


def _compute_domain_distances(sequence_objects, method_params) -> list[np.ndarray]:
    """
    Compute distance matrices for each domain using specified methods.

    Parameters:
    - sequence_objects: List of sequence data objects, one per domain
    - method_params: List of parameter dictionaries for distance computation, one per domain

    Returns:
    - List of distance matrices, one per domain
    """
    # Validate input parameters
    if method_params is None or len(method_params) != len(sequence_objects):
        raise ValueError("[CombT] Number of method_params must match number of domains.")

    for i, params in enumerate(method_params):
        if "method" not in params:
            raise ValueError(f"[CombT] Required parameter 'method' missing in method_params[{i}]")

    distances = []
    for seq, params in zip(sequence_objects, method_params):
        diss = get_distance_matrix(seqdata=seq, **params)
        distances.append(diss)
    return distances


def _assemble_combined_typology(cluster_labels: list[np.ndarray], ids: np.ndarray, sep: str = "+") -> pd.Series:
    """
    Assemble the combined typology from individual domain cluster labels.

    Parameters:
    - cluster_labels: List of cluster label arrays, one per domain
    - ids: Array of sequence IDs
    - sep: Separator to use for combining label strings (default: "+")

    Returns:
    - Pandas Series containing the combined typology labels with IDs as index
    """
    n = len(cluster_labels[0])
    assert all(len(cl) == n for cl in cluster_labels), "[CombT] Cluster label arrays must have the same length."
    combined = [sep.join(str(cl[i]) for cl in cluster_labels) for i in range(n)]
    return pd.Series(combined, index=ids, name="CombT")


def _get_combt_membership_table(ids: np.ndarray,
                                cluster_labels: list[np.ndarray],
                                combined_typology: pd.Series,
                                domain_names: list[str] = None) -> pd.DataFrame:
    """
    Create a membership table that shows the domain cluster and combined typology for each ID.

    Parameters:
    - ids: Array of sequence IDs
    - cluster_labels: List of cluster label arrays, one per domain
    - combined_typology: Combined typology labels
    - domain_names: Optional list of domain names (default: Domain_1, Domain_2, etc.)

    Returns:
    - DataFrame with domain cluster memberships and combined typology
    """
    df = pd.DataFrame(index=ids)
    # Use domain_names if provided, otherwise generate default names
    domain_names = domain_names or [f"Domain_{i + 1}" for i in range(len(cluster_labels))]

    for name, labels in zip(domain_names, cluster_labels):
        df[f"{name}_Cluster"] = labels
    df["CombT"] = combined_typology
    return df


def get_interactive_combined_typology(domains, method_params, domain_names=None, norm="zscore",
                                      interactive=True, predefined_clusters=None):
    """
    Interactive or automated interface for the CombT workflow.
    """
    # 首先导入必要的库并设置matplotlib后端
    import matplotlib
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # 在函数开始处强制使用Agg后端

    try:
        diss_matrices = _compute_domain_distances(domains, method_params)

        cluster_labels = []
        ids = domains[0].ids

        # Use domain_names if provided, otherwise generate default names
        domain_names = domain_names or [f"Domain_{i + 1}" for i in range(len(domains))]

        # Check if predefined clusters are provided for non-interactive mode
        if not interactive and predefined_clusters is not None:
            if len(predefined_clusters) != len(domains):
                raise ValueError("[CombT] Number of predefined clusters must match number of domains.")

            for i, (diss, seq) in enumerate(zip(diss_matrices, domains)):
                print(f"\n[>] Processing domain: {domain_names[i]}")
                clus = Cluster(matrix=diss, entity_ids=seq.ids)
                k = predefined_clusters[i]
                print(f"[>] Using predefined number of clusters for domain '{domain_names[i]}': {k}")
                labels = clus.get_cluster_labels(num_clusters=k)
                cluster_labels.append(labels)

        else:  # Interactive mode
            for i, (diss, seq) in enumerate(zip(diss_matrices, domains)):
                print(f"\n[>] Processing domain: {domain_names[i]}")
                clus = Cluster(matrix=diss, entity_ids=seq.ids)
                quality = ClusterQuality(clus)
                quality.compute_cluster_quality_scores()

                try:
                    # 使用savefig替代save_and_show_results
                    fig = quality.plot_combined_scores(norm=norm,
                                                       title=f"Cluster Quality - {domain_names[i]}",
                                                       show=False)  # 不显示，只返回图形对象

                    # 直接保存图形而不显示
                    fig.savefig(f"Cluster Quality - {domain_names[i]}.png", dpi=200)
                    plt.close(fig)  # 确保关闭图形

                    print(
                        f"[>] Cluster Quality - {domain_names[i]}.png has been saved. Please check it and then come back.\n")

                except Exception as e:
                    print(f"[!] Warning: Could not create or save plot: {e}")
                    print(f"[>] Continuing without visualization...")

                while True:
                    try:
                        k = int(input(f"[?] Enter number of clusters for domain '{domain_names[i]}': "))
                        labels = clus.get_cluster_labels(num_clusters=k)
                        cluster_labels.append(labels)
                        break
                    except Exception as e:
                        print(f"[!] Invalid input: {e}. Please try again.")

        combt_series = _assemble_combined_typology(cluster_labels, ids=ids)
        membership_df = _get_combt_membership_table(ids, cluster_labels, combt_series, domain_names)

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

        # Optional bar plot - with error handling
        try:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=freq_table, x="CombT", y="Proportion (%)", color="skyblue")
            plt.xticks(rotation=45, ha="right")
            plt.title("Frequency of Combined Typologies")
            plt.xlabel("CombT")
            plt.ylabel("Frequency")
            plt.tight_layout()

            # 直接保存图形而不显示
            plt.savefig("Frequency of Combined Typologies.png", dpi=200)
            plt.close()  # 确保关闭图形

            print("\n[>] Frequency of Combined Typologies.png has been saved.")
        except Exception as e:
            print(f"[!] Warning: Could not create frequency plot: {e}")

        return diss_matrices, membership_df

    finally:
        # 无论如何，在函数结束时恢复原始后端
        matplotlib.use(original_backend)


def _compute_silhouette_score(diss_matrix, labels):
    """
    Compute silhouette score with precomputed distance matrix.

    Parameters:
    - diss_matrix: Square distance matrix
    - labels: Cluster labels

    Returns:
    - Silhouette score
    """
    return silhouette_score(diss_matrix, labels, metric='precomputed')


def merge_sparse_combt_types(distance_matrix,
                             labels,
                             min_size=30,  # For sample size about 2,000
                             asw_threshold=0.5,  # Silhouette score threshold
                             verbose=True,
                             print_merge_details=True,
                             visualize_process=True,
                             visualization_path="merge_progress.png"):
    """
    Merge sparse CombT labels based on silhouette score threshold strategy.

    This implements the algorithm described in the CombT paper to avoid scarce
    combined types by merging them while maintaining cluster quality.

    Parameters:
    - distance_matrix: np.ndarray or pd.DataFrame, full square dissimilarity matrix.
    - labels: array-like of original CombT string labels.
    - min_size: int, minimum samples per allowed group.
    - asw_threshold: float, minimum silhouette score threshold to accept a merge.
    - verbose: bool, print steps or not.
    - print_merge_details: bool, whether to print detailed merge history at the end.
    - visualize_process: bool, whether to create a visualization of the merge process.
    - visualization_path: str, file path to save the visualization (if visualize_process=True).

    Returns:
    - new_labels: numpy array of updated labels after merging.
    - merge_info: dict, containing merge history and quality information.
    """
    # Parameter validation
    if not isinstance(distance_matrix, (np.ndarray, pd.DataFrame)):
        raise TypeError("distance_matrix must be numpy.ndarray or pandas.DataFrame")

    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square (n × n)")

    labels = np.array(labels)
    if len(labels) != distance_matrix.shape[0]:
        raise ValueError(
            f"Length of labels ({len(labels)}) does not match distance matrix dimensions ({distance_matrix.shape[0]})")

    # Track merge history and quality metrics
    merge_info = {
        "merge_history": [],
        "initial_silhouette": None,
        "final_silhouette": None,
        "initial_cluster_count": None,
        "final_cluster_count": None,
        "small_clusters_merged": 0
    }

    label_counts = pd.Series(labels).value_counts()
    original_labels = sorted(label_counts.index.tolist())

    # Record initial state
    merge_info["initial_cluster_count"] = len(original_labels)

    # Create a mapping between string labels and numeric labels
    label_map = {label: f"C{i + 1}" for i, label in enumerate(original_labels)}
    numeric_labels = np.array([label_map[l] for l in labels])
    reverse_map = {v: k for k, v in label_map.items()}

    unique_labels = np.unique(numeric_labels)
    current_score = _compute_silhouette_score(distance_matrix, numeric_labels)
    merge_info["initial_silhouette"] = current_score

    if verbose:
        print(f"[>] Initial clusters: {len(unique_labels)}, Initial ASW: {current_score:.4f}")
        print(f"[>] Beginning merge process with min_size={min_size} and ASW threshold={asw_threshold}")

    total_merges = 0
    iterations = 0

    # Main merging loop
    while current_score >= asw_threshold:
        iterations += 1
        if verbose and iterations % 10 == 0:
            print(f"[>] Iteration {iterations}, current clusters: {len(np.unique(numeric_labels))}")

        counts = Counter(numeric_labels)
        small_clusters = [lab for lab, cnt in counts.items() if cnt < min_size]

        # Exit if no small clusters remain
        if not small_clusters:
            break

        merged = False
        for small in small_clusters:
            other_labels = [lab for lab in np.unique(numeric_labels) if lab != small]

            best_score = -np.inf
            best_target = None

            # Find best merge target
            for target in other_labels:
                temp_labels = numeric_labels.copy()
                temp_labels[temp_labels == small] = target
                try:
                    score = _compute_silhouette_score(distance_matrix, temp_labels)
                    if score > best_score:
                        best_score = score
                        best_target = target
                except Exception as e:
                    if verbose:
                        print(f"[!] Error computing silhouette for merge {small} → {target}: {e}")
                    continue

            # Execute merge if it maintains quality threshold
            if best_score >= asw_threshold:
                old_count = counts[small]
                numeric_labels[numeric_labels == small] = best_target
                current_score = best_score

                # Record merge details
                merge_details = {
                    "iteration": iterations,
                    "source": reverse_map[small],
                    "target": reverse_map[best_target],
                    "source_size": old_count,
                    "new_asw": best_score
                }
                merge_info["merge_history"].append(merge_details)
                merge_info["small_clusters_merged"] += 1
                total_merges += 1

                if verbose:
                    print(
                        f"[+] Merged {small} ({reverse_map[small]}, size={old_count}) → {best_target} ({reverse_map[best_target]}) | New ASW: {current_score:.4f}")

                merged = True
                break

        # Exit if no suitable merges found
        if not merged:
            if verbose:
                print(f"[!] No suitable merges found that maintain ASW >= {asw_threshold}")
            break

    # Convert back to original label format
    merged_map = {old: reverse_map[old] for old in np.unique(numeric_labels)}
    new_combined = [merged_map[l] for l in numeric_labels]

    original_cluster_count = len(set(labels))
    final_cluster_count = len(set(new_combined))

    # Update final metrics
    merge_info["final_silhouette"] = current_score
    merge_info["final_cluster_count"] = final_cluster_count
    merge_info["total_merges"] = total_merges

    if verbose:
        print(f"\n[>] CombT clusters before merging: {original_cluster_count}")
        print(f"[>] CombT clusters after merging: {final_cluster_count}")
        print(f"[>] Total merges performed: {total_merges}")
        print(f"[>] Final ASW: {current_score:.4f}")

    # Print merge history details if requested
    if verbose and print_merge_details and merge_info["merge_history"]:
        print("\n[>] Merge History Details:")
        for i, merge in enumerate(merge_info["merge_history"]):
            print(
                f"  Merge {i + 1}: {merge['source']} (size={merge['source_size']}) → {merge['target']} | ASW: {merge['new_asw']:.4f}")

    # Visualize merge process if requested
    if visualize_process and merge_info["merge_history"]:
        try:
            _plot_merge_progress(merge_info, save_as=visualization_path)
            if verbose:
                print(f"\n[>] Merge process visualization saved to: {visualization_path}")
        except Exception as e:
            if verbose:
                print(f"[!] Warning: Could not create merge visualization: {e}")

    return np.array(new_combined), merge_info


def _plot_merge_progress(merge_info, save_as=None):
    """
    Internal function to visualize the progress of the cluster merging process.

    Parameters:
    - merge_info: dict, merge information returned by merge_sparse_combt_types
    - save_as: str, filename to save the plot
    """
    if not merge_info["merge_history"]:
        print("No merges were performed.")
        return

    # 保存当前matplotlib后端并临时切换到Agg
    import matplotlib
    current_backend = matplotlib.get_backend()

    try:
        # 使用上下文管理器临时更改后端
        with plt.rc_context({'backend': 'Agg'}):
            # Extract data
            iterations = [m["iteration"] for m in merge_info["merge_history"]]
            asw_scores = [m["new_asw"] for m in merge_info["merge_history"]]

            # Calculate cluster counts at each iteration
            clusters = [merge_info["initial_cluster_count"]]
            for i in range(len(iterations)):
                clusters.append(clusters[-1] - 1)  # Each merge reduces clusters by 1

            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # ASW scores plot
            ax1.plot(iterations, asw_scores, 'o-', color='blue')
            ax1.set_ylabel('Silhouette Score (ASW)')
            ax1.set_title('Merge Process Progress')
            ax1.grid(True, linestyle='--', alpha=0.7)

            # Cluster count plot
            ax2.plot(iterations, clusters[1:], 'o-', color='green')
            ax2.set_xlabel('Merge Iteration')
            ax2.set_ylabel('Number of Clusters')
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Add reference lines
            ax1.axhline(y=merge_info["initial_silhouette"], color='blue', linestyle='--', alpha=0.5,
                        label=f'Initial ASW: {merge_info["initial_silhouette"]:.4f}')
            ax1.axhline(y=merge_info["final_silhouette"], color='red', linestyle='--', alpha=0.5,
                        label=f'Final ASW: {merge_info["final_silhouette"]:.4f}')
            ax1.legend()

            ax2.axhline(y=merge_info["initial_cluster_count"], color='green', linestyle='--', alpha=0.5,
                        label=f'Initial clusters: {merge_info["initial_cluster_count"]}')
            ax2.axhline(y=merge_info["final_cluster_count"], color='red', linestyle='--', alpha=0.5,
                        label=f'Final clusters: {merge_info["final_cluster_count"]}')
            ax2.legend()

            plt.tight_layout()

            if save_as:
                plt.savefig(save_as, dpi=200)

            plt.close()  # 确保关闭图形

    except Exception as e:
        print(f"[!] Error creating merge progress plot: {e}")

    finally:
        # 恢复原始后端
        matplotlib.use(current_backend)


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

    # NOTE: The order of domains is critical - must match between domains list and domain_names
    diss_matrices, membership_df = get_interactive_combined_typology(domains,
                                                                     method_params,
                                                                     domain_names=["Left", "Child", "Married"])

    dat_matrix = compute_dat_distance_matrix(domains, method_params=method_params)

    # Use CombT as the label
    labels = membership_df["CombT"].values

    # Merge sparse clusters - important to check the proportions before deciding min_size
    merged_labels, merge_info = merge_sparse_combt_types(distance_matrix=dat_matrix,
                                                         labels=labels,
                                                         min_size=50,
                                                         asw_threshold=0.5,
                                                         verbose=True,
                                                         # Optional parameters below, the default is True
                                                         print_merge_details=True,
                                                         visualize_process=True,
                                                         visualization_path="merge_progress_combt.png"
                                                         )

    # Update the membership dataframe
    membership_df["CombT_Merged"] = merged_labels

    # Save results
    membership_df.reset_index().rename(columns={"index": "id"}).to_csv("combt_membership_table.csv", index=False)
    print("\n[>] combt_membership_table.csv has been saved.")
