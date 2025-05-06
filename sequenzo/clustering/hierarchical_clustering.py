"""
@Author  : Yuqi Liang 梁彧祺
@File    : hierarchical_clustering.py
@Time    : 18/12/2024 17:59
@Desc    :
    This module provides a flexible and user-friendly implementation of hierarchical clustering,
    along with tools to evaluate cluster quality and analyze clustering results.

    It supports common hierarchical clustering methods and evaluation metrics,
    designed for social sequence analysis and other research applications.

    This module leverages fastcluster, a tool specifically designed to enhance the efficiency of large-scale hierarchical clustering.
    Unlike native Python tools such as SciPy, fastcluster optimizes linkage matrix computations,
    enabling it to handle datasets with millions of entries more efficiently.

    It has three main components:
    1. Cluster Class: Performs hierarchical clustering on a precomputed distance matrix.
    2. ClusterQuality Class: Evaluates the quality of clustering for different numbers of clusters using various metrics.
    3. ClusterResults Class: Analyzes and visualizes the clustering results (e.g., membership tables and cluster distributions).

    Note that the CQI equivalence of R is here: https://github.com/cran/WeightedCluster/blob/master/src/clusterquality.cpp
"""
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, silhouette_samples
from fastcluster import linkage

from .utils.point_biserial import point_biserial


# Corrected imports: Use relative imports *within* the package.
from ..visualization.utils import save_and_show_results


class Cluster:
    def __init__(self,
                 matrix,
                 entity_ids,
                 clustering_method="ward"):
        """
        A class to handle hierarchical clustering operations using fastcluster for improved performance.

        :param matrix: Precomputed distance matrix (full square form).
        :param entity_ids: List of IDs corresponding to the entities in the matrix.
        :param clustering_method: Clustering algorithm to use (default: "ward").
        :param n_jobs: Number of parallel jobs to use (-1 for all available cores).
        """
        # Ensure entity_ids is a numpy array for consistent processing
        self.entity_ids = np.array(entity_ids)

        # Check if entity_ids is valid
        if len(self.entity_ids) != len(matrix):
            raise ValueError("Length of entity_ids must match the size of the matrix.")

        # Optional: Check uniqueness of entity_ids
        if len(np.unique(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids must contain unique values.")

        # Convert matrix to numpy array if it's a DataFrame
        if isinstance(matrix, pd.DataFrame):
            print("[>] Converting DataFrame to NumPy array...")
            self.full_matrix = matrix.values
        else:
            self.full_matrix = matrix

        # Verify matrix is in square form
        if len(self.full_matrix.shape) != 2 or self.full_matrix.shape[0] != self.full_matrix.shape[1]:
            raise ValueError("Input must be a full square-form distance matrix.")

        self.clustering_method = clustering_method.lower()

        # Supported clustering methods
        supported_methods = ["ward", "single", "complete", "average", "centroid", "median"]
        if self.clustering_method not in supported_methods:
            raise ValueError(
                f"Unsupported clustering method '{clustering_method}'. Supported methods: {supported_methods}")

        # Compute linkage matrix using fastcluster
        self.linkage_matrix = self._compute_linkage()

    def _compute_linkage(self):
        """
        Compute the linkage matrix using fastcluster for improved performance.
        """
        # Check for NaNs and Infs
        if np.any(np.isnan(self.full_matrix)) or np.any(np.isinf(self.full_matrix)):
            print("[!] Warning: Distance matrix contains NaN or Inf values. Replacing with maximum finite value...")
            max_valid = np.nanmax(self.full_matrix[np.isfinite(self.full_matrix)])
            self.full_matrix[~np.isfinite(self.full_matrix)] = max_valid

        # Ensure the matrix is symmetric
        if not np.allclose(self.full_matrix, self.full_matrix.T, rtol=1e-5, atol=1e-8):
            print("[!] Warning: Distance matrix is not symmetric. Symmetrizing...")
            self.full_matrix = (self.full_matrix + self.full_matrix.T) / 2

        # Convert square matrix to condensed form
        self.condensed_matrix = squareform(self.full_matrix)

        linkage_matrix= linkage(self.condensed_matrix, method=self.clustering_method)
        return linkage_matrix

    def plot_dendrogram(self,
                        save_as=None,
                        style="whitegrid",
                        title="Dendrogram",
                        xlabel="Entities",
                        ylabel="Distance",
                        grid=False,
                        dpi=200,
                        figsize=(12, 8)):
        """
        Plot a dendrogram of the hierarchical clustering with optional high-resolution output.

        :param save_as: File path to save the plot. If None, the plot will be shown.
        :param style: Seaborn style for the plot.
        :param title: Title of the plot.
        :param xlabel: X-axis label.
        :param ylabel: Y-axis label.
        :param grid: Whether to display grid lines.
        :param dpi: Dots per inch for the saved image (default: 300 for high resolution).
        :param figsize: Tuple specifying the figure size in inches (default: (12, 8)).
        """
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix is not computed.")

        sns.set(style=style)
        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix, labels=None)  # Do not plot labels for large datasets
        plt.xticks([])
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if not grid:
            plt.grid(False)

        save_and_show_results(save_as, dpi=200)

    def get_cluster_labels(self, num_clusters):
        """
        Get cluster labels for a specified number of clusters.

        There is a common point of confusion because
        k is typically used to represent the number of clusters in clustering algorithms (e.g., k-means).

        However, SciPy's hierarchical clustering API specifically uses t as the parameter name.

        :param num_clusters: The number of clusters to create.
        :return: Array of cluster labels corresponding to entity_ids.
        """
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix is not computed.")

        cluster_labels = fcluster(self.linkage_matrix, t=num_clusters, criterion="maxclust")

        return cluster_labels


class ClusterQuality:
    def __init__(self, matrix_or_cluster, max_clusters=20, clustering_method=None):
        """
        Initialize the ClusterQuality class for precomputed distance matrices or a Cluster instance.

        Allow the ClusterQuality class to directly accept a Cluster instance
        and internally extract the relevant matrix (cluster.full_matrix)
        and clustering method (cluster.clustering_method).

        This keeps the user interface clean and simple while handling the logic under the hood.

        :param matrix_or_cluster: The precomputed distance matrix (full square form or condensed form)
                                   or an instance of the Cluster class.
        :param max_clusters: Maximum number of clusters to evaluate (default: 20).
        :param clustering_method: Clustering algorithm to use. If None, inherit from Cluster instance.
        """
        if isinstance(matrix_or_cluster, Cluster):
            # Extract matrix and clustering method from the Cluster instance
            self.matrix = matrix_or_cluster.full_matrix
            self.clustering_method = matrix_or_cluster.clustering_method
            self.linkage_matrix = matrix_or_cluster.linkage_matrix

        elif isinstance(matrix_or_cluster, (np.ndarray, pd.DataFrame)):
            # Handle direct matrix input
            if isinstance(matrix_or_cluster, pd.DataFrame):
                print("[>] Detected Pandas DataFrame. Converting to NumPy array...")
                matrix_or_cluster = matrix_or_cluster.values
            self.matrix = matrix_or_cluster
            self.clustering_method = clustering_method or "ward"

        else:
            raise ValueError(
                "Input must be a Cluster instance, a NumPy array, or a Pandas DataFrame."
            )

        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be a full square-form distance matrix.")

        self.max_clusters = max_clusters
        self.scores = {
            "ASW": [],
            "ASWw": [],
            "HG": [],
            "PBC": [],
            "CH": [],
            "R2": [],
            "HC": [],
        }

    def compute_cluster_quality_scores(self):
        """
        Compute clustering quality scores for different numbers of clusters.
        """

        for k in range(2, self.max_clusters + 1):
            # fcluster() performs the final 'pruning' step on the results computed by the Rust code that generates a tree.
            labels = fcluster(self.linkage_matrix, k, criterion="maxclust")
            self.scores["ASW"].append(self._compute_silhouette(labels))
            self.scores["ASWw"].append(self._compute_weighted_silhouette(labels))
            self.scores["HG"].append(self._compute_homogeneity(labels))
            self.scores["PBC"].append(self._compute_point_biserial(labels))
            self.scores["CH"].append(self._compute_calinski_harabasz(labels))
            self.scores["R2"].append(self._compute_r2(labels))
            self.scores["HC"].append(self._compute_hierarchical_criterion(labels))

    def _compute_silhouette(self, labels) -> float:
        """
        Compute Silhouette Score (ASW).
        """
        if len(set(labels)) > 1:
            return silhouette_score(self.matrix, labels, metric="precomputed")
        return np.nan

    def _compute_weighted_silhouette(self, labels) -> float:
        """
        Compute Weighted Silhouette Score (ASWw).
        """
        sil_samples = silhouette_samples(self.matrix, labels, metric="precomputed")
        cluster_sizes = np.bincount(labels)[1:]
        total_points = len(labels)
        weights = cluster_sizes / total_points
        mean_silhouette_per_cluster = []

        for cluster_id in range(1, len(cluster_sizes) + 1):
            sil_vals = sil_samples[labels == cluster_id]
            if len(sil_vals) > 0:
                mean_silhouette_per_cluster.append(np.mean(sil_vals))
            else:
                mean_silhouette_per_cluster.append(0.0)

        return np.sum(weights * np.array(mean_silhouette_per_cluster))

    def _compute_homogeneity(self, labels) -> float:
        """
        Compute Homogeneity (HG).
        """
        cluster_sizes = np.bincount(labels)[1:]
        total_points = len(labels)
        return np.sum((cluster_sizes / total_points) ** 2)

    def _compute_point_biserial(self, labels) -> float:
        """
        Compute Point-Biserial Correlation (PBC).
        """
        # We must explicitly convert labels to the correct type
        # This ensures compatibility with the int64_t[:] Cython type signature.
        labels = np.asarray(labels, dtype=np.int64)
        score = point_biserial(self.matrix, labels)
        return score

    def _compute_calinski_harabasz(self, labels) -> float:
        """
        Pseudo Calinski-Harabasz score using distance matrix.
        Approximates between-cluster and within-cluster dispersion
        based only on distances.

        Returns a value similar in spirit to traditional CH index.
        """
        n_samples = len(labels)
        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters)

        if n_clusters <= 1 or n_clusters >= n_samples:
            return np.nan  # CH undefined

        # Compute total mean of distances (upper triangle only to avoid redundancy)
        triu_indices = np.triu_indices_from(self.matrix, k=1)
        total_mean = np.mean(self.matrix[triu_indices])

        # Initialize within and between-cluster variances
        within_ss = 0.0
        between_ss = 0.0

        for cluster in unique_clusters:
            indices = np.where(labels == cluster)[0]
            if len(indices) < 2:
                continue

            # Within-cluster sum of squares (mean squared pairwise distance)
            submatrix = self.matrix[np.ix_(indices, indices)]
            intra_dists = submatrix[np.triu_indices_from(submatrix, k=1)]
            within_mean = np.mean(intra_dists)
            within_ss += len(indices) * (within_mean ** 2)

            # Between-cluster dispersion approximated by cluster center's distance to global mean
            cluster_mean_dist = np.mean(self.matrix[indices][:, indices].flatten())
            between_ss += len(indices) * ((cluster_mean_dist - total_mean) ** 2)

        if within_ss == 0:
            return np.nan  # Avoid division by zero

        return (between_ss / (n_clusters - 1)) / (within_ss / (n_samples - n_clusters))

    def _compute_r2(self, labels) -> float:
        """
        Compute R-squared (R2).
        """
        n_samples = len(labels)
        within_cluster_sum_of_squares = sum(
            [np.sum((self.matrix[labels == cluster] - np.mean(self.matrix[labels == cluster])) ** 2)
             for cluster in np.unique(labels)]
        )
        total_sum_of_squares = np.sum((self.matrix - np.mean(self.matrix)) ** 2)
        return 1 - within_cluster_sum_of_squares / total_sum_of_squares

    def _compute_hierarchical_criterion(self, labels) -> float:
        """
        Compute Hierarchical Criterion (HC).
        """
        return np.var([np.mean(self.matrix[labels == cluster]) for cluster in np.unique(labels)])

    def _normalize_scores(self, method="zscore") -> None:
        """
        Normalize each metric independently.

        :param method: Normalization method. Options are "zscore" or "range".
        """
        for metric in self.scores:
            values = np.array(self.scores[metric])
            if method == "zscore":
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val > 0:
                    self.scores[metric] = (values - mean_val) / std_val
            elif method == "range":
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                if max_val > min_val:
                    self.scores[metric] = (values - min_val) / (max_val - min_val)

    def get_metrics_table(self):
        """
        Generate a summary table of clustering quality metrics with concise column names.

        :return: Pandas DataFrame summarizing the optimal number of clusters (N groups),
                 the corresponding metric values (stat), and normalized values (z-score and min-max normalization).
        """
        # Temporarily store original scores to avoid overwriting during normalization
        original_scores = self.scores.copy()

        # Apply z-score normalization
        self._normalize_scores(method="zscore")
        zscore_normalized = {metric: np.array(values) for metric, values in self.scores.items()}

        # Apply min-max normalization
        self.scores = original_scores.copy()  # Restore original scores
        self._normalize_scores(method="range")
        minmax_normalized = {metric: np.array(values) for metric, values in self.scores.items()}

        # Restore original scores for safety
        self.scores = original_scores

        # Generate summary table
        summary = {
            "Metric": [],
            "Opt. Clusters": [],  # Abbreviated from "Optimal Clusters"
            "Opt. Value": [],  # Abbreviated from "Optimal Value"
            "Z-Score Norm.": [],  # Abbreviated from "Z-Score Normalized Value"
            "Min-Max Norm.": []  # Abbreviated from "Min-Max Normalized Value"
        }

        # Get maximum value and its position from original scores
        for metric, values in original_scores.items():
            values = np.array(values)
            optimal_k = np.nanargmax(values) + 2  # Add 2 because k starts at 2
            max_value = values[optimal_k - 2]  # Get the original maximum value

            # Add data to the summary table
            summary["Metric"].append(metric)
            summary["Opt. Clusters"].append(optimal_k)
            summary["Opt. Value"].append(max_value)
            summary["Z-Score Norm."].append(zscore_normalized[metric][optimal_k - 2])
            summary["Min-Max Norm."].append(minmax_normalized[metric][optimal_k - 2])

        return pd.DataFrame(summary)

    def plot_combined_scores(self,
                             metrics_list=None,
                             norm="zscore",
                             palette="husl",
                             line_width=2,
                             style="whitegrid",
                             title=None,
                             xlabel="Number of Clusters",
                             ylabel="Normalized Score",
                             grid=True,
                             save_as=None,
                             dpi=200,
                             figsize=(12, 8),
                             show=True
                             ):
        """
        Plot combined scores for clustering quality metrics with customizable parameters.

        This function displays normalized metric values for easier comparison while preserving
        the original statistical properties in the legend.

        It first calculates raw means and standard deviations from the original data before applying any normalization,
        then uses these raw statistics in the legend labels to provide context about the actual scale and
        distribution of each metric.

        :param metrics_list: List of metrics to plot (default: all available metrics)
        :param norm: Normalization method for plotting ("zscore", "range", or "none")
        :param palette: Color palette for the plot
        :param line_width: Width of plotted lines
        :param style: Seaborn style for the plot
        :param title: Plot title
        :param xlabel: X-axis label
        :param ylabel: Y-axis label
        :param grid: Whether to show grid lines
        :param save_as: File path to save the plot
        :param dpi: DPI for saved image
        :param figsize: Figure size in inches
        :param show: Whether to display the figure (default: True)

        :return: The figure object
        """
        # Store original scores before normalization
        original_scores = self.scores.copy()

        # Calculate statistics from original data
        original_stats = {}
        for metric in metrics_list or self.scores.keys():
            values = np.array(original_scores[metric])
            original_stats[metric] = {
                'mean': np.nanmean(values),
                'std': np.nanstd(values)
            }

        # Apply normalization if requested
        if norm != "none":
            self._normalize_scores(method=norm)

        # Set up plot
        sns.set(style=style)
        palette_colors = sns.color_palette(palette, len(metrics_list) if metrics_list else len(self.scores))
        plt.figure(figsize=figsize)

        if metrics_list is None:
            metrics_list = self.scores.keys()

        # Plot each metric
        for idx, metric in enumerate(metrics_list):
            values = np.array(self.scores[metric])

            # Use original statistics for legend
            mean_val = original_stats[metric]['mean']
            std_val = original_stats[metric]['std']
            legend_label = f"{metric} ({mean_val:.2f} / {std_val:.2f})"

            plt.plot(
                range(2, self.max_clusters + 1),
                values,
                label=legend_label,
                color=palette_colors[idx],
                linewidth=line_width,
            )

        # Set title and labels
        if title is None:
            title = "Cluster Quality Metrics"

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        # Configure ticks and legend
        plt.xticks(ticks=range(2, self.max_clusters + 1), fontsize=10)
        plt.yticks(fontsize=10)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.legend(title="Metrics (Raw Mean / Std Dev)", fontsize=10, title_fontsize=12)

        # Add a note about normalization
        norm_note = f"Note: Lines show {norm} normalized values; legend shows raw statistics"
        plt.figtext(0.5, 0.01, norm_note, ha='center', fontsize=10, style='italic')

        # Configure grid
        if grid:
            plt.grid(True, linestyle="--", alpha=0.7)
        else:
            plt.grid(False)

        # Adjust layout to make room for the note
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        # Save and show the plot
        return save_and_show_results(save_as, dpi, show=show)


class ClusterResults:
    def __init__(self, cluster):
        """
        Initialize the ClusterResults class.

        :param cluster: An instance of the Cluster class.
        """
        if not isinstance(cluster, Cluster):
            raise ValueError("Input must be an instance of the Cluster class.")

        self.linkage_matrix = cluster.linkage_matrix
        self.entity_ids = cluster.entity_ids  # Retrieve entity IDs from Cluster class

    def get_cluster_memberships(self, num_clusters) -> pd.DataFrame:
        """
        Generate a table mapping entity IDs to their corresponding cluster IDs.
        Based on this table, users later can link this to the original dataframe for further regression models.

        There is a common point of confusion because
        k is typically used to represent the number of clusters in clustering algorithms (e.g., k-means).
        However, SciPy's hierarchical clustering API specifically uses t as the parameter name.

        :param num_clusters: The number of clusters to create.
        :return: Pandas DataFrame with entity IDs and cluster memberships.
        """
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix is not computed.")

        # Generate cluster labels
        cluster_labels = fcluster(self.linkage_matrix, t=num_clusters, criterion="maxclust")
        return pd.DataFrame({"Entity ID": self.entity_ids, "Cluster ID": cluster_labels})

    def get_cluster_distribution(self, num_clusters) -> pd.DataFrame:
        """
        Generate a distribution summary of clusters showing counts and percentages.

        This function calculates how many entities belong to each cluster and what
        percentage of the total they represent.

        :param num_clusters: The number of clusters to create.
        :return: DataFrame with cluster distribution information.
        """
        # Get cluster memberships
        memberships_df = self.get_cluster_memberships(num_clusters)

        # Count entities in each cluster
        cluster_counts = memberships_df['Cluster ID'].value_counts().sort_index()

        # Calculate percentages
        total_entities = len(memberships_df)
        cluster_percentages = (cluster_counts / total_entities * 100).round(2)

        # Create distribution dataframe
        distribution = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Count': cluster_counts.values,
            'Percentage': cluster_percentages.values
        }).sort_values('Cluster')

        return distribution

    def plot_cluster_distribution(self, num_clusters, save_as=None, title=None,
                                  style="whitegrid", dpi=200, figsize=(10, 6)):
        """
        Plot the distribution of entities across clusters as a bar chart.

        This visualization shows how many entities belong to each cluster, providing
        insight into the balance and size distribution of the clustering result.

        :param num_clusters: The number of clusters to create.
        :param save_as: File path to save the plot. If None, the plot will be shown.
        :param title: Title for the plot. If None, a default title will be used.
        :param style: Seaborn style for the plot.
        :param dpi: DPI for saved image.
        :param figsize: Figure size in inches.
        """
        # Get cluster distribution data
        distribution = self.get_cluster_distribution(num_clusters)

        # Set up plot
        sns.set(style=style)
        plt.figure(figsize=figsize)

        # Create bar plot with a more poetic, fresh color palette
        # 'muted', 'pastel', and 'husl' are good options for fresher colors
        ax = sns.barplot(x='Cluster', y='Count', data=distribution, palette='pastel')

        # Set the Y-axis range to prevent text overflow
        ax.set_ylim(0, distribution['Count'].max() * 1.2)

        # Ensure Y-axis uses integer ticks
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add percentage labels on top of bars
        for p, (_, row) in zip(ax.patches, distribution.iterrows()):
            height = p.get_height()
            percentage = row['Percentage']
            ax.text(p.get_x() + p.get_width() / 2., height + 0.5,
                    f'{percentage:.1f}%', ha="center", fontsize=9)

        # Set a simple label for entity count at the top
        if title is None:
            title = f"N = {len(self.entity_ids)}"

        # Use a lighter, non-bold title style
        plt.title(title, fontsize=12, fontweight="normal", loc='right')

        plt.xlabel("Cluster ID", fontsize=12)
        plt.ylabel("Number of Entities", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Ensure integer ticks for cluster IDs
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Add grid for better readability but make it lighter
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Adjust layout
        plt.tight_layout()

        # Adjust layout to make room for the note
        plt.subplots_adjust(bottom=0.13)

        # Add a note about normalization
        norm_note = f"Note: Y-axis shows entity counts; percentages above bars indicate their relative frequency."
        plt.figtext(0.5, 0.01, norm_note, ha='center', fontsize=10, style='italic')

        # Save and show the plot
        from sequenzo.visualization.utils import save_and_show_results
        save_and_show_results(save_as, dpi)




