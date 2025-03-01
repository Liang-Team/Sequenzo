"""
@Author  : 梁彧祺
@File    : 241220_hierarchical_clustering.py
@Time    : 18/12/2024 17:59
@Desc    :

    fastcluster 是专为提升大规模层次聚类效率设计的工具，可以处理百万级的数据矩阵。fastcluster 的链接矩阵计算效率更高。

    This module provides a flexible and user-friendly implementation of hierarchical clustering,
    along with tools to evaluate cluster quality and analyze clustering results.

    It supports common hierarchical clustering methods and evaluation metrics,
    designed for social sequence analysis and other research applications.

    The python_source_code has three main components:
    1. Cluster Class: Performs hierarchical clustering on a precomputed distance matrix.
    2. ClusterQuality Class: Evaluates the quality of clustering for different numbers of clusters using various metrics.
    3. ClusterResults Class: Analyzes and visualizes the clustering results (e.g., membership tables and cluster distributions).
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from fastcluster import linkage  # 使用 fastcluster 替代 scipy.cluster.hierarchy.linkage
from joblib import Parallel, delayed


class Cluster:
    def __init__(self, matrix, entity_ids, clustering_method="ward", n_jobs=-1):
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
            print("Converting DataFrame to NumPy array...")
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
        print(f"Computing linkage matrix using fastcluster with {self.clustering_method} method...")
        self.linkage_matrix = self._compute_linkage()

    def _compute_linkage(self):
        """
        Compute the linkage matrix using fastcluster for improved performance.
        """
        # Convert the distance matrix to condensed form for linkage
        condensed_matrix = squareform(self.full_matrix)

        # Use fastcluster's linkage function
        return linkage(condensed_matrix, method=self.clustering_method)

    def plot_dendrogram(self, save_as=None, style="whitegrid", title="Dendrogram",
                        xlabel="Entities", ylabel="Distance", grid=False, dpi=200,
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

        if save_as:
            # Ensure the filename has an extension
            if not any(save_as.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
                save_as = f"{save_as}.png"  # Add default .png extension

            plt.savefig(save_as, dpi=dpi, bbox_inches='tight')

        plt.show()
        plt.close()  # Release resources

    def get_cluster_labels(self, num_clusters):
        """
        Get cluster labels for a specified number of clusters.

        :param num_clusters: The number of clusters to create.
        :return: Array of cluster labels corresponding to entity_ids.
        """
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix is not computed.")

        return fcluster(self.linkage_matrix, t=num_clusters, criterion='maxclust')


class ClusterQuality:
    def __init__(self, matrix_or_cluster, max_clusters=20, clustering_method=None):
        """
        Initialize the ClusterQuality class for precomputed distance matrices
        or a Cluster instance.

        allow the ClusterQuality class to directly accept a Cluster instance
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
        elif isinstance(matrix_or_cluster, (np.ndarray, pd.DataFrame)):
            # Handle direct matrix input
            if isinstance(matrix_or_cluster, pd.DataFrame):
                print("Detected Pandas DataFrame. Converting to NumPy array...")
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
        self.linkage_matrix = None

    def compute_cluster_quality_scores(self):
        """
        Compute clustering quality scores for different numbers of clusters.
        """
        self.linkage_matrix = linkage(squareform(self.matrix), method=self.clustering_method)

        for k in range(2, self.max_clusters + 1):
            labels = fcluster(self.linkage_matrix, k, criterion="maxclust")
            self.scores["ASW"].append(self._compute_silhouette(labels))
            self.scores["ASWw"].append(self._compute_weighted_silhouette(labels))
            self.scores["HG"].append(self._compute_homogeneity(labels))
            self.scores["PBC"].append(self._compute_point_biserial(labels))
            self.scores["CH"].append(self._compute_calinski_harabasz(labels))
            self.scores["R2"].append(self._compute_r2(labels))
            self.scores["HC"].append(self._compute_hierarchical_criterion(labels))

    def _compute_silhouette(self, labels):
        """
        Compute Silhouette Score (ASW).
        """
        if len(set(labels)) > 1:
            return silhouette_score(self.matrix, labels, metric="precomputed")
        return np.nan

    def _compute_weighted_silhouette(self, labels):
        """
        Compute Weighted Silhouette Score (ASWw).
        """
        cluster_sizes = np.bincount(labels)[1:]
        total_points = len(labels)
        weights = cluster_sizes / total_points
        silhouette_scores = [silhouette_score(self.matrix, labels, metric="precomputed")]
        return np.sum(weights * silhouette_scores)

    def _compute_homogeneity(self, labels):
        """
        Compute Homogeneity (HG).
        """
        cluster_sizes = np.bincount(labels)[1:]
        total_points = len(labels)
        return np.sum((cluster_sizes / total_points) ** 2)

    def _compute_point_biserial(self, labels):
        """
        Compute Point-Biserial Correlation (PBC).
        """
        # Ensure distances is in full matrix form for proper indexing
        distances = self.matrix  # Use the full square-form matrix (already ensured during initialization)
        within = []
        between = []
        for cluster in np.unique(labels):
            indices = np.where(labels == cluster)[0]
            others = np.where(labels != cluster)[0]
            # Collect distances within and between clusters
            within.extend(distances[np.ix_(indices, indices)].ravel())
            between.extend(distances[np.ix_(indices, others)].ravel())

        within = np.array(within)
        between = np.array(between)

        if len(within) == 0 or len(between) == 0:
            return np.nan  # Avoid division by zero or invalid PBC calculation

        # Calculate PBC
        return (between.mean() - within.mean()) / np.std(np.concatenate([within, between]))

    def _compute_calinski_harabasz(self, labels):
        """
        Compute Calinski-Harabasz Index (CH).
        """
        n_samples = len(labels)
        n_clusters = len(np.unique(labels))
        within_var = np.sum([np.var(self.matrix[labels == cluster]) for cluster in np.unique(labels)])
        total_var = np.var(self.matrix)
        return (total_var - within_var) * (n_samples - n_clusters) / (within_var * (n_clusters - 1))

    def _compute_r2(self, labels):
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

    def _compute_hierarchical_criterion(self, labels):
        """
        Compute Hierarchical Criterion (HC).
        """
        return np.var([np.mean(self.matrix[labels == cluster]) for cluster in np.unique(labels)])

    def _normalize_scores(self, method="zscore"):
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

        for metric, values in original_scores.items():
            values = np.array(values)
            optimal_k = np.nanargmax(values) + 2  # Adding 2 because k starts at 2
            max_value = np.nanmax(values)

            # Add data to the summary table
            summary["Metric"].append(metric)
            summary["Opt. Clusters"].append(optimal_k)
            summary["Opt. Value"].append(max_value)
            summary["Z-Score Norm."].append(zscore_normalized[metric][optimal_k - 2])
            summary["Min-Max Norm."].append(minmax_normalized[metric][optimal_k - 2])

        return pd.DataFrame(summary)

    def plot_combined_scores(
                                self,
                                metrics_list=None,
                                norm="none",
                                palette="husl",
                                line_width=2,
                                style="whitegrid",
                                title=None,
                                xlabel="Number of Clusters",
                                ylabel="Normalized Score",
                                grid=True,
                                save_as=None,
                                dpi=200,
                                figsize=(12, 8)
                            ):
        """
        Plot combined scores for clustering quality metrics with customizable parameters.
        :param dpi: Dots per inch for the saved image (default: 300 for high resolution).
        :param figsize: Tuple specifying the figure size in inches (default: (12, 8)).
        """
        if norm != "none":
            self._normalize_scores(method=norm)

        sns.set(style=style)
        palette_colors = sns.color_palette(palette, len(metrics_list) if metrics_list else len(self.scores))
        plt.figure(figsize=figsize)  # Set the figure size

        if metrics_list is None:
            metrics_list = self.scores.keys()

        legend_labels = []  # To store legend labels

        for idx, metric in enumerate(metrics_list):

            values = np.array(self.scores[metric])
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            legend_label = f"{metric} ({mean_val:.2f} / {std_val:.2f})"
            legend_labels.append(legend_label)

            plt.plot(
                range(2, self.max_clusters + 1),
                values,
                label=legend_label,
                color=palette_colors[idx],
                linewidth=line_width,
            )

        if title is None:
            title = f"Cluster Quality Metrics"

        plt.legend(title="Metrics (Mean / Std Dev)", fontsize=10, title_fontsize=12)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(ticks=range(2, self.max_clusters + 1), fontsize=10)
        plt.yticks(fontsize=10)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if title is not None:
            plt.title(title, fontsize=14, fontweight="bold")

        if grid:
            plt.grid(True, linestyle="--", alpha=0.7)
        else:
            plt.grid(False)

        if save_as:
            plt.savefig(save_as, dpi=dpi, bbox_inches='tight')
        plt.show()


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

    def get_cluster_memberships(self, num_clusters):
        """
        Generate a table mapping entity IDs to their corresponding cluster IDs.
        Based on this table, users later can link this to the original dataframe for further regression models.

        :param num_clusters: The number of clusters to create.
        :return: Pandas DataFrame with entity IDs and cluster memberships.
        """
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix is not computed.")

        # Generate cluster labels
        cluster_labels = fcluster(self.linkage_matrix, k=num_clusters, criterion="maxclust")
        return pd.DataFrame({"Entity ID": self.entity_ids, "Cluster ID": cluster_labels})



if __name__ == '__main__':
    # Import necessary libraries
    from sequenzo import *  # Social sequence analysis
    import pandas as pd  # Data manipulation

    # List all the available datasets in Sequenzo
    print('Available datasets in Sequenzo: ', list_datasets())

    # Load the data that we would like to explore in this tutorial
    # `df` is the short for `dataframe`, which is a common variable name for a dataset
    df = load_dataset('country_co2_emissions')

    # Create a SequenceData object from the dataset

    # Define the time-span variable
    time = list(df.columns)[1:]

    states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

    sequence_data = SequenceData(df, time=time, time_type="year", id_col="country", states=states)

    om = get_distance_matrix(seqdata=sequence_data,
                             method='OM',
                             sm="TRATE",
                             indel="auto")

    from sequenzo.clustering import Cluster

    Cluster(om, sequence_data.ids, clustering_method='ward').plot_dendrogram(xlabel="Countries", ylabel="Distance", save_as='dendrogram')

