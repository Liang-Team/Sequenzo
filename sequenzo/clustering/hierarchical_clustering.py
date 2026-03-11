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

    WEIGHTED CLUSTERING SUPPORT:
    All classes now support weighted data analysis:
    - Cluster: Hierarchical linkage is computed on the given distance matrix (unweighted). Optional weights are applied to evaluation and summaries
    - ClusterQuality: Computes weighted versions of quality metrics (ASWw, HG, R2, HC)
    - ClusterResults: Provides weighted cluster distribution statistics and visualizations

    Weighted metrics account for sequence importance when calculating clustering quality,
    making the analysis more representative when sequences have different sampling weights
    or population sizes.

    WARD METHOD VARIANTS:
    The module supports two Ward linkage variants:
    - 'ward_d' (Ward D): Classic Ward method using squared Euclidean distances ÷ 2
    - 'ward_d2' (Ward D2): Ward method using squared Euclidean distances
    For backward compatibility, 'ward' maps to 'ward_d'.
    
    The difference affects clustering results and dendrogram heights:
    - Ward D produces smaller distances in the linkage matrix
    - Ward D2 produces distances equal to the increase in cluster variance
    - Both methods produce identical cluster assignments, only distances differ

    ROBUSTNESS AND VALIDATION FEATURES:
    - Ward Method Validation: Automatic detection of non-Euclidean distance matrices
    - One-time Warning System: Alerts users when Ward methods are used with potentially incompatible distances
    - Robust Matrix Cleanup: Handles NaN/Inf values using 95th percentile replacement
    - Distance Matrix Validation: Ensures zero diagonal and non-negativity
    - Symmetry Handling: Automatically symmetrizes matrices when required by clustering algorithms
    - Method Recommendations: Suggests alternative methods for sequence distances

    For sequence distances (OM, LCS, etc.), Ward linkage methods may produce suboptimal results.
    Consider using alternative methods like 'average' (UPGMA) for better theoretical validity.

    Original code references:
        Cluster(): Derived from `hclust`, a key function from fastcluster
            R code: https://github.com/cran/fastcluster/blob/master/R/fastcluster.R
            Python code: https://github.com/fastcluster/fastcluster/blob/master/src/fastcluster.cpp
            The Python version of facluster does not support Ward D method but only Ward D2, whereas R supports both.
            Thus, we provide Ward D by ourselves here. 

        ClusterQuality(): Derived from ``, a key function from weightedcluster
            CQI equivalence of R is here (two files):
                https://github.com/cran/WeightedCluster/blob/master/src/clusterquality.cpp
                https://github.com/cran/WeightedCluster/blob/master/src/clusterqualitybody.cpp
            plot_cqi_scores(): `wcCmpCluster()` produces `clustrangefamily` object + `plot.clustrangefamily()` for plotting
"""
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.ticker import MaxNLocator

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster, dendrogram
from scipy.spatial.distance import squareform, pdist
# sklearn metrics no longer needed - using C++ implementation
# Import from sequenzo_fastcluster (our custom fastcluster with ward_d and ward_d2 support)
try:
    from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage, linkage_vector
except ImportError:
    # Fallback: try absolute import
    try:
        from sequenzo_fastcluster.fastcluster import linkage, linkage_vector
    except ImportError:
        # Last resort: try relative import
        from .sequenzo_fastcluster.fastcluster import linkage, linkage_vector

# Import C++ clustering extensions (required for all clustering operations)
try:
    from . import clustering_c_code
    _CPP_AVAILABLE = True
    _CPP_CUTREE_AVAILABLE = hasattr(clustering_c_code, "cutree_maxclust")
    _CPP_EUCLIDEAN_CHECK_AVAILABLE = hasattr(clustering_c_code, "check_euclidean_compatibility")
except ImportError:
    _CPP_AVAILABLE = False
    _CPP_CUTREE_AVAILABLE = False
    _CPP_EUCLIDEAN_CHECK_AVAILABLE = False
    print("[!] Warning: C++ clustering extensions not available. Cluster class will not work.")


# Corrected imports: Use relative imports *within* the package.
from sequenzo.visualization.utils import save_and_show_results

# Global flag to ensure Ward warning is only shown once per session
_WARD_WARNING_SHOWN = False
_WARN_NONFINITE = 1 << 0
_WARN_NEGATIVE = 1 << 1
_WARN_SYMMETRIZED = 1 << 2
_WARN_WARD_NON_EUCLIDEAN = 1 << 3


def _cutree_maxclust(linkage_matrix, num_clusters):
    """
    Cut linkage tree into `num_clusters` flat clusters (1-based labels).
    Prefer C++ implementation for speed and fall back to SciPy if unavailable.
    """
    if _CPP_AVAILABLE and _CPP_CUTREE_AVAILABLE:
        n = linkage_matrix.shape[0] + 1
        labels = clustering_c_code.cutree_maxclust(
            np.asarray(linkage_matrix, dtype=np.float64, order="C"),
            int(n),
            int(num_clusters),
        )
        return np.asarray(labels, dtype=np.int32)

    return fcluster(linkage_matrix, t=num_clusters, criterion="maxclust")


def _check_euclidean_compatibility(matrix, method):
    """
    Check if a distance matrix is likely compatible with Euclidean-based methods like Ward.
    Uses C++ implementation exclusively.
    """
    if method.lower() not in ["ward", "ward_d", "ward_d2"]:
        return True

    if not (_CPP_AVAILABLE and _CPP_EUCLIDEAN_CHECK_AVAILABLE):
        raise RuntimeError(
            "C++ check_euclidean_compatibility is not available. "
            "Please ensure the C++ extensions are properly compiled.")

    result = clustering_c_code.check_euclidean_compatibility(
        np.asarray(matrix, dtype=np.float64, order="C"),
        method.lower(),
    )
    return bool(result.get("compatible", True))


def _warn_ward_usage_once(matrix, method, euclidean_compatible=None, warning_flags=None):
    """
    Issue a one-time warning about using Ward with potentially non-Euclidean distances.
    """
    global _WARD_WARNING_SHOWN
    
    # Check for both Ward D and Ward D2 methods
    if not _WARD_WARNING_SHOWN and method.lower() in ["ward", "ward_d", "ward_d2"]:
        if warning_flags is not None:
            is_compatible = (int(warning_flags) & _WARN_WARD_NON_EUCLIDEAN) == 0
        else:
            is_compatible = (
                _check_euclidean_compatibility(matrix, method)
                if euclidean_compatible is None
                else bool(euclidean_compatible)
            )
        if not is_compatible:
            warnings.warn(
                "\n[!] Ward linkage method detected with potentially non-Euclidean distance matrix!\n"
                "   Ward clustering (both Ward D and Ward D2) assumes Euclidean distances for theoretical validity.\n"
                "   \n"
                "   Ward method variants:\n"
                "   - 'ward_d' (classic): Uses squared Euclidean distances ÷ 2\n"
                "   - 'ward_d2': Uses squared Euclidean distances\n"
                "   \n"
                "   For sequence distances (OM, LCS, etc.), consider using:\n"
                "   - method='average' (UPGMA)\n"
                "   - method='complete' (complete linkage)\n"
                "   - method='single' (single linkage)\n"
                "   \n"
                "   Note: 'centroid' and 'median' methods may also produce inversions\n"
                "   (non-monotonic dendrograms) with non-Euclidean distances.\n"
                "   \n"
                "   This warning is shown only once per session.",
                UserWarning,
                stacklevel=3
            )
        _WARD_WARNING_SHOWN = True


class Cluster:
    def __init__(self,
                 matrix=None,
                 entity_ids=None,
                 clustering_method="ward",
                 weights=None,
                 X_features=None,
                 fast_path=False):
        """
        Hierarchical clustering with the full computational pipeline in C++.

        :param matrix: Precomputed distance matrix (full square form). Required when X_features is None.
        :param entity_ids: List of IDs corresponding to the entities in the matrix.
        :param clustering_method: Clustering algorithm to use. Options include:
            - "ward" or "ward_d": Classic Ward method (squared Euclidean distances ÷ 2) [default]
            - "ward_d2": Ward method with squared Euclidean distances
            - "single": Single linkage (minimum method)
            - "complete": Complete linkage (maximum method)
            - "average": Average linkage (UPGMA)
            - "centroid": Centroid linkage
            - "median": Median linkage
        :param weights: Optional array of weights for each entity (default: None for equal weights).
        :param X_features: Optional (n x d) feature matrix for Euclidean Ward clustering. When provided
            with ward/ward_d/ward_d2, uses memory-efficient linkage_vector (O(ND) vs O(N²)).
        :param fast_path: If True, skips Ward compatibility checking and full_matrix retention.
        """
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ clustering core is not available. "
                "Please ensure the C++ extensions are properly compiled.")

        # Users may pass a DataFrame, a `float32` array, a Fortran-order array, or even a Python list.
        # Converting all of these to a C-contiguous `float64` `ndarray` is a necessary prerequisite
        # before invoking the C++ code.
        method = clustering_method.lower()

        ward_methods = ("ward", "ward_d", "ward_d2")
        use_vector_path = (X_features is not None and method in ward_methods)

        if use_vector_path:
            X = np.asarray(X_features, dtype=np.float64, order="C")
            if X.ndim != 2:
                raise ValueError("X_features must be a 2D array (n x d).")
            n = X.shape[0]
        else:
            if matrix is None:
                raise ValueError("Either matrix or X_features (with ward/ward_d2) must be provided.")
            if isinstance(matrix, pd.DataFrame):
                matrix = matrix.values
            matrix = np.asarray(matrix, dtype=np.float64, order="C")
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input must be a full square-form distance matrix.")
            n = matrix.shape[0]

        # `entity_ids` and `weights` are not involved in the linkage computation;
        # they are used later by `get_cluster_labels()`, `ClusterQuality`, and `plot_dendrogram()`.
        # Passing them into C++ and then returning them would only introduce an unnecessary data copy (Python → C++ → Python)
        # without any computational benefit.

        self.entity_ids = np.array(entity_ids) if entity_ids is not None else np.arange(n)
        if len(self.entity_ids) != n:
            raise ValueError("Length of entity_ids must match the number of data points.")
        if len(np.unique(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids must contain unique values.")

        if weights is not None:
            self.weights = np.array(weights, dtype=np.float64)
            if len(self.weights) != n:
                raise ValueError("Length of weights must match the number of data points.")
            if np.any(self.weights < 0) or np.sum(self.weights) == 0:
                raise ValueError("All weights must be non-negative and sum > 0.")
        else:
            self.weights = np.ones(n, dtype=np.float64)

        # Call C++ core
        if use_vector_path:
            result = clustering_c_code.cluster_from_features(X, method)
            self._X_features = X
        else:
            result = clustering_c_code.cluster_from_matrix(
                matrix, method, bool(fast_path))
            self._X_features = None

        # Store results
        self.clustering_method = "ward_d" if method == "ward" else method
        self.fast_path = bool(fast_path)

        lm = result["linkage_matrix"]
        self.linkage_matrix = np.asarray(lm) if lm is not None else None

        cm = result["condensed_matrix"]
        self.condensed_matrix = np.asarray(cm) if cm is not None else None

        fm = result["full_matrix"]
        self._full_matrix = np.asarray(fm) if fm is not None else None

        # Emit Python-side warnings from C++ flags
        self._warn_from_flags(int(result["warning_flags"]), self.clustering_method)

    @property
    def full_matrix(self):
        """Full distance matrix. Lazy-computed from X_features or condensed_matrix."""
        if self._full_matrix is None and self._X_features is not None:
            self._full_matrix = squareform(pdist(self._X_features, "euclidean"))
        elif self._full_matrix is None and self.condensed_matrix is not None:
            self._full_matrix = squareform(self.condensed_matrix)
        return self._full_matrix

    @full_matrix.setter
    def full_matrix(self, value):
        self._full_matrix = value

    def _warn_from_flags(self, flags, method):
        """Translate C++ warning_flags bitmask into Python warnings."""
        if flags & _WARN_SYMMETRIZED:
            print("[!] Warning: Distance matrix is not symmetric.")
            print("    Automatically symmetrized using (matrix + matrix.T) / 2")
        if flags & _WARN_NONFINITE:
            print("[!] Warning: Distance matrix contained NaN or Inf values (replaced).")
        if flags & _WARN_NEGATIVE:
            print("[!] Warning: Distance matrix contained negative values (clipped to zero).")
        if flags & _WARN_WARD_NON_EUCLIDEAN:
            _warn_ward_usage_once(
                self.full_matrix, method,
                euclidean_compatible=False,
                warning_flags=flags)

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
        return _cutree_maxclust(self.linkage_matrix, num_clusters)


def ward_labels_only(
    num_clusters,
    matrix=None,
    X_features=None,
    entity_ids=None,
    ward_variant="ward_d2",
    fast_path=True,
    early_stop=False,
):
    """
    Labels-only API for Ward clustering.

    Default path uses Sequenzo's Cluster with optional fast_path.
    When early_stop=True and X_features is provided, uses sklearn's
    AgglomerativeClustering(compute_full_tree=False) for early stopping.
    """
    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1")

    if early_stop:
        if X_features is None:
            raise ValueError("early_stop=True requires X_features input.")
        from sklearn.cluster import AgglomerativeClustering

        X = np.asarray(X_features, dtype=np.float64)
        ac = AgglomerativeClustering(
            n_clusters=int(num_clusters),
            linkage="ward",
            metric="euclidean",
            compute_full_tree=False,
        )
        labels = ac.fit_predict(X) + 1  # Keep Sequenzo's 1-based label convention.
        return labels.astype(np.int32)

    if X_features is None and matrix is None:
        raise ValueError("Provide either matrix or X_features.")

    if X_features is not None:
        n = np.asarray(X_features).shape[0]
    else:
        n = np.asarray(matrix).shape[0]
    if entity_ids is None:
        entity_ids = np.arange(n)

    cluster = Cluster(
        matrix=matrix,
        entity_ids=entity_ids,
        clustering_method=ward_variant,
        X_features=X_features,
        fast_path=bool(fast_path),
    )
    return cluster.get_cluster_labels(num_clusters)


class ClusterQuality:
    def __init__(self, matrix_or_cluster, max_clusters=20, clustering_method=None, weights=None):
        """
        Initialize the ClusterQuality class for precomputed distance matrices or a Cluster instance.

        All heavy computation (linkage, CQI scores, summary, range table) is performed
        in a single C++ call. Python only handles parameter validation and result unpacking.

        :param matrix_or_cluster: The precomputed distance matrix (full square form)
                                   or an instance of the Cluster class.
        :param max_clusters: Maximum number of clusters to evaluate (default: 20).
        :param clustering_method: Clustering algorithm to use. If None, inherit from Cluster instance.
        :param weights: Optional array of weights for each entity. If None and using Cluster instance,
                       weights will be extracted from the Cluster object.
        """
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ clustering core is not available. "
                "Please ensure the C++ extensions are properly compiled.")

        self.metric_order = [
            "PBC", "HG", "HGSD", "ASW", "ASWw",
            "CH", "R2", "CHsq", "R2sq", "HC",
        ]

        if isinstance(matrix_or_cluster, Cluster):
            cluster = matrix_or_cluster
            self.matrix = cluster.full_matrix
            self.clustering_method = cluster.clustering_method
            self.linkage_matrix = cluster.linkage_matrix
            self.weights = cluster.weights
            self._condensed_matrix = cluster.condensed_matrix

            n = len(cluster.entity_ids)
            k_max = min(int(max_clusters), n)
            if k_max < 2:
                raise ValueError("max_clusters must be at least 2 and no greater than the sample size.")

            result = clustering_c_code.cluster_quality_from_cluster_data(
                np.asarray(self._condensed_matrix, dtype=np.float64, order="C"),
                np.asarray(self.linkage_matrix, dtype=np.float64, order="C"),
                np.asarray(self.weights, dtype=np.float64, order="C"),
                n, 2, k_max,
            )

        elif isinstance(matrix_or_cluster, (np.ndarray, pd.DataFrame)):
            if isinstance(matrix_or_cluster, pd.DataFrame):
                print("[>] Detected Pandas DataFrame. Converting to NumPy array...")
                matrix_or_cluster = matrix_or_cluster.values
            matrix = np.asarray(matrix_or_cluster, dtype=np.float64, order="C")
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Matrix must be a full square-form distance matrix.")

            n = matrix.shape[0]
            method = (clustering_method or "ward_d").lower()
            if method == "ward":
                method = "ward_d"
            k_max = min(int(max_clusters), n)
            if k_max < 2:
                raise ValueError("max_clusters must be at least 2 and no greater than the sample size.")

            if weights is not None:
                w = np.asarray(weights, dtype=np.float64, order="C")
                if len(w) != n:
                    raise ValueError("Length of weights must match the size of the matrix.")
            else:
                w = np.ones(n, dtype=np.float64)

            result = clustering_c_code.cluster_quality_from_matrix(
                matrix, method, k_max, w,
            )

            self.linkage_matrix = np.asarray(result["linkage_matrix"])
            self._condensed_matrix = np.asarray(result["condensed_matrix"])
            self.matrix = np.asarray(result["full_matrix"])
            self.clustering_method = result["clustering_method"]
            self.weights = w

            self._warn_from_flags(int(result["warning_flags"]), self.clustering_method)
        else:
            raise ValueError(
                "Input must be a Cluster instance, a NumPy array, or a Pandas DataFrame."
            )

        self.max_clusters = k_max
        self._unpack_cqi_result(result)

    def _unpack_cqi_result(self, result):
        """Unpack the C++ pipeline result dict into instance attributes."""
        self.scores = {}
        for metric in self.metric_order:
            self.scores[metric] = np.asarray(result[metric], dtype=np.float64).tolist()

        self.original_scores = {}
        for metric in self.metric_order:
            self.original_scores[metric] = np.asarray(result[metric], dtype=np.float64).copy()

        self._summary_opt = np.asarray(result["opt_clusters"], dtype=np.float64)
        self._summary_raw = np.asarray(result["raw_values"], dtype=np.float64)
        self._summary_z = np.asarray(result["z_scores"], dtype=np.float64)
        self._range_table_values = np.asarray(result["range_table"], dtype=np.float64)

    @staticmethod
    def _warn_from_flags(flags, method):
        """Translate C++ warning_flags bitmask into Python warnings."""
        if flags & _WARN_SYMMETRIZED:
            print("[!] Warning: Distance matrix is not symmetric.")
            print("    Automatically symmetrized using (matrix + matrix.T) / 2")
        if flags & _WARN_NONFINITE:
            print("[!] Warning: Distance matrix contained NaN or Inf values (replaced).")
        if flags & _WARN_NEGATIVE:
            print("[!] Warning: Distance matrix contained negative values (clipped to zero).")
        if flags & _WARN_WARD_NON_EUCLIDEAN:
            _warn_ward_usage_once(None, method, euclidean_compatible=False, warning_flags=flags)

    def compute_cluster_quality_scores(self):
        """
        Kept for API compatibility. Scores are already computed in __init__.
        Calling this is a no-op.
        """
        pass

    def _normalize_scores(self, method="zscore", scores=None):
        """
        Normalize each metric independently without mutating source scores.

        :param method: Normalization method. Options are "zscore" or "range".
        :param scores: Optional metric dict to normalize. Defaults to self.scores.
        :return: New dict of normalized numpy arrays.
        """
        source_scores = self.scores if scores is None else scores
        normalized_scores = {}
        for metric in source_scores:
            values = np.asarray(source_scores[metric], dtype=np.float64)
            if method == "zscore":
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val > 0:
                    normalized_scores[metric] = (values - mean_val) / std_val
                else:
                    normalized_scores[metric] = values.copy()
            elif method == "range":
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                if max_val > min_val:
                    normalized_scores[metric] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_scores[metric] = values.copy()
            else:
                normalized_scores[metric] = values.copy()
        return normalized_scores

    def get_cluster_range_table(self) -> pd.DataFrame:
        """
        Return a metrics-by-cluster table mirroring R's `as.clustrange()` output.

        :return: DataFrame indexed by cluster count ("cluster2", ...)
                 with raw metric values for each quality indicator.
        """
        values = self._range_table_values
        n_rows = values.shape[0]
        index_labels = [f"cluster{k}" for k in range(2, 2 + n_rows)]
        table = pd.DataFrame(values, index=index_labels, columns=self.metric_order)
        table.index.name = "Cluster"
        return table

    def get_cqi_table(self):
        """
        Generate a summary table of clustering quality indicators with concise column names.

        :return: Pandas DataFrame summarizing the optimal number of clusters (N groups),
                 the corresponding raw metric values, and z-score normalized values.
        """
        return pd.DataFrame({
            "Metric": self.metric_order,
            "Opt. Clusters": self._summary_opt,
            "Raw Value": self._summary_raw,
            "Z-Score Norm.": self._summary_z,
        })

    def plot_cqi_scores(self,
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
        Plot combined scores for clustering quality indicators with customizable parameters.

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
        original_scores = {
            metric: np.asarray(values, dtype=np.float64).copy()
            for metric, values in self.scores.items()
        }

        original_stats = {}
        for metric in metrics_list or self.metric_order:
            values = np.array(original_scores[metric])
            original_stats[metric] = {
                'mean': np.nanmean(values),
                'std': np.nanstd(values)
            }

        if norm == "none":
            plot_scores = original_scores
        else:
            plot_scores = self._normalize_scores(method=norm, scores=original_scores)

        sns.set(style=style)
        palette_colors = sns.color_palette(palette, len(metrics_list) if metrics_list else len(plot_scores))
        plt.figure(figsize=figsize)

        if metrics_list is None:
            metrics_list = list(self.metric_order)
        else:
            metrics_list = [metric for metric in metrics_list if metric in self.metric_order]

        for idx, metric in enumerate(metrics_list):
            values = np.asarray(plot_scores[metric], dtype=np.float64)
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

        if title is None:
            title = "Cluster Quality Metrics"

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        plt.xticks(ticks=range(2, self.max_clusters + 1), fontsize=10)
        plt.yticks(fontsize=10)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(title="Metrics (Raw Mean / Std Dev)", fontsize=10, title_fontsize=12)

        norm_note = f"Note: Lines show {norm} normalized values; legend shows raw statistics"
        plt.figtext(0.5, 0.01, norm_note, ha='center', fontsize=10, style='italic')

        if grid:
            plt.grid(True, linestyle="--", alpha=0.7)
        else:
            plt.grid(False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        return save_and_show_results(save_as, dpi, show=show)


class ClusterResults:
    def __init__(self, cluster):
        """
        Initialize the ClusterResults class.

        :param cluster: An instance of the Cluster class.
        """
        if not isinstance(cluster, Cluster):
            raise ValueError("Input must be an instance of the Cluster class.")
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ clustering core is not available. "
                "Please ensure the C++ extensions are properly compiled.")

        self.linkage_matrix = cluster.linkage_matrix
        self.entity_ids = cluster.entity_ids
        self.weights = cluster.weights
        self._results_cache = {}

    def _get_cached_result(self, num_clusters):
        """Single C++ call for cutree + distribution, with per-k caching."""
        if num_clusters not in self._results_cache:
            self._results_cache[num_clusters] = clustering_c_code.cluster_results_compute(
                np.asarray(self.linkage_matrix, dtype=np.float64, order="C"),
                len(self.entity_ids),
                int(num_clusters),
                np.asarray(self.weights, dtype=np.float64, order="C"),
            )
        return self._results_cache[num_clusters]

    def get_cluster_memberships(self, num_clusters) -> pd.DataFrame:
        """
        Generate a table mapping entity IDs to their corresponding cluster IDs.

        :param num_clusters: The number of clusters to create.
        :return: Pandas DataFrame with entity IDs and cluster memberships.
        """
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix is not computed.")
        result = self._get_cached_result(num_clusters)
        return pd.DataFrame({
            "Entity ID": self.entity_ids,
            "Cluster": np.asarray(result["labels"], dtype=np.int32),
        })

    def get_cluster_distribution(self, num_clusters, weighted=False) -> pd.DataFrame:
        """
        Generate a distribution summary of clusters showing counts, percentages,
        and optionally weighted statistics.

        :param num_clusters: The number of clusters to create.
        :param weighted: If True, include weighted statistics in the distribution.
        :return: DataFrame with cluster distribution information.
        """
        result = self._get_cached_result(num_clusters)
        distribution = pd.DataFrame({
            "Cluster": np.asarray(result["Cluster"], dtype=np.int32),
            "Count": np.asarray(result["Count"], dtype=np.int32),
            "Percentage": np.round(np.asarray(result["Percentage"], dtype=np.float64), 2),
            "Weight_Sum": np.asarray(result["Weight_Sum"], dtype=np.float64),
            "Weight_Percentage": np.round(np.asarray(result["Weight_Percentage"], dtype=np.float64), 2),
        }).sort_values("Cluster")

        if not weighted:
            return distribution[["Cluster", "Count", "Percentage"]]
        return distribution

    def plot_cluster_distribution(self, num_clusters, save_as=None, title=None,
                                  style="whitegrid", dpi=200, figsize=(10, 6), weighted=False):
        """
        Plot the distribution of entities across clusters as a bar chart.

        :param num_clusters: The number of clusters to create.
        :param save_as: File path to save the plot. If None, the plot will be shown.
        :param title: Title for the plot. If None, a default title will be used.
        :param style: Seaborn style for the plot.
        :param dpi: DPI for saved image.
        :param figsize: Figure size in inches.
        :param weighted: If True, display weighted percentages instead of entity count percentages.
        """
        distribution = self.get_cluster_distribution(num_clusters, weighted=weighted)

        sns.set(style=style)
        plt.figure(figsize=figsize)

        if weighted and 'Weight_Sum' in distribution.columns:
            y_column = 'Weight_Sum'
            percentage_column = 'Weight_Percentage'
            ylabel = "Total Weight"
            note_text = "Y-axis shows weight sums; percentages above bars indicate weight-based relative frequency."
        else:
            y_column = 'Count'
            percentage_column = 'Percentage'
            ylabel = "Number of Entities"
            note_text = "Y-axis shows entity counts; percentages above bars indicate their relative frequency."

        ax = sns.barplot(x='Cluster', y=y_column, data=distribution, palette='pastel')
        ax.set_ylim(0, distribution[y_column].max() * 1.2)

        if not weighted:
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        for p, (_, row) in zip(ax.patches, distribution.iterrows()):
            height = p.get_height()
            percentage = row[percentage_column]
            ax.text(p.get_x() + p.get_width() / 2., height + max(height * 0.02, 0.5),
                    f'{percentage:.1f}%', ha="center", fontsize=9)

        if title is None:
            if weighted:
                title = f"N = {len(self.entity_ids)}, Total Weight = {np.sum(self.weights):.1f}"
            else:
                title = f"N = {len(self.entity_ids)}"

        plt.title(title, fontsize=12, fontweight="normal", loc='right')
        plt.xlabel("Cluster ID", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.13)
        plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=10, style='italic')

        save_and_show_results(save_as, dpi)


# For xinyi's test, because she can't debug in Jupyter :
    # Traceback (most recent call last):
    #   File "/Applications/PyCharm.app/Contents/plugins/python-ce/helpers/pydev/_pydevd_bundle/pydevd_comm.py", line 736, in make_thread_stack_str
    #     append('file="%s" line="%s">' % (make_valid_xml_value(my_file), lineno))
    #   File "/Applications/PyCharm.app/Contents/plugins/python-ce/helpers/pydev/_pydevd_bundle/pydevd_xml.py", line 36, in make_valid_xml_value
    #     return s.replace("&", "&amp;").replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    # AttributeError: 'tuple' object has no attribute 'replace'

if __name__ == '__main__':
    # Import necessary libraries
    # Your calling code (e.g., in a script or notebook)

    from sequenzo import *  # Import the package, give it a short alias
    import pandas as pd  # Data manipulation
    import numpy as np

    # List all the available datasets in Sequenzo
    # Now access functions using the alias:
    print('Available datasets in Sequenzo: ', list_datasets())

    # Load the data that we would like to explore in this tutorial
    # `df` is the short for `dataframe`, which is a common variable name for a dataset
    # df = load_dataset('country_co2_emissions')
    # df = load_dataset('mvad')
    df = pd.read_csv("/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/orignal data/mvad.csv")

    # 时间列表
    time_list = ['Jul.93', 'Aug.93', 'Sep.93', 'Oct.93', 'Nov.93', 'Dec.93',
                 'Jan.94', 'Feb.94', 'Mar.94', 'Apr.94', 'May.94', 'Jun.94', 'Jul.94',
                 'Aug.94', 'Sep.94', 'Oct.94', 'Nov.94', 'Dec.94', 'Jan.95', 'Feb.95',
                 'Mar.95', 'Apr.95', 'May.95', 'Jun.95', 'Jul.95', 'Aug.95', 'Sep.95',
                 'Oct.95', 'Nov.95', 'Dec.95', 'Jan.96', 'Feb.96', 'Mar.96', 'Apr.96',
                 'May.96', 'Jun.96', 'Jul.96', 'Aug.96', 'Sep.96', 'Oct.96', 'Nov.96',
                 'Dec.96', 'Jan.97', 'Feb.97', 'Mar.97', 'Apr.97', 'May.97', 'Jun.97',
                 'Jul.97', 'Aug.97', 'Sep.97', 'Oct.97', 'Nov.97', 'Dec.97', 'Jan.98',
                 'Feb.98', 'Mar.98', 'Apr.98', 'May.98', 'Jun.98', 'Jul.98', 'Aug.98',
                 'Sep.98', 'Oct.98', 'Nov.98', 'Dec.98', 'Jan.99', 'Feb.99', 'Mar.99',
                 'Apr.99', 'May.99', 'Jun.99']

    # 方法1: 使用pandas获取所有唯一值
    time_states_df = df[time_list]
    all_unique_states = set()

    for col in time_list:
        unique_vals = df[col].dropna().unique()  # Remove NaN values
        all_unique_states.update(unique_vals)

    # 转换为排序的列表
    states = sorted(list(all_unique_states))
    print("All unique states:")
    for i, state in enumerate(states, 1):
        print(f"{i:2d}. {state}")

    print(f"\nstates list:")
    print(f"states = {states}")

    # Create a SequenceData object

    # Define the time-span variable
    time_list = ['Jul.93', 'Aug.93', 'Sep.93', 'Oct.93', 'Nov.93', 'Dec.93',
                 'Jan.94', 'Feb.94', 'Mar.94', 'Apr.94', 'May.94', 'Jun.94', 'Jul.94',
                 'Aug.94', 'Sep.94', 'Oct.94', 'Nov.94', 'Dec.94', 'Jan.95', 'Feb.95',
                 'Mar.95', 'Apr.95', 'May.95', 'Jun.95', 'Jul.95', 'Aug.95', 'Sep.95',
                 'Oct.95', 'Nov.95', 'Dec.95', 'Jan.96', 'Feb.96', 'Mar.96', 'Apr.96',
                 'May.96', 'Jun.96', 'Jul.96', 'Aug.96', 'Sep.96', 'Oct.96', 'Nov.96',
                 'Dec.96', 'Jan.97', 'Feb.97', 'Mar.97', 'Apr.97', 'May.97', 'Jun.97',
                 'Jul.97', 'Aug.97', 'Sep.97', 'Oct.97', 'Nov.97', 'Dec.97', 'Jan.98',
                 'Feb.98', 'Mar.98', 'Apr.98', 'May.98', 'Jun.98', 'Jul.98', 'Aug.98',
                 'Sep.98', 'Oct.98', 'Nov.98', 'Dec.98', 'Jan.99', 'Feb.99', 'Mar.99',
                 'Apr.99', 'May.99', 'Jun.99']

    states = ['FE', 'HE', 'employment', 'joblessness', 'school', 'training']
    labels = ['further education', 'higher education', 'employment', 'joblessness', 'school', 'training']

    # TODO: write a try and error: if no such a parameter, then ask to pass the right ones
    # sequence_data = SequenceData(df, time=time, id_col="country", ids=df['country'].values, states=states)

    sequence_data = SequenceData(df,
                                 time=time_list,
                                 id_col="id",
                                 states=states,
                                 labels=labels,
                                 )

    om = get_distance_matrix(sequence_data,
                             method="OM",
                             sm="CONSTANT",
                             indel=1)

    cluster = Cluster(om, sequence_data.ids, clustering_method='ward_d')
    cluster.plot_dendrogram(xlabel="Individuals", ylabel="Distance")

    # Create a ClusterQuality object to evaluate clustering quality
    cluster_quality = ClusterQuality(cluster)
    cluster_quality.compute_cluster_quality_scores()
    cluster_quality.plot_cqi_scores(norm='zscore')
    summary_table = cluster_quality.get_cqi_table()
    print(summary_table)

    table = cluster_quality.get_cluster_range_table()
    # table.to_csv("cluster_quality_table.csv")

    print(table)
