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
from scipy.spatial.distance import squareform
# sklearn metrics no longer needed - using C++ implementation
# Import from sequenzo_fastcluster (our custom fastcluster with ward_d and ward_d2 support)
try:
    from sequenzo.clustering.sequenzo_fastcluster.fastcluster import linkage
except ImportError:
    # Fallback: try absolute import
    try:
        from sequenzo_fastcluster.fastcluster import linkage
    except ImportError:
        # Last resort: try relative import
        from .sequenzo_fastcluster.fastcluster import linkage

# Import C++ cluster quality functions
try:
    from . import clustering_c_code
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    print("[!] Warning: C++ cluster quality functions not available. Using Python fallback.")


# Corrected imports: Use relative imports *within* the package.
from sequenzo.visualization.utils import save_and_show_results

# Global flag to ensure Ward warning is only shown once per session
_WARD_WARNING_SHOWN = False

def _check_euclidean_compatibility(matrix, method):
    """
    Check if a distance matrix is likely compatible with Euclidean-based methods like Ward.
    
    This performs heuristic checks rather than exact validation since perfect validation
    would be computationally expensive for large matrices.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Distance matrix to check
    method : str
        Clustering method name
        
    Returns:
    --------
    bool
        True if matrix appears Euclidean-compatible, False otherwise
    """
    # Check for Ward methods (both Ward D and Ward D2 require Euclidean distances)
    if method.lower() not in ["ward", "ward_d", "ward_d2"]:
        return True  # Other methods don't require Euclidean distances
    
    # Basic checks for Euclidean properties
    n = matrix.shape[0]
    
    # Check 1: Triangle inequality violations (sample a subset for large matrices)
    sample_size = min(50, n)  # Sample up to 50 points for efficiency
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        sample_matrix = matrix[np.ix_(indices, indices)]
    else:
        sample_matrix = matrix
        indices = np.arange(n)
    
    sample_n = sample_matrix.shape[0]
    violations = 0
    total_checks = 0
    
    # Check triangle inequality: d(i,k) <= d(i,j) + d(j,k)
    for i in range(sample_n):
        for j in range(i + 1, sample_n):
            for k in range(j + 1, sample_n):
                dij = sample_matrix[i, j]
                dik = sample_matrix[i, k]
                djk = sample_matrix[j, k]
                
                # Check all three triangle inequalities
                if (dik > dij + djk + 1e-10 or 
                    dij > dik + djk + 1e-10 or 
                    djk > dij + dik + 1e-10):
                    violations += 1
                total_checks += 1
    
    if total_checks > 0:
        violation_rate = violations / total_checks
        if violation_rate > 0.1:  # More than 10% violations suggests non-Euclidean
            return False
    
    # Check 2: Negative eigenvalues in distance matrix (indicates non-Euclidean)
    # Use double centering to convert distances to inner products
    try:
        # For efficiency, only check this for smaller matrices
        if sample_n <= 100:
            H = np.eye(sample_n) - np.ones((sample_n, sample_n)) / sample_n
            B = -0.5 * H @ (sample_matrix ** 2) @ H
            eigenvals = np.linalg.eigvals(B)
            
            # Check if there are significant negative eigenvalues
            negative_eigenvals = eigenvals[eigenvals < -1e-10]
            if len(negative_eigenvals) > 0:
                neg_energy = -np.sum(negative_eigenvals)
                total_energy = np.sum(np.abs(eigenvals))
                if neg_energy / total_energy > 0.1:  # > 10% negative energy
                    return False
    except np.linalg.LinAlgError:
        # If eigenvalue computation fails, assume potentially problematic
        pass
    
    return True


def _warn_ward_usage_once(matrix, method):
    """
    Issue a one-time warning about using Ward with potentially non-Euclidean distances.
    """
    global _WARD_WARNING_SHOWN
    
    # Check for both Ward D and Ward D2 methods
    if not _WARD_WARNING_SHOWN and method.lower() in ["ward", "ward_d", "ward_d2"]:
        if not _check_euclidean_compatibility(matrix, method):
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


def _clean_distance_matrix(matrix):
    """
    Clean and validate a distance matrix for hierarchical clustering.
    
    This function:
    1. Handles NaN/Inf values using robust percentile-based replacement
    2. Sets diagonal to zero
    3. Ensures non-negativity
    
    Note: Symmetry is NOT enforced at this stage since distance matrices may legitimately 
    be asymmetric (e.g., directed sequence distances, time-dependent measures, etc.).
    However, symmetrization will be performed later in linkage computation when required 
    by clustering algorithms.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input distance matrix
        
    Returns:
    --------
    np.ndarray
        Cleaned distance matrix
    """
    matrix = matrix.copy()  # Don't modify the original
    
    # Step 1: Handle NaN/Inf values with percentile-based replacement
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        print("[!] Warning: Distance matrix contains NaN or Inf values.")
        
        # Get finite values for percentile calculation
        finite_vals = matrix[np.isfinite(matrix)]
        
        if len(finite_vals) > 0:
            # Use 95th percentile as replacement value (more conservative than max)
            replacement_val = np.percentile(finite_vals, 95)
            print(f"    Replacing with 95th percentile value: {replacement_val:.6f}")
        else:
            # If no finite values, use 1.0 as default
            replacement_val = 1.0
            print(f"    No finite values found, using default: {replacement_val}")
        
        matrix[~np.isfinite(matrix)] = replacement_val
    
    # Step 2: Force diagonal to be exactly zero (self-distance should be zero)
    np.fill_diagonal(matrix, 0.0)
    
    # Step 3: Ensure non-negativity (distance matrices should be non-negative)
    if np.any(matrix < 0):
        print("[!] Warning: Distance matrix contains negative values. Clipping to zero...")
        matrix = np.maximum(matrix, 0.0)
    
    return matrix


def _hclust_to_linkage_matrix(linkage_matrix):
    """
    Convert an R `hclust` object to a SciPy-compatible linkage matrix.

    This function takes an `hclust` object returned by R (e.g., from
    `fastcluster::hclust`) and converts it into the standard linkage matrix
    format used by SciPy (`scipy.cluster.hierarchy.linkage`), which can be
    used for dendrogram plotting or further clustering analysis in Python.

    Parameters
    ----------
    linkage_matrix : rpy2.robjects.ListVector
        An R `hclust` object. Expected to contain at least the following fields:
        - 'merge': ndarray of shape (n-1, 2), indicating which clusters are merged
                   at each step (negative indices for original observations,
                   positive indices for previously merged clusters).
        - 'height': ndarray of shape (n-1,), distances at which merges occur.
        - 'order': ordering of the leaves.

    Returns
    -------
    Z : numpy.ndarray, shape (n-1, 4), dtype=float
        A SciPy-compatible linkage matrix where each row represents a merge:
        - Z[i, 0] : index of the first cluster (0-based)
        - Z[i, 1] : index of the second cluster (0-based)
        - Z[i, 2] : distance between the merged clusters
        - Z[i, 3] : total number of original samples in the newly formed cluster

    Notes
    -----
    - The conversion handles the difference in indexing:
        - In R's `hclust`, negative numbers in 'merge' indicate original samples
          and positive numbers indicate previously merged clusters (1-based).
        - In the returned SciPy linkage matrix, all indices are converted to 0-based.
    - The function iteratively tracks cluster sizes to populate the fourth column
      (sample counts) required by SciPy.
    """

    n = len(linkage_matrix.rx2("order"))  # 样本数
    merge = np.array(linkage_matrix.rx2("merge"), dtype=int)  # (n-1, 2)
    height = np.array(linkage_matrix.rx2("height"), dtype=float)

    cluster_sizes = np.ones(n, dtype=int)  # 单个样本初始大小 = 1
    Z = np.zeros((n - 1, 4), dtype=float)

    for i in range(n - 1):
        a, b = merge[i]

        # R hclust 编号负数表示原始样本
        if a < 0:
            idx1 = -a - 1  # 转成 0-based
            size1 = 1
        else:
            idx1 = n + a - 1  # 已合并簇，0-based
            size1 = cluster_sizes[idx1]

        if b < 0:
            idx2 = -b - 1
            size2 = 1
        else:
            idx2 = n + b - 1
            size2 = cluster_sizes[idx2]

        Z[i, 0] = idx1
        Z[i, 1] = idx2
        Z[i, 2] = height[i]
        Z[i, 3] = size1 + size2

        # 更新 cluster_sizes，用于后续簇
        cluster_sizes = np.append(cluster_sizes, size1 + size2)

    return Z

class Cluster:
    def __init__(self,
                 matrix,
                 entity_ids,
                 clustering_method="ward",
                 weights=None):
        """
        A class to handle hierarchical clustering operations using fastcluster for improved performance.

        :param matrix: Precomputed distance matrix (full square form).
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
        """
        # Ensure entity_ids is a numpy array for consistent processing
        self.entity_ids = np.array(entity_ids)

        # Check if entity_ids is valid
        if len(self.entity_ids) != len(matrix):
            raise ValueError("Length of entity_ids must match the size of the matrix.")

        # Optional: Check uniqueness of entity_ids
        if len(np.unique(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids must contain unique values.")

        # Initialize and validate weights
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float64)
            if len(self.weights) != len(matrix):
                raise ValueError("Length of weights must match the size of the matrix.")
            if np.any(self.weights < 0):
                raise ValueError("All weights must be non-negative.")
            if np.sum(self.weights) == 0:
                raise ValueError("Sum of weights must be greater than zero.")
        else:
            # Default to equal weights (all ones)
            self.weights = np.ones(len(matrix), dtype=np.float64)

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
        supported_methods = ["ward", "ward_d", "ward_d2", "single", "complete", "average", "centroid", "median"]
        if self.clustering_method not in supported_methods:
            raise ValueError(
                f"Unsupported clustering method '{clustering_method}'. Supported methods: {supported_methods}")
        
        # Handle backward compatibility: 'ward' maps to 'ward_d' (classic Ward method)
        if self.clustering_method == "ward":
            self.clustering_method = "ward_d"
            print("[>] Note: 'ward' method maps to 'ward_d' (classic Ward method).")
            print("    Use 'ward_d2' for Ward method with squared Euclidean distances.")

        # Compute linkage matrix using fastcluster
        self.linkage_matrix = self._compute_linkage()

    def _compute_linkage(self):
        """
        Compute the linkage matrix using fastcluster for improved performance.
        Supports both Ward D (classic) and Ward D2 methods.
        """
        # Clean and validate the distance matrix using robust methods
        self.full_matrix = _clean_distance_matrix(self.full_matrix)
        
        # Check Ward compatibility and issue one-time warning if needed
        _warn_ward_usage_once(self.full_matrix, self.clustering_method)

        # Check symmetry before converting to condensed form
        # squareform() requires symmetric matrices
        if not np.allclose(self.full_matrix, self.full_matrix.T, rtol=1e-5, atol=1e-8):
            print("[!] Warning: Distance matrix is not symmetric.")
            print("    Hierarchical clustering algorithms require symmetric distance matrices.")
            print("    Automatically symmetrizing using (matrix + matrix.T) / 2")
            print("    If this is not appropriate for your data, please provide a symmetric matrix.")
            self.full_matrix = (self.full_matrix + self.full_matrix.T) / 2

        # Convert square matrix to condensed form
        self.condensed_matrix = squareform(self.full_matrix)

        # Map our method names to fastcluster's expected method names
        fastcluster_method = self._map_method_name(self.clustering_method)

        linkage_matrix = linkage(self.condensed_matrix, method=fastcluster_method)

        return linkage_matrix

    def _map_method_name(self, method):
        """
        Map our internal method names to fastcluster's expected method names.
        """
        method_mapping = {
            "ward_d": "ward",    # Classic Ward (will be corrected later) (updated: it was solved on Nov.15, 2025 by Xinyi)
            "ward_d2": "ward_d2",   # Ward D2 (no correction needed)
            "single": "single",
            "complete": "complete",
            "average": "average",
            "centroid": "centroid",
            "median": "median"
        }
        return method_mapping.get(method, method)
    
    def _apply_ward_d_correction(self, linkage_matrix):
        """
        Apply Ward D correction by dividing distances by 2.
        This converts Ward D2 results to classic Ward D results.
        """
        linkage_corrected = linkage_matrix.copy()
        linkage_corrected[:, 2] = linkage_corrected[:, 2] / 2.0
        print("[>] Applied Ward D correction: distances divided by 2 for classic Ward method.")
        return linkage_corrected

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
    def __init__(self, matrix_or_cluster, max_clusters=20, clustering_method=None, weights=None):
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
        :param weights: Optional array of weights for each entity. If None and using Cluster instance,
                       weights will be extracted from the Cluster object.
        """
        if isinstance(matrix_or_cluster, Cluster):
            # Extract matrix, clustering method, and weights from the Cluster instance
            self.matrix = matrix_or_cluster.full_matrix
            self.clustering_method = matrix_or_cluster.clustering_method
            self.linkage_matrix = matrix_or_cluster.linkage_matrix
            self.weights = matrix_or_cluster.weights

        elif isinstance(matrix_or_cluster, (np.ndarray, pd.DataFrame)):
            # Handle direct matrix input
            if isinstance(matrix_or_cluster, pd.DataFrame):
                print("[>] Detected Pandas DataFrame. Converting to NumPy array...")
                matrix_or_cluster = matrix_or_cluster.values
            self.matrix = matrix_or_cluster
            self.clustering_method = clustering_method or "ward_d"  # Default to classic Ward
            
            # Initialize weights for direct matrix input
            if weights is not None:
                self.weights = np.array(weights, dtype=np.float64)
                if len(self.weights) != len(self.matrix):
                    raise ValueError("Length of weights must match the size of the matrix.")
            else:
                self.weights = np.ones(len(self.matrix), dtype=np.float64)
            
            # Compute linkage matrix for direct input (needed for clustering operations)
            self.linkage_matrix = self._compute_linkage_for_direct_input()

        else:
            raise ValueError(
                "Input must be a Cluster instance, a NumPy array, or a Pandas DataFrame."
            )

        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be a full square-form distance matrix.")

        self.max_clusters = max_clusters
        self.metric_order = [
            "PBC",
            "HG",
            "HGSD",
            "ASW",
            "ASWw",
            "CH",
            "R2",
            "CHsq",
            "R2sq",
            "HC",
        ]
        self.scores = {metric: [] for metric in self.metric_order}

        # Store original scores separately to preserve raw values
        self.original_scores = None

    def _compute_linkage_for_direct_input(self):
        """
        Compute linkage matrix for direct matrix input (similar to Cluster class logic).
        Supports both Ward D and Ward D2 methods.
        """
        # Handle backward compatibility: 'ward' maps to 'ward_d'
        if self.clustering_method == "ward":
            self.clustering_method = "ward_d"
            print("[>] Note: 'ward' method maps to 'ward_d' (classic Ward method).")
            print("    Use 'ward_d2' for Ward method with squared Euclidean distances.")
            
        # Clean and validate the distance matrix using robust methods
        self.matrix = _clean_distance_matrix(self.matrix)
        
        # Check Ward compatibility and issue one-time warning if needed
        _warn_ward_usage_once(self.matrix, self.clustering_method)

        # Check symmetry before converting to condensed form
        # squareform() requires symmetric matrices
        if not np.allclose(self.matrix, self.matrix.T, rtol=1e-5, atol=1e-8):
            print("[!] Warning: Distance matrix is not symmetric.")
            print("    Hierarchical clustering algorithms require symmetric distance matrices.")
            print("    Automatically symmetrizing using (matrix + matrix.T) / 2")
            print("    If this is not appropriate for your data, please provide a symmetric matrix.")
            self.matrix = (self.matrix + self.matrix.T) / 2

        # Convert square matrix to condensed form for linkage computation
        condensed_matrix = squareform(self.matrix)

        try:
            # Map our method names to fastcluster's expected method names
            fastcluster_method = self._map_method_name(self.clustering_method)
            linkage_matrix = linkage(condensed_matrix, method=fastcluster_method)
            
            # Apply Ward D correction if needed
            if self.clustering_method == "ward_d":
                linkage_matrix = self._apply_ward_d_correction(linkage_matrix)
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute linkage with method '{self.clustering_method}'. "
                "Check that the distance matrix is square, symmetric, finite, non-negative, and has a zero diagonal. "
                "For sequence distances, consider using 'average', 'complete', or 'single' instead of Ward methods. "
                f"Original error: {e}"
            )
        return linkage_matrix
        
    def _map_method_name(self, method):
        """
        Map our internal method names to fastcluster's expected method names.
        """
        method_mapping = {
            "ward_d": "ward",    # Classic Ward (will be corrected later)
            "ward_d2": "ward",   # Ward D2 (no correction needed)
            "single": "single",
            "complete": "complete",
            "average": "average",
            "centroid": "centroid",
            "median": "median"
        }
        return method_mapping.get(method, method)
    
    def _apply_ward_d_correction(self, linkage_matrix):
        """
        Apply Ward D correction by dividing distances by 2.
        This converts Ward D2 results to classic Ward D results.
        """
        linkage_corrected = linkage_matrix.copy()
        linkage_corrected[:, 2] = linkage_corrected[:, 2] / 2.0
        print("[>] Applied Ward D correction: distances divided by 2 for classic Ward method.")
        return linkage_corrected

    def compute_cluster_quality_scores(self):
        """
        Compute clustering quality scores for different numbers of clusters.
        
        Uses C++ implementation for accuracy and performance.
        This implementation aligns with R WeightedCluster package results.
        """
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ cluster quality implementation is not available. "
                "Please ensure the C++ extensions are properly compiled."
            )
        self._compute_cluster_quality_scores_cpp()
        
        # Save original scores immediately after computation
        self.original_scores = {}
        for metric, values in self.scores.items():
            self.original_scores[metric] = np.array(values).copy()

    def _compute_cluster_quality_scores_cpp(self):
        """
        Compute clustering quality scores using C++ implementation (matches R WeightedCluster).
        """
        # Convert matrix to format expected by C++
        # Ensure we have a full square matrix
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square for C++ implementation")
        
        # Convert to condensed once to reduce per-call overhead in C++
        condensed = squareform(self.matrix)

        for k in range(2, self.max_clusters + 1):
            # Get cluster labels (fcluster returns 1-based labels, which C++ expects)
            labels = fcluster(self.linkage_matrix, k, criterion="maxclust")
            
            try:
                # Call C++ function (condensed) - expects 1-based labels
                result = clustering_c_code.cluster_quality_condensed(
                    condensed.astype(np.float64, copy=False),
                    labels.astype(np.int32, copy=False),
                    self.weights.astype(np.float64, copy=False),
                    self.matrix.shape[0],
                    k
                )
                
                # Extract results from C++ (mapping to match R WeightedCluster exactly)
                for metric in self.metric_order:
                    self.scores[metric].append(result.get(metric, np.nan))
                
            except Exception as e:
                print(f"[!] Error: C++ computation failed for k={k}: {e}")
                print("    Python fallback has been removed due to accuracy issues.")
                # Insert NaN values for failed computation
                for metric in self.metric_order:
                    self.scores[metric].append(np.nan)
                raise RuntimeError(f"C++ cluster quality computation failed for k={k}. "
                                   "Python fallback is not available.")

    def _compute_cluster_quality_scores_python(self):
        """
        Python fallback implementation has been removed.
        Only C++ implementation is available for accuracy and performance.
        """
        raise NotImplementedError(
            "Python cluster quality implementation has been removed due to accuracy issues. "
            "Please use C++ implementation by setting use_cpp=True (default)."
        )

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

    def get_cluster_range_table(self) -> pd.DataFrame:
        """
        Return a metrics-by-cluster table mirroring R's `as.clustrange()` output.

        :return: DataFrame indexed by cluster count ("cluster2", ...)
                 with raw metric values for each quality indicator.
        """
        # Prefer preserved raw scores to avoid normalization side-effects
        if self.original_scores is not None:
            scores_to_use = self.original_scores
        else:
            scores_to_use = self.scores

        # Ensure metrics are available
        if not scores_to_use or not any(len(scores_to_use[m]) for m in self.metric_order):
            raise ValueError("Cluster quality scores are empty. Run `compute_cluster_quality_scores()` first.")

        # Determine number of evaluated cluster counts
        lengths = [len(scores_to_use[metric]) for metric in self.metric_order if metric in scores_to_use]
        if not lengths:
            raise ValueError("No recognized metrics found in scores.")

        if len(set(lengths)) != 1:
            raise ValueError("Inconsistent metric lengths detected. Please recompute cluster quality scores.")

        n_rows = lengths[0]
        if n_rows == 0:
            raise ValueError("Cluster quality scores contain no entries.")

        # Build DataFrame matching R output ordering
        data = {}
        for metric in self.metric_order:
            values = scores_to_use.get(metric)
            if values is None:
                continue
            data[metric] = np.array(values, dtype=np.float64)

        index_labels = [f"cluster{k}" for k in range(2, 2 + n_rows)]
        table = pd.DataFrame(data, index=index_labels)
        table.index.name = "Cluster"

        return table

    def get_cqi_table(self):
        """
        Generate a summary table of clustering quality indicators with concise column names.

        :return: Pandas DataFrame summarizing the optimal number of clusters (N groups),
                 the corresponding raw metric values, and z-score normalized values.
        """
        # Use original scores if available, otherwise fall back to current scores
        if self.original_scores is not None:
            scores_to_use = self.original_scores
        else:
            scores_to_use = self.scores
        
        # Deep copy to avoid overwriting during normalization
        original_scores = {}
        for metric, values in scores_to_use.items():
            original_scores[metric] = np.array(values).copy()

        # Create temporary copy for z-score normalization
        temp_scores = {}
        for metric, values in original_scores.items():
            temp_scores[metric] = values.copy()
        
        # Apply z-score normalization to temp copy
        zscore_normalized = {}
        for metric in temp_scores:
            values = temp_scores[metric]
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            if std_val > 0:
                zscore_normalized[metric] = (values - mean_val) / std_val
            else:
                zscore_normalized[metric] = values.copy()

        # Generate summary table (removed redundant Min-Max Norm column)
        summary = {
            "Metric": [],
            "Opt. Clusters": [],  # Abbreviated from "Optimal Clusters"
            "Raw Value": [],  # Raw optimal value (not normalized)
            "Z-Score Norm.": [],  # Z-Score normalized optimal value
        }

        # Get maximum value and its position from original scores
        for metric in self.metric_order:
            values = original_scores.get(metric)
            if values is None:
                continue

            if np.all(np.isnan(values)):
                optimal_k, raw_value, z_val = np.nan, np.nan, np.nan
            else:
                pos = np.nanargmax(values)
                optimal_k = pos + 2
                raw_value = values[pos]  # Use raw original value
                z_val = zscore_normalized[metric][pos]

            # Add data to the summary table
            summary["Metric"].append(metric)
            summary["Opt. Clusters"].append(optimal_k)
            summary["Raw Value"].append(raw_value)  # Raw value, not normalized
            summary["Z-Score Norm."].append(z_val)

        return pd.DataFrame(summary)

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
        # Store original scores before normalization
        original_scores = self.scores.copy()

        # Calculate statistics from original data
        original_stats = {}
        for metric in metrics_list or self.metric_order:
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
            metrics_list = list(self.metric_order)
        else:
            metrics_list = [metric for metric in metrics_list if metric in self.metric_order]

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
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
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
        self.weights = cluster.weights  # Retrieve weights from Cluster class

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
        return pd.DataFrame({"Entity ID": self.entity_ids, "Cluster": cluster_labels})

    def get_cluster_distribution(self, num_clusters, weighted=False) -> pd.DataFrame:
        """
        Generate a distribution summary of clusters showing counts, percentages, and optionally weighted statistics.

        This function calculates how many entities belong to each cluster and what
        percentage of the total they represent. When weighted=True, it also provides
        weight-based statistics.

        :param num_clusters: The number of clusters to create.
        :param weighted: If True, include weighted statistics in the distribution.
        :return: DataFrame with cluster distribution information.
        """
        # Get cluster memberships
        memberships_df = self.get_cluster_memberships(num_clusters)

        # Count entities in each cluster
        cluster_counts = memberships_df['Cluster'].value_counts().sort_index()

        # Calculate percentages
        total_entities = len(memberships_df)
        cluster_percentages = (cluster_counts / total_entities * 100).round(2)

        # Create basic distribution dataframe
        distribution = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Count': cluster_counts.values,
            'Percentage': cluster_percentages.values
        }).sort_values('Cluster')

        # Add weighted statistics if requested
        if weighted:
            cluster_weights = []
            weighted_percentages = []
            total_weight = np.sum(self.weights)
            
            for cluster_id in distribution['Cluster']:
                # Find entities in this cluster
                cluster_mask = memberships_df['Cluster'] == cluster_id
                cluster_entity_indices = memberships_df.index[cluster_mask]
                
                # Sum weights for entities in this cluster
                cluster_weight = np.sum(self.weights[cluster_entity_indices])
                cluster_weights.append(cluster_weight)
                
                # Calculate weighted percentage
                weighted_pct = (cluster_weight / total_weight * 100) if total_weight > 0 else 0.0
                weighted_percentages.append(round(weighted_pct, 2))
            
            distribution['Weight_Sum'] = cluster_weights
            distribution['Weight_Percentage'] = weighted_percentages

        return distribution

    def plot_cluster_distribution(self, num_clusters, save_as=None, title=None,
                                  style="whitegrid", dpi=200, figsize=(10, 6), weighted=False):
        """
        Plot the distribution of entities across clusters as a bar chart.

        This visualization shows how many entities belong to each cluster, providing
        insight into the balance and size distribution of the clustering result.
        When weighted=True, displays weight-based percentages.

        :param num_clusters: The number of clusters to create.
        :param save_as: File path to save the plot. If None, the plot will be shown.
        :param title: Title for the plot. If None, a default title will be used.
        :param style: Seaborn style for the plot.
        :param dpi: DPI for saved image.
        :param figsize: Figure size in inches.
        :param weighted: If True, display weighted percentages instead of entity count percentages.
        """
        # Get cluster distribution data (include weights if needed)
        distribution = self.get_cluster_distribution(num_clusters, weighted=weighted)

        # Set up plot
        sns.set(style=style)
        plt.figure(figsize=figsize)

        # Choose what to plot based on weighted parameter
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

        # Create bar plot with a more poetic, fresh color palette
        # 'muted', 'pastel', and 'husl' are good options for fresher colors
        ax = sns.barplot(x='Cluster', y=y_column, data=distribution, palette='pastel')

        # Set the Y-axis range to prevent text overflow
        ax.set_ylim(0, distribution[y_column].max() * 1.2)

        # Ensure Y-axis uses appropriate ticks
        if not weighted:
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        # Add percentage labels on top of bars
        for p, (_, row) in zip(ax.patches, distribution.iterrows()):
            height = p.get_height()
            percentage = row[percentage_column]
            ax.text(p.get_x() + p.get_width() / 2., height + max(height * 0.02, 0.5),
                    f'{percentage:.1f}%', ha="center", fontsize=9)

        # Set a simple label for entity count at the top
        if title is None:
            if weighted:
                title = f"N = {len(self.entity_ids)}, Total Weight = {np.sum(self.weights):.1f}"
            else:
                title = f"N = {len(self.entity_ids)}"

        # Use a lighter, non-bold title style
        plt.title(title, fontsize=12, fontweight="normal", loc='right')

        plt.xlabel("Cluster ID", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Ensure integer ticks for cluster IDs
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Add grid for better readability but make it lighter
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Adjust layout
        plt.tight_layout()

        # Adjust layout to make room for the note
        plt.subplots_adjust(bottom=0.13)

        # Add a note about what is being displayed
        plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=10, style='italic')

        # Save and show the plot
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
