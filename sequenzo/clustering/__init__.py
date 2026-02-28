"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 27/02/2025 09:58
@Desc    : 
"""
from .hierarchical_clustering import Cluster, ClusterResults, ClusterQuality
from .KMedoids import KMedoids
from .seqs2vars_utils import max_distance, cluster_labels_to_dummies
from .sequences_to_variables import (
    representativeness_matrix,
    medoid_indices_from_kmedoids_result,
    cluster_labels_from_kmedoids_result,
    hard_classification_variables,
    fanny_membership,
    soft_classification_variables,
    pseudoclass_regression,
)


def _import_c_code():
    """Lazily import the c_code module to avoid circular dependencies during installation"""
    try:
        # Import built pybind11 extension placed under this package
        from sequenzo.clustering import clustering_c_code
        return clustering_c_code
    except ImportError:
        # If the C extension cannot be imported, return None
        print(
            "Warning: The C++ extension (c_code) could not be imported. Please ensure the extension module is compiled correctly.")
        return None


__all__ = [
    "Cluster",
    "ClusterResults",
    "ClusterQuality",
    "KMedoids",
    "max_distance",
    "cluster_labels_to_dummies",
    "representativeness_matrix",
    "medoid_indices_from_kmedoids_result",
    "cluster_labels_from_kmedoids_result",
    "hard_classification_variables",
    "fanny_membership",
    "soft_classification_variables",
    "pseudoclass_regression",
]
