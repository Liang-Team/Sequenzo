"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 27/02/2025 09:58
@Desc    : 
"""
from .compare_cluster_methods import (
    ClusterRangeFamilyResult,
    compare_cluster_methods,
    hierarchical_cluster_range,
)
from .hierarchical_clustering import Cluster, ClusterResults, ClusterQuality
from .k_medoids import KMedoids
from .k_medoids_range import k_medoids_range
from .utils.aggregate_cases import AggregateCasesResult, aggregate_cases
from .sequences_to_variables import (
    representativeness_matrix,
    medoid_indices_from_kmedoids_result,
    cluster_labels_from_kmedoids_result,
    hard_classification_variables,
    fanny,
    FannyResult,
    fanny_membership,
    medoid_membership_approximation,
    highest_membership_indices_from_membership,
    soft_classification_variables,
    pseudoclass_regression,
    max_distance,
    cluster_labels_to_dummies,
)
from .fuzzy_clustering import (
    wfcmdd,
    crispness,
    WfcmddResult,
    fuzzy_sequence_plot,
    fuzzy_sequence_plot_single,
    get_fuzzy_clusters,
    FuzzyClusterResult,
    membership_summary,
    most_typical_members,
    prepare_dirichlet_data,
    DirichletRegData,
    DirichletRegResult,
    dirichlet_regression,
    beta_regression,
)
from .property_based_clustering import (
    extract_sequence_properties,
    property_based_clustering,
    seqpropclust,
    cluster_split_schedule,
    cut_tree,
    prune_property_tree,
    tree_labels,
    property_clustering_quality,
    print_property_tree,
    plot_property_tree,
    SUPPORTED_PROPERTIES,
)
from .validation import (
    cluster_range_from_partitions,
    compute_partition_quality,
    ClusterRangeResult,
    boot_cluster_range,
    cluster_association,
    plot_cluster_association,
    BootClusterRangeResult,
    observation_silhouette,
    rarcat,
    RarcatResult,
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
    "ClusterRangeFamilyResult",
    "compare_cluster_methods",
    "hierarchical_cluster_range",
    "KMedoids",
    "k_medoids_range",
    "AggregateCasesResult",
    "aggregate_cases",
    "max_distance",
    "cluster_labels_to_dummies",
    "representativeness_matrix",
    "medoid_indices_from_kmedoids_result",
    "cluster_labels_from_kmedoids_result",
    "hard_classification_variables",
    "fanny",
    "FannyResult",
    "fanny_membership",
    "medoid_membership_approximation",
    "highest_membership_indices_from_membership",
    "soft_classification_variables",
    "pseudoclass_regression",
    "wfcmdd",
    "crispness",
    "WfcmddResult",
    "fuzzy_sequence_plot",
    "fuzzy_sequence_plot_single",
    "get_fuzzy_clusters",
    "FuzzyClusterResult",
    "membership_summary",
    "most_typical_members",
    "prepare_dirichlet_data",
    "DirichletRegData",
    "DirichletRegResult",
    "dirichlet_regression",
    "beta_regression",
    "extract_sequence_properties",
    "property_based_clustering",
    "seqpropclust",
    "cluster_split_schedule",
    "cut_tree",
    "prune_property_tree",
    "tree_labels",
    "property_clustering_quality",
    "print_property_tree",
    "plot_property_tree",
    "SUPPORTED_PROPERTIES",
    "cluster_range_from_partitions",
    "compute_partition_quality",
    "ClusterRangeResult",
    "boot_cluster_range",
    "cluster_association",
    "plot_cluster_association",
    "BootClusterRangeResult",
    "observation_silhouette",
    "rarcat",
    "RarcatResult",
]
