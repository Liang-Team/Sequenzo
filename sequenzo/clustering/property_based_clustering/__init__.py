"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 13/05/2026 10:21
@Desc    :
Property-based clustering for sequence analysis (Studer 2018).

Mirrors WeightedCluster ``seqpropclust`` and related tree utilities.
"""
from .property_extraction import SUPPORTED_PROPERTIES, extract_sequence_properties
from .property_clustering import property_based_clustering, seqpropclust
from .quality import as_clustrange_property_tree, property_clustering_quality
from .tree_schedule import cluster_split_schedule, cut_tree, prune_property_tree, tree_labels
from .visualization import plot_property_tree, print_property_tree

__all__ = [
    "SUPPORTED_PROPERTIES",
    "extract_sequence_properties",
    "property_based_clustering",
    "seqpropclust",
    "cluster_split_schedule",
    "cut_tree",
    "prune_property_tree",
    "tree_labels",
    "property_clustering_quality",
    "as_clustrange_property_tree",
    "print_property_tree",
    "plot_property_tree",
]
