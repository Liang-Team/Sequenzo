"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 27/02/2025 09:58
@Desc    :
"""
from __future__ import annotations

import importlib
import inspect
import sys
import types
from typing import Any


_LAZY: dict[str, tuple[str, str]] = {
    "Cluster": ("sequenzo.clustering.hierarchical_clustering", "Cluster"),
    "ClusterResults": ("sequenzo.clustering.hierarchical_clustering", "ClusterResults"),
    "ClusterQuality": ("sequenzo.clustering.hierarchical_clustering", "ClusterQuality"),
    "ClusterRangeFamilyResult": (
        "sequenzo.clustering.compare_cluster_methods",
        "ClusterRangeFamilyResult",
    ),
    "compare_cluster_methods": (
        "sequenzo.clustering.compare_cluster_methods",
        "compare_cluster_methods",
    ),
    "hierarchical_cluster_range": (
        "sequenzo.clustering.compare_cluster_methods",
        "hierarchical_cluster_range",
    ),
    "KMedoids": ("sequenzo.clustering.k_medoids", "KMedoids"),
    "k_medoids_range": ("sequenzo.clustering.k_medoids_range", "k_medoids_range"),
    "AggregateCasesResult": (
        "sequenzo.clustering.utils.aggregate_cases",
        "AggregateCasesResult",
    ),
    "aggregate_cases": ("sequenzo.clustering.utils.aggregate_cases", "aggregate_cases"),
    "max_distance": ("sequenzo.clustering.sequences_to_variables", "max_distance"),
    "cluster_labels_to_dummies": (
        "sequenzo.clustering.sequences_to_variables",
        "cluster_labels_to_dummies",
    ),
    "representativeness_matrix": (
        "sequenzo.clustering.sequences_to_variables",
        "representativeness_matrix",
    ),
    "medoid_indices_from_kmedoids_result": (
        "sequenzo.clustering.sequences_to_variables",
        "medoid_indices_from_kmedoids_result",
    ),
    "cluster_labels_from_kmedoids_result": (
        "sequenzo.clustering.sequences_to_variables",
        "cluster_labels_from_kmedoids_result",
    ),
    "hard_classification_variables": (
        "sequenzo.clustering.sequences_to_variables",
        "hard_classification_variables",
    ),
    "fanny": ("sequenzo.clustering.sequences_to_variables", "fanny"),
    "FannyResult": ("sequenzo.clustering.sequences_to_variables", "FannyResult"),
    "fanny_membership": (
        "sequenzo.clustering.sequences_to_variables",
        "fanny_membership",
    ),
    "medoid_membership_approximation": (
        "sequenzo.clustering.sequences_to_variables",
        "medoid_membership_approximation",
    ),
    "highest_membership_indices_from_membership": (
        "sequenzo.clustering.sequences_to_variables",
        "highest_membership_indices_from_membership",
    ),
    "soft_classification_variables": (
        "sequenzo.clustering.sequences_to_variables",
        "soft_classification_variables",
    ),
    "pseudoclass_regression": (
        "sequenzo.clustering.sequences_to_variables",
        "pseudoclass_regression",
    ),
    "wfcmdd": ("sequenzo.clustering.fuzzy_clustering", "wfcmdd"),
    "crispness": ("sequenzo.clustering.fuzzy_clustering", "crispness"),
    "WfcmddResult": ("sequenzo.clustering.fuzzy_clustering", "WfcmddResult"),
    "fuzzy_sequence_plot": (
        "sequenzo.clustering.fuzzy_clustering",
        "fuzzy_sequence_plot",
    ),
    "fuzzy_sequence_plot_single": (
        "sequenzo.clustering.fuzzy_clustering",
        "fuzzy_sequence_plot_single",
    ),
    "get_fuzzy_clusters": (
        "sequenzo.clustering.fuzzy_clustering",
        "get_fuzzy_clusters",
    ),
    "FuzzyClusterResult": (
        "sequenzo.clustering.fuzzy_clustering",
        "FuzzyClusterResult",
    ),
    "membership_summary": (
        "sequenzo.clustering.fuzzy_clustering",
        "membership_summary",
    ),
    "most_typical_members": (
        "sequenzo.clustering.fuzzy_clustering",
        "most_typical_members",
    ),
    "prepare_dirichlet_data": (
        "sequenzo.clustering.fuzzy_clustering",
        "prepare_dirichlet_data",
    ),
    "DirichletRegData": (
        "sequenzo.clustering.fuzzy_clustering",
        "DirichletRegData",
    ),
    "DirichletRegResult": (
        "sequenzo.clustering.fuzzy_clustering",
        "DirichletRegResult",
    ),
    "dirichlet_regression": (
        "sequenzo.clustering.fuzzy_clustering",
        "dirichlet_regression",
    ),
    "beta_regression": ("sequenzo.clustering.fuzzy_clustering", "beta_regression"),
    "extract_sequence_properties": (
        "sequenzo.clustering.property_based_clustering",
        "extract_sequence_properties",
    ),
    "property_based_clustering": (
        "sequenzo.clustering.property_based_clustering",
        "property_based_clustering",
    ),
    "seqpropclust": (
        "sequenzo.clustering.property_based_clustering",
        "seqpropclust",
    ),
    "cluster_split_schedule": (
        "sequenzo.clustering.property_based_clustering",
        "cluster_split_schedule",
    ),
    "cut_tree": ("sequenzo.clustering.property_based_clustering", "cut_tree"),
    "prune_property_tree": (
        "sequenzo.clustering.property_based_clustering",
        "prune_property_tree",
    ),
    "tree_labels": ("sequenzo.clustering.property_based_clustering", "tree_labels"),
    "property_clustering_quality": (
        "sequenzo.clustering.property_based_clustering",
        "property_clustering_quality",
    ),
    "print_property_tree": (
        "sequenzo.clustering.property_based_clustering",
        "print_property_tree",
    ),
    "plot_property_tree": (
        "sequenzo.clustering.property_based_clustering",
        "plot_property_tree",
    ),
    "SUPPORTED_PROPERTIES": (
        "sequenzo.clustering.property_based_clustering",
        "SUPPORTED_PROPERTIES",
    ),
    "cluster_range_from_partitions": (
        "sequenzo.clustering.validation",
        "cluster_range_from_partitions",
    ),
    "compute_partition_quality": (
        "sequenzo.clustering.validation",
        "compute_partition_quality",
    ),
    "ClusterRangeResult": ("sequenzo.clustering.validation", "ClusterRangeResult"),
    "boot_cluster_range": ("sequenzo.clustering.validation", "boot_cluster_range"),
    "cluster_association": ("sequenzo.clustering.validation", "cluster_association"),
    "plot_cluster_association": (
        "sequenzo.clustering.validation",
        "plot_cluster_association",
    ),
    "BootClusterRangeResult": (
        "sequenzo.clustering.validation",
        "BootClusterRangeResult",
    ),
    "observation_silhouette": (
        "sequenzo.clustering.validation",
        "observation_silhouette",
    ),
    "rarcat": ("sequenzo.clustering.validation", "rarcat"),
    "RarcatResult": ("sequenzo.clustering.validation", "RarcatResult"),
}

_loaded: dict[str, Any] = {}


class _CallableExportModule(types.ModuleType):
    """Submodule that can also be called like its same-named exported function."""

    _sequenzo_export_attr: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self, self._sequenzo_export_attr)(*args, **kwargs)


def _is_same_named_submodule(name: str, mod_path: str, attr: str) -> bool:
    return attr == name and mod_path.rsplit(".", 1)[-1] == name


def _callable_submodule(module: types.ModuleType, attr: str) -> types.ModuleType:
    if not isinstance(module, _CallableExportModule):
        module.__class__ = _CallableExportModule
    module._sequenzo_export_attr = attr
    exported = getattr(module, attr)
    module.__signature__ = inspect.signature(exported)
    module.__wrapped__ = exported
    return module


def _load_export(name: str) -> Any:
    mod_path, attr = _LAZY[name]
    module = _loaded.get(mod_path)
    if module is None:
        module = importlib.import_module(mod_path)
        _loaded[mod_path] = module
    if _is_same_named_submodule(name, mod_path, attr):
        return _callable_submodule(module, attr)
    return getattr(module, attr)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = _load_export(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


class _LazyExportModule(types.ModuleType):
    """Resolve lazy exports even after Python attaches same-named submodules."""

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        lazy = super().__getattribute__("_LAZY")
        if name in lazy:
            mod_path, attr = lazy[name]
            if isinstance(value, types.ModuleType) and getattr(value, "__name__", None) == mod_path:
                if _is_same_named_submodule(name, mod_path, attr):
                    return _callable_submodule(value, attr)
                return getattr(value, attr)
        return value


sys.modules[__name__].__class__ = _LazyExportModule


def _import_c_code():
    """Lazily import the clustering C++ extension."""
    try:
        return importlib.import_module("sequenzo.clustering.clustering_c_code")
    except ImportError:
        print(
            "Warning: The C++ extension (c_code) could not be imported. "
            "Please ensure the extension module is compiled correctly."
        )
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
