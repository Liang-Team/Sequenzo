import subprocess
import sys
import types


def test_same_named_package_exports_are_callable_submodules():
    code = """
import inspect
import types
import sequenzo.clustering
import sequenzo.clustering.k_medoids_range
import sequenzo.clustering.compare_cluster_methods
import sequenzo.clustering.property_based_clustering
import sequenzo.dissimilarity_measures
import sequenzo.dissimilarity_measures.get_distance_matrix
import sequenzo.dissimilarity_measures.get_substitution_cost_matrix
import sequenzo.clustering.sequences_to_variables.fanny as dotted_fanny

targets = [
    (sequenzo.clustering.k_medoids_range, "k_medoids_range"),
    (sequenzo.clustering.compare_cluster_methods, "compare_cluster_methods"),
    (sequenzo.clustering.property_based_clustering, "property_based_clustering"),
    (sequenzo.dissimilarity_measures.get_distance_matrix, "get_distance_matrix"),
    (
        sequenzo.dissimilarity_measures.get_substitution_cost_matrix,
        "get_substitution_cost_matrix",
    ),
    (dotted_fanny, "fanny"),
]

for module, export_name in targets:
    assert isinstance(module, types.ModuleType), type(module)
    assert callable(module), module
    assert callable(getattr(module, export_name)), export_name
    assert inspect.signature(module) == inspect.signature(getattr(module, export_name))
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_from_package_imports_remain_callable_for_same_named_exports():
    from sequenzo.clustering import compare_cluster_methods, k_medoids_range
    from sequenzo.clustering.sequences_to_variables import fanny
    from sequenzo.dissimilarity_measures import (
        get_distance_matrix,
        get_substitution_cost_matrix,
    )

    assert callable(k_medoids_range)
    assert callable(compare_cluster_methods)
    assert callable(fanny)
    assert callable(get_distance_matrix)
    assert callable(get_substitution_cost_matrix)


def test_callable_submodule_exports_preserve_function_signatures():
    import inspect
    import sequenzo.clustering.k_medoids_range as k_medoids_range_module
    import sequenzo.clustering.sequences_to_variables.fanny as fanny_module
    import sequenzo.dissimilarity_measures.get_distance_matrix as get_distance_matrix_module
    from sequenzo.clustering import k_medoids_range
    from sequenzo.clustering.sequences_to_variables import fanny
    from sequenzo.dissimilarity_measures import get_distance_matrix

    assert inspect.signature(k_medoids_range) == inspect.signature(
        k_medoids_range_module.k_medoids_range
    )
    assert inspect.signature(fanny) == inspect.signature(fanny_module.fanny)
    assert inspect.signature(get_distance_matrix) == inspect.signature(
        get_distance_matrix_module.get_distance_matrix
    )


def test_importlib_returns_same_named_submodules():
    import importlib

    targets = [
        "sequenzo.clustering.k_medoids_range",
        "sequenzo.clustering.compare_cluster_methods",
        "sequenzo.clustering.property_based_clustering",
        "sequenzo.clustering.sequences_to_variables.fanny",
        "sequenzo.dissimilarity_measures.get_distance_matrix",
        "sequenzo.dissimilarity_measures.get_substitution_cost_matrix",
    ]

    for path in targets:
        module = importlib.import_module(path)
        assert isinstance(module, types.ModuleType)
