"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:42
@Desc    : Sequenzo Package Initialization
"""
# sequenzo/__init__.py (Top-level)

__version__ = "0.1.12"

# Lazy import: public submodules
from sequenzo import (datasets, data_preprocessing, visualization, clustering,
                      dissimilarity_measures, big_data, define_sequence_data,
                      multidomain, prefix_tree)


def __getattr__(name):
    try:
        if name == "datasets":
            from sequenzo import datasets
            return datasets
        elif name == "data_preprocessing":
            from sequenzo import data_preprocessing
        elif name == "visualization":
            from sequenzo import visualization
            return visualization
        elif name == "clustering":
            from sequenzo import clustering
            return clustering
        elif name == "dissimilarity_measures":
            from sequenzo import dissimilarity_measures
            return dissimilarity_measures
        elif name == "SequenceData":
            from sequenzo.define_sequence_data import SequenceData
            return SequenceData
        elif name == "big_data":
            from sequenzo.big_data import big_data
            return big_data
        elif name == "multidomain":
            from sequenzo import multidomain
            return multidomain
        elif name == "prefix_tree":
            from sequenzo import prefix_tree
            return prefix_tree
    except ImportError as e:
        raise AttributeError(f"Could not import {name}: {e}")

    raise AttributeError(f"module 'sequenzo' has no attribute '{name}'")


# Explicit re-export for IDE autocomplete
SequenceData = define_sequence_data.SequenceData

__all__ = [
    'datasets',
    'data_preprocessing',
    'visualization',
    'clustering',
    'dissimilarity_measures',
    'SequenceData',
    'big_data',
    'multidomain',
    'prefix_tree'
]
