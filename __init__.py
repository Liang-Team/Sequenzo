"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:42
@Desc    : Sequenzo Package Initialization
"""
# sequenzo/__init__.py (Top-level)

__version__ = "0.1.39"

import importlib


_SUBMODULES = {
    "datasets",
    "data_preprocessing",
    "visualization",
    "clustering",
    "dissimilarity_measures",
    "big_data",
    "define_sequence_data",
    "multidomain",
    "prefix_tree",
    "suffix_tree",
    "sequence_characteristics_indicators",
}


def __getattr__(name):
    try:
        if name in _SUBMODULES:
            return importlib.import_module(f"sequenzo.{name}")
        elif name == "SequenceData":
            from sequenzo.define_sequence_data import SequenceData
            return SequenceData
    except ImportError as e:
        raise AttributeError(f"Could not import {name}: {e}")

    raise AttributeError(f"module 'sequenzo' has no attribute '{name}'")


# Explicit lightweight re-export for IDE autocomplete.
from sequenzo.define_sequence_data import SequenceData

__all__ = [
    'datasets',
    'data_preprocessing',
    'visualization',
    'clustering',
    'dissimilarity_measures',
    'SequenceData',
    'big_data',
    'multidomain',
    'prefix_tree',
    'suffix_tree',
    'sequence_characteristics_indicators',
]
