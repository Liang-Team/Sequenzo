"""
@Author  : 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:42
@Desc    : Sequenzo Package Initialization
"""
# Define the version number - this must be done before all imports
__version__ = "0.1.0"

from sequenzo import datasets, visualization, clustering, dissimilarity_measures, SequenceData, big_data


def __getattr__(name):
    try:
        if name == "datasets":
            from sequenzo import datasets
            return datasets
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
            from sequenzo.big_data import clara
            return clara
    except ImportError as e:
        raise AttributeError(f"Could not import {name}: {e}")

    raise AttributeError(f"module 'sequenzo' has no attribute '{name}'")


# These are the public APIs of the package, but use __getattr__ for lazy imports
__all__ = [
    'datasets',
    'visualization',
    'clustering',
    'dissimilarity_measures',
    'SequenceData',
    'big_data',
]


