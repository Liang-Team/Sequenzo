"""
@Author  : 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:42
@Desc    : Sequenzo Package Initialization
"""

# Define the version number - this must be done before all imports
__version__ = "0.1.0"


def __getattr__(name):
    """
    # Delay imports to avoid circular dependency issues during installation
    # This allows setuptools to retrieve `__version__` without requiring all dependencies to be installed first.
    """
    if name == "datasets":
        from . import datasets
        return datasets
    elif name == "visualization":
        from . import visualization
        return visualization
    elif name == "clustering":
        from . import clustering
        return clustering
    elif name == "dissimilarity_measures":
        from . import dissimilarity_measures
        return dissimilarity_measures
    elif name == "SequenceData":
        from .define_sequence_data import SequenceData
        return SequenceData
    elif name == "big_data":
        from .big_data import clara
        return clara

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
