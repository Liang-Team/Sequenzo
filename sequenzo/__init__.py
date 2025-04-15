"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 11/02/2025 16:41
@Desc    : 
"""
# sequenzo/sequenzo/__init__.py (Inside the inner sequenzo package)
# This file is now needed for making the submodules accessible.

# You *don't* need to handle the Rust extension here.
# You can have package level docstring here.
# No need for __all__ here, use relative import in the top-level __init__.py

from .datasets import load_dataset, list_datasets

# Import the core functions that should be directly available from the sequenzo package
from sequenzo.define_sequence_data import *
from .visualization import (plot_sequence_index,
                            plot_most_frequent_sequences,
                            plot_single_medoid,
                            plot_state_distribution,
                            plot_modal_state,
                            plot_relative_frequency,
                            plot_mean_time,
                            plot_transition_matrix)

from .dissimilarity_measures.get_distance_matrix import get_distance_matrix
from .dissimilarity_measures.get_substitution_cost_matrix import get_substitution_cost_matrix

from .clustering import Cluster, ClusterResults, ClusterQuality
from .big_data.clara.clara import clara
from .big_data.clara.visualization import plot_scores_from_dataframe


# Define `__all__` to specify the public API when using `from sequenzo import *`
__all__ = [
    "load_dataset",
    "list_datasets",
    "SequenceData",
    "plot_sequence_index",
    "plot_most_frequent_sequences",
    "plot_single_medoid",
    "get_distance_matrix",
    "get_substitution_cost_matrix",
    "Cluster",
    "ClusterResults",
    "ClusterQuality",
    "plot_state_distribution",
    "plot_modal_state",
    "plot_relative_frequency",
    "plot_mean_time",
    "plot_transition_matrix",
    "clara",
    "plot_scores_from_dataframe"
]

