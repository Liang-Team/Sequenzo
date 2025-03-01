"""
@Author  : 梁彧祺
@File    : __init__.py.py
@Time    : 11/02/2025 16:42
@Desc    : 
"""
# sequenzo/visualization/__init__.py

from .plot_sequence_index import plot_sequence_index
from .plot_most_frequent_sequences import plot_most_frequent_sequences
from .plot_relative_frequency import plot_relative_frequency
from .plot_transition_matrix import compute_transition_matrix, print_transition_matrix, plot_transition_matrix
from .plot_mean_time import plot_mean_time
from .plot_single_medoid import plot_single_medoid, compute_medoids_from_distance_matrix
from .plot_state_distribution import plot_state_distribution
from .plot_modal_state import plot_modal_state


# Delay imports to avoid circular dependency issues during installation
def _get_standard_scaler():
    try:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler
    except ImportError:
        print("Warning: Not able to install StandardScaler。Please ensure that you have installed scikit-learn successfully.")
        return None


__all__ = [
    "plot_mean_time",
    "plot_most_frequent_sequences",
    "plot_relative_frequency",
    "plot_sequence_index",
    "plot_single_medoid",
    "plot_state_distribution",
    "plot_transition_matrix",
    "plot_modal_state",
    # Add other functions as needed
]