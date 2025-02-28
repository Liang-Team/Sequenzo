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
from .plot_transition_rate_matrix import compute_transition_matrix, print_transition_matrix, plot_transition_matrix
from .plot_mean_time import plot_mean_time
from .plot_single_medoid import plot_single_medoid, compute_medoids_from_distance_matrix


# 改为延迟导入
def _get_standard_scaler():
    try:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler
    except ImportError:
        print("警告: 无法导入 StandardScaler。请确保已正确安装 scikit-learn。")
        return None


__all__ = [
    "plot_sequence_index",
    "plot_most_frequent_sequences",
    "plot_single_medoid",
    # Add other functions as needed
]