"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 02/05/2025 11:05
@Desc    :
    Prefix Tree Framework - exposes core indicators and utilities for sequence divergence analysis.
"""
from .system_level_indicators import (
    build_prefix_tree,
    compute_prefix_count,
    compute_branching_factor,
    compute_js_divergence,
    plot_system_indicators,
    plot_system_indicators_multiple_comparison
)

from .individual_level_indicators import IndividualDivergence, plot_prefix_rarity_distribution, plot_individual_indicators_correlation

from .utils import (
    extract_sequences,
    get_state_space,
    convert_to_prefix_tree_data
)

__all__ = [
    # System-level
    "build_prefix_tree",
    "compute_prefix_count",
    "compute_branching_factor",
    "compute_js_divergence",
    "plot_system_indicators",
    "plot_system_indicators_multiple_comparison",

    # Individual-level
    "IndividualDivergence",
    "plot_prefix_rarity_distribution",
    "plot_individual_indicators_correlation",

    # Utilities
    "extract_sequences",
    "get_state_space",
    "convert_to_prefix_tree_data",
]