"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 08/08/2025 15:50
@Desc    :
    Suffix Tree Framework - exposes core indicators and utilities for sequence convergence analysis.
"""
from .system_level_indicators import (
    build_suffix_tree,
    compute_suffix_count,
    compute_merging_factor,
    compute_js_convergence,
    plot_system_indicators,
    plot_system_indicators_multiple_comparison,
)

from .individual_level_indicators import (
    IndividualConvergence,
    compute_path_uniqueness_by_group,
    plot_suffix_rarity_distribution,
)

from .utils import (
    extract_sequences,
    get_state_space,
    convert_to_suffix_tree_data
)

__all__ = [
    # System-level
    "build_suffix_tree",
    "compute_suffix_count",
    "compute_merging_factor",
    "compute_js_convergence",
    # plotting
    "plot_system_indicators",
    "plot_system_indicators_multiple_comparison",

    # Individual-level
    "IndividualConvergence",
    "compute_path_uniqueness_by_group",
    "plot_suffix_rarity_distribution",

    # Utilities
    "extract_sequences",
    "get_state_space",
    "convert_to_suffix_tree_data",
]