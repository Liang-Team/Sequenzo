"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py.py
@Time    : 02/05/2025 11:05
@Desc    :
    Prefix Tree Framework – exposes core indicators and utilities for sequence divergence analysis.
"""
from .system_level_indicators import (
    build_prefix_tree,
    compute_prefix_count,
    compute_branching_factor,
    compute_js_divergence,
    compute_composite_score,
    plot_system_indicators
)

from .individual_level_indicators import IndividualDivergence

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
    "compute_composite_score",
    "plot_system_indicators",

    # Individual-level
    "IndividualDivergence",

    # Utilities
    "extract_sequences",
    "get_state_space",
    "convert_to_prefix_tree_data",
]