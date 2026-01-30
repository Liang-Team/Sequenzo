"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 02/05/2025 11:05
@Desc    :
    Prefix Tree Framework - exposes core indicators and utilities for sequence divergence analysis.
    Supports both position-based (level = time) and spell-based (level = spell) modes via hub.
"""
from .hub import build_prefix_tree
from .system_level_indicators import (
    PrefixTree,
    get_depth_stats,
    compute_prefix_count,
    compute_branching_factor,
    compute_js_divergence,
    plot_system_indicators,
    plot_system_indicators_multiple_comparison,
)
from .spell_level_indicators import (
    SpellPrefixTree,
    build_spell_prefix_tree,
    compute_js_divergence_spell,
    convert_seqdata_to_spells,
)

from .individual_level_indicators import IndividualDivergence, plot_prefix_rarity_distribution, plot_individual_indicators_correlation

from .utils import (
    extract_sequences,
    get_state_space,
    convert_to_prefix_tree_data
)

__all__ = [
    # Hub (unified entry point: mode="position" | "spell", expcost)
    "build_prefix_tree",

    # System-level (works for both PrefixTree and SpellPrefixTree)
    "PrefixTree",
    "SpellPrefixTree",
    "get_depth_stats",
    "compute_prefix_count",
    "compute_branching_factor",
    "compute_js_divergence",
    "compute_js_divergence_spell",
    "build_spell_prefix_tree",
    "convert_seqdata_to_spells",
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