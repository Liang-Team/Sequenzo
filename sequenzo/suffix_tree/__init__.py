"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 08/08/2025 15:50
@Desc    :
    Suffix Tree Framework - exposes core indicators and utilities for sequence convergence analysis.
    Supports both position-based (level = time from end) and spell-based (level = spell from end) modes via hub.
"""
from .hub import build_suffix_tree
from .system_level_indicators import (
    SuffixTree,
    get_depth_stats,
    compute_suffix_count,
    compute_merging_factor,
    compute_js_convergence,
    plot_system_indicators,
    plot_system_indicators_multiple_comparison,
)
from .spell_level_indicators import (
    SpellSuffixTree,
    build_spell_suffix_tree,
    compute_js_convergence_spell,
)
from .spell_individual_level_indicators import SpellIndividualConvergence

from .individual_level_indicators import (
    IndividualConvergence,
    compute_path_uniqueness_by_group,
    plot_suffix_rarity_distribution,
    plot_individual_indicators_correlation,
)

from .utils import (
    extract_sequences,
    get_state_space,
    convert_to_suffix_tree_data
)

__all__ = [
    # Hub (unified entry point: mode="position" | "spell", expcost)
    "build_suffix_tree",

    # System-level (works for both SuffixTree and SpellSuffixTree)
    "SuffixTree",
    "SpellSuffixTree",
    "get_depth_stats",
    "compute_suffix_count",
    "compute_merging_factor",
    "compute_js_convergence",
    "compute_js_convergence_spell",
    "build_spell_suffix_tree",
    "SpellIndividualConvergence",
    "plot_system_indicators",
    "plot_system_indicators_multiple_comparison",

    # Individual-level
    "IndividualConvergence",
    "compute_path_uniqueness_by_group",
    "plot_suffix_rarity_distribution",
    "plot_individual_indicators_correlation",

    # Utilities
    "extract_sequences",
    "get_state_space",
    "convert_to_suffix_tree_data",
]