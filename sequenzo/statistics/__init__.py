"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 2026-04-27 17:21
@Desc    : 
Statistics namespace for user-facing analytical helpers.
"""

from .weighted import (
    get_weighted_five_number_summary,
    get_weighted_mean,
    get_weighted_variance,
)
from .sequence_statistics import (
    get_distinct_state_sequences,
    get_individual_state_distribution,
    get_mean_time_by_state,
    get_modal_state_sequence,
    get_sequence_length_summary,
    get_state_spell_durations,
    get_transition_count_summary,
)

__all__ = [
    "get_weighted_mean",
    "get_weighted_variance",
    "get_weighted_five_number_summary",
    "get_distinct_state_sequences",
    "get_state_spell_durations",
    "get_mean_time_by_state",
    "get_individual_state_distribution",
    "get_modal_state_sequence",
    "get_sequence_length_summary",
    "get_transition_count_summary",
]
