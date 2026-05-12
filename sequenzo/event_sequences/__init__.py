"""
Event Sequences API (TraMineR-friendly entrypoint).

Use this module for:
- event-sequence data structures
- event-sequence pattern mining and group comparison
- event-sequence visualization

Do NOT use this module for event-history modeling (SAMM / sequence history).
Those methods live in `sequenzo.with_event_history_analysis`.
"""

from __future__ import annotations

from typing import Any

from .core import (
    EventSequence,
    EventSequenceConstraint,
    EventSequenceData,
    EventSequenceList,
    SubsequenceList,
    check_event_subsequence_containment,
    compare_groups,
    compute_event_transition_matrix,
    convert_event_sequences_to_tse,
    count_subsequence_occurrences,
    create_event_sequences,
    find_frequent_subsequences,
)
from .visualization import (
    plot_event_dynamics,
    plot_event_sequences,
    plot_event_parallel_coordinates,
    plot_subsequence_frequencies,
    plot_subsequence_group_contrasts,
)


def is_event_sequence(obj: Any) -> bool:
    """Type check for a single event sequence."""
    return isinstance(obj, EventSequence)


def is_event_sequence_collection(obj: Any) -> bool:
    """Type check for an event-sequence collection."""
    return isinstance(obj, EventSequenceList)


def get_event_sequence_lengths(obj: EventSequence | EventSequenceList):
    """
    Get sequence length information.

    - EventSequence -> int
    - EventSequenceList -> numpy array of per-sequence lengths
    """
    if isinstance(obj, EventSequence):
        return len(obj)
    if isinstance(obj, EventSequenceList):
        return obj.lengths
    raise TypeError("get_event_sequence_lengths expects EventSequence or EventSequenceList.")


def get_event_sequence_weights(obj: EventSequenceList):
    """Get sequence weights for an EventSequenceList."""
    if isinstance(obj, EventSequenceList):
        return obj.weights
    raise TypeError("get_event_sequence_weights expects EventSequenceList.")


__all__ = [
    "create_event_sequences",
    "find_frequent_subsequences",
    "count_subsequence_occurrences",
    "compare_groups",
    "convert_event_sequences_to_tse",
    "compute_event_transition_matrix",
    "check_event_subsequence_containment",
    "EventSequence",
    "EventSequenceData",
    "EventSequenceList",
    "EventSequenceConstraint",
    "SubsequenceList",
    "plot_event_sequences",
    "plot_event_parallel_coordinates",
    "plot_subsequence_frequencies",
    "plot_subsequence_group_contrasts",
    "plot_event_dynamics",
    "is_event_sequence",
    "is_event_sequence_collection",
    "get_event_sequence_lengths",
    "get_event_sequence_weights",
]
