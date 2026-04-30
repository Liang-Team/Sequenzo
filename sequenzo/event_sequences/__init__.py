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
from .visualization import plot_event_sequences, plot_subsequence_frequencies


def is_eseq(obj: Any) -> bool:
    """TraMineR-style type check for a single event sequence."""
    return isinstance(obj, EventSequence)


def is_seqelist(obj: Any) -> bool:
    """TraMineR-style type check for an event-sequence collection."""
    return isinstance(obj, EventSequenceList)


def seqelength(obj: EventSequence | EventSequenceList):
    """
    TraMineR-style sequence length helper.

    - EventSequence -> int
    - EventSequenceList -> numpy array of per-sequence lengths
    """
    if isinstance(obj, EventSequence):
        return len(obj)
    if isinstance(obj, EventSequenceList):
        return obj.lengths
    raise TypeError("seqelength expects EventSequence or EventSequenceList.")


def seqeweight(obj: EventSequenceList):
    """TraMineR-style weight accessor for EventSequenceList."""
    if isinstance(obj, EventSequenceList):
        return obj.weights
    raise TypeError("seqeweight expects EventSequenceList.")


__all__ = [
    "create_event_sequences",
    "find_frequent_subsequences",
    "count_subsequence_occurrences",
    "compare_groups",
    "convert_event_sequences_to_tse",
    "compute_event_transition_matrix",
    "check_event_subsequence_containment",
    "EventSequence",
    "EventSequenceList",
    "EventSequenceConstraint",
    "SubsequenceList",
    "plot_event_sequences",
    "plot_subsequence_frequencies",
    "is_eseq",
    "is_seqelist",
    "seqelength",
    "seqeweight",
]
