"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 30/09/2025 23:34
@Desc    : Event History Analysis module for sequence analysis
"""

from .sequence_analysis_multi_state_model import (
    SAMM,
    sequence_analysis_multi_state_model,
    plot_samm,
    seqsammseq,
    set_typology,
    seqsammeha,
    # Keep old names for backward compatibility
    seqsamm
)

from .sequence_history_analysis import (
    seqsha,
    person_level_to_person_period
)

# Event Sequence Analysis (TraMineR-compatible)
from .event_sequence import (
    create_event_sequences,
    find_frequent_subsequences,
    compare_groups,
    count_subsequence_occurrences,
    EventSequence,
    EventSequenceList,
    EventSequenceConstraint,
    SubsequenceList,
)
from .event_sequence_visualization import (
    plot_event_sequences,
    plot_subsequence_frequencies,
)

__all__ = [
    # Event History Analysis (SAMM)
    'SAMM',
    'sequence_analysis_multi_state_model',
    'plot_samm',
    'seqsammseq',
    'set_typology',
    'seqsammeha',
    'seqsha',
    'person_level_to_person_period',
    # Keep old names for backward compatibility
    'seqsamm',
    # Event Sequence Analysis
    'create_event_sequences',
    'find_frequent_subsequences',
    'compare_groups',
    'count_subsequence_occurrences',
    'plot_event_sequences',
    'plot_subsequence_frequencies',
    'EventSequence',
    'EventSequenceList',
    'EventSequenceConstraint',
    'SubsequenceList'
]
