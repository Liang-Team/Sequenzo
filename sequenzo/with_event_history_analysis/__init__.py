"""
@Author  : Yuqi Liang 梁彧祺
@File    : __init__.py
@Time    : 30/09/2025 23:34
@Desc    : Event History Analysis module (SAMM / sequence history only)

Boundary note:
- Event-sequence construction/mining/visualization is now in `sequenzo.event_sequences`.
- This module is reserved for event-history methods (e.g., SAMM, get_sequence_history_data).
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
    get_sequence_history_data,
    person_level_to_person_period
)

__all__ = [
    # Event History Analysis (SAMM)
    'SAMM',
    'sequence_analysis_multi_state_model',
    'plot_samm',
    'seqsammseq',
    'set_typology',
    'seqsammeha',
    'get_sequence_history_data',
    'person_level_to_person_period',
    # Keep old names for backward compatibility
    'seqsamm',
]
