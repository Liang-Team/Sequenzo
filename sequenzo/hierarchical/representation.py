"""
@Author  : 梁彧祺 Yuqi Liang, Jan Meyerhoff-Liang
@File    : representation.py
@Time    : 03/04/2026 20:17
@Desc    :
    Sequence representation helpers for hierarchical analysis (state vs spell).
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .data import RelationalSequenceData, RelationalSequenceRecord

Spell = Tuple[Any, int]


def state_sequence_to_spells(sequence: Sequence[Any]) -> List[Spell]:
    """
    Collapse a state sequence into consecutive spells (state, duration).

    Missing values (NaN) end the current spell and are skipped in output.
    """
    if len(sequence) == 0:
        return []

    spells: List[Spell] = []
    current_state = None
    run_length = 0

    for state in sequence:
        if _is_missing(state):
            if run_length > 0:
                spells.append((current_state, run_length))
                run_length = 0
                current_state = None
            continue

        if current_state is None:
            current_state = state
            run_length = 1
        elif state == current_state:
            run_length += 1
        else:
            spells.append((current_state, run_length))
            current_state = state
            run_length = 1

    if run_length > 0 and current_state is not None:
        spells.append((current_state, run_length))

    return spells


def to_spell_sequences(
    sequence_data: Union[RelationalSequenceData, Sequence[RelationalSequenceRecord]],
) -> List[List[Spell]]:
    """
    Convert pair-level state sequences to spell representations.

    Parameters
    ----------
    sequence_data : RelationalSequenceData or sequence of records

    Returns
    -------
    list
        One list of (state, duration) spells per pair, in the same order as input.
    """
    if isinstance(sequence_data, RelationalSequenceData):
        records = sequence_data.records
    else:
        records = list(sequence_data)

    return [state_sequence_to_spells(rec.sequence) for rec in records]


def encode_states(
    sequences: Sequence[Sequence[Any]],
) -> Tuple[np.ndarray, List[Any]]:
    """
    Map categorical states to integer codes for internal use.

    Returns
    -------
    codes : ndarray of shape (n_sequences, max_length)
        -1 marks padding when sequences differ in length.
    alphabet : list
        State labels in code order.
    """
    if len(sequences) == 0:
        return np.empty((0, 0), dtype=int), []

    alphabet = sorted(
        {s for seq in sequences for s in seq if not _is_missing(s)},
        key=lambda x: str(x),
    )
    state_to_code = {s: i for i, s in enumerate(alphabet)}
    max_len = max(len(seq) for seq in sequences) if sequences else 0
    n = len(sequences)
    codes = np.full((n, max_len), -1, dtype=int)

    for i, seq in enumerate(sequences):
        for j, s in enumerate(seq):
            if _is_missing(s):
                continue
            codes[i, j] = state_to_code[s]

    return codes, alphabet


def _is_missing(value: Any) -> bool:
    """True for None, NaN, NaT, and pandas missing markers."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (ValueError, TypeError):
        return False
