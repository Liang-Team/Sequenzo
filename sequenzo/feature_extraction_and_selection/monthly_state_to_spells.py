"""
@Author  : Yuqi Liang 梁彧祺
@File    : monthly_state_to_spells.py
@Time    : 19/03/2026 14:26
@Desc    :
    Convert monthly state trajectories to spell trajectories with start/end times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from sequenzo.define_sequence_data import SequenceData
from sequenzo.prefix_tree import convert_seqdata_to_spells

from .time_binning_utils import coerce_numeric_time_labels


@dataclass(frozen=True)
class SpellWithTimes:
    state: Any
    start_time: float
    end_time: float
    duration_months: float


def extract_spells_with_times(seqdata: SequenceData) -> List[List[SpellWithTimes]]:
    if not isinstance(seqdata, SequenceData):
        raise TypeError("seqdata must be a SequenceData instance.")

    spell_states, spell_durations, _ = convert_seqdata_to_spells(seqdata)
    time_vals = coerce_numeric_time_labels(seqdata.time)
    T = len(time_vals)

    spells_per_individual: List[List[SpellWithTimes]] = []
    for states_i, durs_i in zip(spell_states, spell_durations):
        spells_i: List[SpellWithTimes] = []
        pos = 0
        for s, d in zip(states_i, durs_i):
            d_int = int(d)
            if d_int <= 0:
                break
            if pos + d_int - 1 >= T:
                raise ValueError("Spell durations exceed the SequenceData time grid length.")
            start_idx = pos
            end_idx = pos + d_int - 1
            spells_i.append(
                SpellWithTimes(
                    state=s,
                    start_time=float(time_vals[start_idx]),
                    end_time=float(time_vals[end_idx]),
                    duration_months=float(d_int),
                )
            )
            pos += d_int
        spells_per_individual.append(spells_i)
    return spells_per_individual

