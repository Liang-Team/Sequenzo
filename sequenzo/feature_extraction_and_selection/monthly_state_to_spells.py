"""
@Author  : Yuqi Liang 梁彧祺
@File    : monthly_state_to_spells.py
@Time    : 19/03/2026 14:26
@Desc    :
    Convert state sequences to spell trajectories with start/end times.

    Spell durations are counted in **sequence positions** (steps on the time
    grid), not necessarily calendar months.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal

from sequenzo.define_sequence_data import SequenceData
from sequenzo.prefix_tree import convert_seqdata_to_spells

from .time_binning_utils import coerce_numeric_time_labels

EndTimeMode = Literal["last_observed", "exit_time"]


@dataclass(frozen=True)
class SpellWithTimes:
    state: Any
    start_time: float
    end_time: float
    duration_steps: float


def extract_spells_with_times(
    seqdata: SequenceData,
    *,
    end_time_mode: EndTimeMode = "last_observed",
) -> List[List[SpellWithTimes]]:
    """
    Extract spells with numeric start/end times from ``seqdata``.

    Parameters
    ----------
    end_time_mode
        How to timestamp leaving a state (relevant for ``END_*`` timing features):

        - ``last_observed``: last occupied time point on the grid (default).
        - ``exit_time``: first time point **after** the spell (transition time),
          when available; otherwise falls back to ``last_observed``.
    """
    if end_time_mode not in {"last_observed", "exit_time"}:
        raise ValueError("end_time_mode must be 'last_observed' or 'exit_time'.")
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
            start_t = float(time_vals[start_idx])
            if end_time_mode == "exit_time" and end_idx + 1 < T:
                end_t = float(time_vals[end_idx + 1])
            else:
                end_t = float(time_vals[end_idx])
            spells_i.append(
                SpellWithTimes(
                    state=s,
                    start_time=start_t,
                    end_time=end_t,
                    duration_steps=float(d_int),
                )
            )
            pos += d_int
        spells_per_individual.append(spells_i)
    return spells_per_individual
