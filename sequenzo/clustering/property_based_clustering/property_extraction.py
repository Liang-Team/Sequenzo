"""
@Author  : Yuqi Liang 梁彧祺
@File    : property_extraction.py
@Time    : 13/05/2026 12:24
@Desc    :
Sequence property extraction for property-based clustering (Studer 2018).

Mirrors the property lists used by WeightedCluster ``seqpropclust``.
"""
from __future__ import annotations

import contextlib
import io
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from sequenzo.event_sequences import (
    EventSequenceConstraint,
    EventSequenceData,
    count_subsequence_occurrences,
    find_frequent_subsequences,
)
from sequenzo.sequence_characteristics_indicators.complexity_index import get_complexity_index
from sequenzo.sequence_characteristics_indicators.simple_characteristics import get_number_of_transitions
from sequenzo.sequence_characteristics_indicators.turbulence import get_turbulence
from sequenzo.sequence_characteristics_indicators.within_sequence_entropy import get_within_sequence_entropy

from sequenzo.event_sequences.core import _find_first_occurrence

SUPPORTED_PROPERTIES = (
    "state",
    "spell.age",
    "spell.dur",
    "duration",
    "pattern",
    "AFpattern",
    "transition",
    "AFtransition",
    "Complexity",
)


def _state_labels(seqdata: SequenceData, with_missing: bool) -> List[str]:
    labels = list(seqdata.labels) if seqdata.labels is not None else [str(s) for s in seqdata.states]
    if with_missing and getattr(seqdata, "missing_state", None) is not None:
        labels = labels + [str(seqdata.missing_state)]
    return labels


def _is_dss_padding(value) -> bool:
    if pd.isna(value):
        return True
    try:
        return int(value) < 0
    except (TypeError, ValueError):
        return False


def _as_state_label(value, seqdata: SequenceData, labels: List[str]) -> Optional[str]:
    """Map a DSS cell value to the state label used in WeightedCluster spell properties."""
    if _is_dss_padding(value):
        return None
    states = list(seqdata.states)
    try:
        numeric = float(value)
        if numeric.is_integer():
            idx = int(numeric) - 1
            if 0 <= idx < len(labels):
                return labels[idx]
    except (TypeError, ValueError):
        pass
    if value in states:
        return labels[states.index(value)]
    str_value = str(value)
    if str_value in labels:
        return str_value
    return str_value


def _r_data_frame_name(name: str) -> str:
    """Match R ``make.names()`` for spell column initialization."""
    if not name:
        return name
    if name[0].isdigit():
        return f"X{name}"
    return name


def _spell_field_name(label: str, kind: str, index: int) -> str:
    return f"{label}_{kind}_{index}"


def _prefix_block_columns(block: pd.DataFrame, block_name: str) -> pd.DataFrame:
    if block.empty:
        return block
    return block.rename(columns={col: f"{block_name}.{col}" for col in block.columns})


def _extract_state_properties(seqdata: SequenceData) -> pd.DataFrame:
    labels = _state_labels(seqdata, with_missing=False)
    panel = seqdata.seqdata.copy()

    def _to_label(value):
        if pd.isna(value):
            return value
        try:
            numeric = float(value)
            if numeric.is_integer():
                idx = int(numeric) - 1
                if 0 <= idx < len(labels):
                    return labels[idx]
        except (TypeError, ValueError):
            pass
        return _as_state_label(value, seqdata, labels)

    mapped = panel.map(_to_label)
    return mapped.reset_index(drop=True)


def _extract_duration_properties(seqdata: SequenceData, with_missing: bool) -> pd.DataFrame:
    labels = _state_labels(seqdata, with_missing=with_missing)
    seq_matrix = seqdata.seqdata.values
    columns = {
        label: (seq_matrix == (idx + 1)).sum(axis=1).astype(float)
        for idx, label in enumerate(labels)
    }
    # TraMineR seqistatd always includes the void state column "*".
    columns.setdefault("*", np.zeros(seq_matrix.shape[0], dtype=float))
    return pd.DataFrame(columns).reset_index(drop=True)


def _extract_spell_properties(
    seqdata: SequenceData,
    *,
    spell_dur: bool,
    spell_age: bool,
    with_missing: bool,
) -> pd.DataFrame:
    labels = _state_labels(seqdata, with_missing=with_missing)
    dss = np.asarray(seqdss(seqdata))
    durations = np.asarray(seqdur(seqdata))
    nbseq = dss.shape[0]
    maxsp = dss.shape[1]

    columns: Dict[str, np.ndarray] = {}
    for label_idx, label in enumerate(labels):
        state_code = label_idx + 1
        max_count = int(np.max(np.sum(dss == state_code, axis=1))) if dss.size else 0
        for spell_idx in range(1, max_count + 1):
            if spell_dur:
                safe = _r_data_frame_name(_spell_field_name(label, "dur", spell_idx))
                columns[safe] = np.zeros(nbseq, dtype=float)
            if spell_age:
                safe = _r_data_frame_name(_spell_field_name(label, "age", spell_idx))
                columns[safe] = np.full(nbseq, np.nan, dtype=float)

    for row in range(nbseq):
        age = 0.0
        spell_counter: Dict[str, int] = {}
        for spell_pos in range(maxsp):
            state = dss[row, spell_pos]
            if _is_dss_padding(state):
                break
            label = _as_state_label(state, seqdata, labels)
            if label is None:
                break
            spell_counter[label] = spell_counter.get(label, 0) + 1
            stnum = spell_counter[label]
            if spell_dur:
                raw = _spell_field_name(label, "dur", stnum)
                if raw not in columns:
                    columns[raw] = np.full(nbseq, np.nan, dtype=float)
                columns[raw][row] = float(durations[row, spell_pos])
            if spell_age:
                raw = _spell_field_name(label, "age", stnum)
                if raw not in columns:
                    columns[raw] = np.full(nbseq, np.nan, dtype=float)
                columns[raw][row] = age
            age += float(durations[row, spell_pos])

    return pd.DataFrame(columns).reset_index(drop=True)


def _sanitize_subsequence_name(subseq) -> str:
    if hasattr(subseq, "to_string"):
        return subseq.to_string().replace(" ", "")
    return str(subseq).replace(" ", "")


def _extract_event_subsequence_properties(
    seqdata: SequenceData,
    *,
    tevent: str,
    method: str,
    pmin_support: float,
    max_k: int,
) -> pd.DataFrame:
    """
    Extract event-subsequence properties from DSS-equivalent event sequences.

    State-to-event conversion emits one event per spell entry (state changes only),
    matching TraMineR ``seqecreate(..., tevent=...)`` on state sequences.
    """
    eseq = EventSequenceData.from_state_sequences(
        seqdata,
        event_representation="transition" if tevent == "transition" else "state",
        use_labels=True,
        weighted=True,
    )
    if pmin_support > 0 and pmin_support <= 1:
        fsub = find_frequent_subsequences(
            eseq,
            min_support_ratio=pmin_support,
            max_k=max_k if max_k > 0 else 3,
        )
    else:
        fsub = find_frequent_subsequences(
            eseq,
            min_support=max(1, int(pmin_support)),
            max_k=max_k if max_k > 0 else 3,
        )

    if len(fsub.subsequences) == 0:
        return pd.DataFrame(index=range(seqdata.seqdata.shape[0]))

    if method == "count":
        values = count_subsequence_occurrences(fsub, counting_method="count")
        columns = [_sanitize_subsequence_name(sub) for sub in fsub.subsequences]
        return pd.DataFrame(values, columns=columns)

    constraint = EventSequenceConstraint()
    n_seq = len(eseq.sequences)
    n_sub = len(fsub.subsequences)
    ages = np.full((n_seq, n_sub), np.nan, dtype=float)
    for j, subseq in enumerate(fsub.subsequences):
        for i, seq in enumerate(eseq.sequences):
            match = _find_first_occurrence(subseq, seq, constraint)
            if match is not None:
                ages[i, j] = match[1]
    columns = [_sanitize_subsequence_name(sub) for sub in fsub.subsequences]
    return pd.DataFrame(ages, columns=columns)


def _extract_complexity_properties(seqdata: SequenceData) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        ici = get_complexity_index(seqdata, silent=True)
        ient = get_within_sequence_entropy(seqdata=seqdata, norm=True)
        turb = get_turbulence(seqdata, norm=False, silent=True)
        trans = get_number_of_transitions(seqdata=seqdata, norm=False)

    out = pd.DataFrame(
        {
            "C": ici.iloc[:, 1].to_numpy(dtype=float),
            "Entropy": ient.iloc[:, 1].to_numpy(dtype=float),
            "Turbulence": turb.iloc[:, 1].to_numpy(dtype=float),
            "Trans.": trans.iloc[:, 1].to_numpy(dtype=float),
        }
    )
    return out.reset_index(drop=True)


def extract_sequence_properties(
    seqdata: SequenceData,
    properties: Sequence[str] = ("state", "duration"),
    other_properties: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    with_missing: bool = True,
    pmin_support: float = 0.05,
    max_k: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract sequence properties used by property-based clustering.

    Mirrors the property extraction stage of WeightedCluster ``seqpropclust``.

    Parameters
    ----------
    seqdata : SequenceData
        Input state sequences.
    properties : sequence of str
        Property names among ``state``, ``spell.age``, ``spell.dur``, ``duration``,
        ``pattern``, ``AFpattern``, ``transition``, ``AFtransition``, ``Complexity``.
    other_properties : DataFrame, optional
        User-defined properties with one row per sequence.
    with_missing : bool, default True
        Include missing states when relevant.
    pmin_support : float, default 0.05
        Minimum support for frequent subsequence mining.
    max_k : int, default -1
        Maximum subsequence length for mining (``-1`` uses the event-sequence default).
    verbose : bool, default True
        Print extraction progress messages like the R function.

    Returns
    -------
    pd.DataFrame
        Combined property matrix with one row per sequence.
    """
    unknown = [name for name in properties if name not in SUPPORTED_PROPERTIES]
    if unknown:
        raise ValueError(f"Unsupported properties: {unknown}")

    nbseq = seqdata.seqdata.shape[0]
    blocks: List[tuple[str, pd.DataFrame]] = []

    if other_properties is not None:
        extra = pd.DataFrame(other_properties).reset_index(drop=True)
        if len(extra) != nbseq:
            raise ValueError("other_properties must have one row per sequence.")
        if verbose:
            print(f" [>] Adding {extra.shape[1]} user defined properties.", end="")
        blocks.append(("other.prop", extra))
        if verbose:
            print()

    if "state" in properties:
        if verbose:
            print(" [>] Extracting 'state' properties...", end="")
        state_block = _extract_state_properties(seqdata)
        blocks.append(("state", state_block))
        if verbose:
            print(f"OK ({state_block.shape[1]} properties extracted)")

    if "spell.dur" in properties or "spell.age" in properties:
        if verbose:
            print(" [>] Extracting 'spell' properties...", end="")
        spell_block = _extract_spell_properties(
            seqdata,
            spell_dur="spell.dur" in properties,
            spell_age="spell.age" in properties,
            with_missing=with_missing,
        )
        blocks.append(("spell", spell_block))
        if verbose:
            print(f"OK ({spell_block.shape[1]} properties extracted)")

    if "duration" in properties:
        if verbose:
            print(" [>] Extracting 'duration' properties...", end="")
        duration_block = _extract_duration_properties(seqdata, with_missing=with_missing)
        blocks.append(("duration", duration_block))
        if verbose:
            print(f"OK ({duration_block.shape[1]} properties extracted)")

    if "transition" in properties:
        if verbose:
            print(" [>] Extracting 'transition' properties...", end="")
        transition_block = _extract_event_subsequence_properties(
            seqdata,
            tevent="transition",
            method="count",
            pmin_support=pmin_support,
            max_k=max_k,
        )
        blocks.append(("transition", transition_block))
        if verbose:
            print(f"OK ({transition_block.shape[1]} properties extracted)")

    if "pattern" in properties:
        if verbose:
            print(" [>] Extracting 'pattern' properties...", end="")
        pattern_block = _extract_event_subsequence_properties(
            seqdata,
            tevent="state",
            method="count",
            pmin_support=pmin_support,
            max_k=max_k,
        )
        blocks.append(("pattern", pattern_block))
        if verbose:
            print(f"OK ({pattern_block.shape[1]} properties extracted)")

    if "AFtransition" in properties:
        if verbose:
            print(" [>] Extracting 'AFtransition' properties...", end="")
        aftransition_block = _extract_event_subsequence_properties(
            seqdata,
            tevent="transition",
            method="age",
            pmin_support=pmin_support,
            max_k=max_k,
        )
        blocks.append(("AFtransition", aftransition_block))
        if verbose:
            print(f"OK ({aftransition_block.shape[1]} properties extracted)")

    if "AFpattern" in properties:
        if verbose:
            print(" [>] Extracting 'AFpattern' properties...", end="")
        afpattern_block = _extract_event_subsequence_properties(
            seqdata,
            tevent="state",
            method="age",
            pmin_support=pmin_support,
            max_k=max_k,
        )
        blocks.append(("AFpattern", afpattern_block))
        if verbose:
            print(f"OK ({afpattern_block.shape[1]} properties extracted)")

    if "Complexity" in properties:
        if verbose:
            print(" [>] Extracting 'Complexity' properties...", end="")
        complexity_block = _extract_complexity_properties(seqdata)
        blocks.append(("Complexity", complexity_block))
        if verbose:
            print(f"OK ({complexity_block.shape[1]} properties extracted)")

    if not blocks:
        raise ValueError("No properties were extracted.")

    prefixed_blocks = [
        block if block_name == "other.prop" else _prefix_block_columns(block, block_name)
        for block_name, block in blocks
    ]
    combined = pd.concat(prefixed_blocks, axis=1)
    if len(combined) != nbseq:
        raise ValueError("Feature extraction failed: row count mismatch.")
    if verbose:
        print(f" [>] {combined.shape[1]} properties extracted.")
    return combined.reset_index(drop=True)
