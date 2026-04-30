"""
@Author  : Yuqi Liang 梁彧祺
@File    : sequence_operations.py
@Time    : 15/04/2026 11:12
@Desc    : 
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData


def _is_missing_marker(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"missing", "nan", "na", "none"}
    return False


def _as_dataframe(data: Any, var: list[str] | None = None) -> pd.DataFrame:
    """
    Convert input into a sequence DataFrame.
    Supports SequenceData, pandas DataFrame, numpy arrays, and list-of-lists.
    """
    if isinstance(data, SequenceData):
        arr = data.values
        df = pd.DataFrame(arr, index=data.ids, columns=data.time)
        return df

    if isinstance(data, pd.DataFrame):
        if var is None:
            return data.copy()
        return data[var].copy()

    arr = np.asarray(data, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = [f"T{i+1}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def seqconc(data, var: list[str] | None = None, sep: str = "-", vname: str = "Sequence", void=np.nan) -> pd.DataFrame:
    """Concatenate each row of sequence states/events into one string."""
    df = _as_dataframe(data, var=var)

    def _join_row(row: pd.Series) -> str:
        vals = []
        for x in row.tolist():
            if pd.isna(void):
                if pd.isna(x):
                    continue
            elif void is not None and x == void:
                continue
            vals.append(str(x))
        return sep.join(vals)

    out = df.apply(_join_row, axis=1)
    return out.to_frame(name=vname)


def seqdecomp(data, var: list[str] | None = None, sep: str = "-", miss: str = "NA", vnames: list[str] | None = None) -> pd.DataFrame:
    """Decompose concatenated sequence strings into one column per state/event."""
    df = _as_dataframe(data, var=var)
    seq_strings = df.iloc[:, 0].astype(str).tolist() if df.shape[1] == 1 else df.astype(str).agg("".join, axis=1).tolist()

    split_rows = [s.split(sep) if len(s) else [] for s in seq_strings]
    max_len = max((len(r) for r in split_rows), default=0)
    out = []
    for r in split_rows:
        row = [None if x == miss else x for x in r]
        if len(row) < max_len:
            row.extend([None] * (max_len - len(row)))
        out.append(row)

    cols = vnames if vnames is not None else [f"[{i+1}]" for i in range(max_len)]
    return pd.DataFrame(out, index=df.index, columns=cols)


def seqsep(seqdata: Iterable[str], sl: int = 1, sep: str = "-") -> list[str]:
    """Split fixed-width sequence strings and insert a separator."""
    out = []
    for i, s in enumerate(seqdata):
        s = str(s)
        n = len(s)
        if n % sl != 0:
            raise ValueError(f"Number of characters does not match sequence length for element {i}.")
        parts = [s[j:j + sl] for j in range(0, n, sl)] if n > 0 else []
        out.append(sep.join(parts))
    return out


def seqshift(seq: Iterable[Any], nbshift: int):
    """Shift sequence as in TraMineR seqshift: keep first nbshift at the end, fill front with NA."""
    seq = list(seq)
    seql = len(seq)
    if nbshift < 0 or nbshift > seql:
        raise ValueError("nbshift must be in [0, len(seq)].")
    return [np.nan] * (seql - nbshift) + seq[:nbshift]


def seqrecode(
    seqdata,
    recodes: dict[str, Iterable[Any]],
    otherwise=None,
    labels: list[str] | None = None,
    cpal: list[str] | None = None,
):
    """
    Recode a sequence object.

    For SequenceData input, this returns a new SequenceData object to stay close
    to TraMineR behavior.
    For other inputs, this returns a pandas DataFrame.
    """
    if isinstance(seqdata, SequenceData):
        # Decode internal integer representation back to states.
        code_to_state = {i + 1: s for i, s in enumerate(seqdata.states)}
        decoded = pd.DataFrame(seqdata.values, index=seqdata.ids, columns=seqdata.time).applymap(
            lambda x: code_to_state.get(int(x), np.nan) if pd.notna(x) else np.nan
        )
        id_col = seqdata.id_col or "id"
        decoded_with_id = decoded.copy()
        decoded_with_id.insert(0, id_col, seqdata.ids)
        recoded_df = seqrecode(decoded_with_id, recodes, otherwise=otherwise)

        # Build output state list in TraMineR-like order.
        used = [v for vv in recodes.values() for v in vv]
        out_states = list(recodes.keys())
        if otherwise is not None:
            out_states.append(otherwise)
        else:
            for s in seqdata.states:
                if s not in used and not _is_missing_marker(s):
                    out_states.append(s)

        # Keep deterministic unique order.
        seen = set()
        out_states = [s for s in out_states if not (s in seen or seen.add(s))]
        out_time = seqdata.time
        return SequenceData(
            data=recoded_df,
            time=out_time,
            states=out_states,
            labels=labels if labels is not None else [str(s) for s in out_states],
            id_col=id_col,
            weights=seqdata.weights,
            custom_colors=cpal,
        )

    df = _as_dataframe(seqdata)
    recoded = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    reverse = {}
    for new_code, old_codes in recodes.items():
        for old in old_codes:
            reverse[old] = new_code

    for col in df.columns:
        vals = []
        for v in df[col].tolist():
            if v in reverse:
                vals.append(reverse[v])
            elif otherwise is None:
                vals.append(v)
            else:
                vals.append(otherwise)
        recoded[col] = vals
    return recoded


def seqasnum(seqdata, with_missing: bool = False) -> np.ndarray:
    """
    Convert sequence states to numeric coding matrix.

    TraMineR-style coding starts at 0 based on alphabet order.
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("seqasnum currently expects a SequenceData object.")

    values = seqdata.values.astype(float)  # 1..K integer coding in SequenceData
    decoded_states = list(seqdata.states)
    include_idx = []
    for idx, st in enumerate(decoded_states, start=1):
        if with_missing or (not _is_missing_marker(st)):
            include_idx.append(idx)

    out = np.full(values.shape, np.nan, dtype=float)
    for new_code, old_code in enumerate(include_idx):
        out[values == old_code] = new_code

    return out
