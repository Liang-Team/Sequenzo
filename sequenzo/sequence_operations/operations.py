"""
@Author  : Yuqi Liang 梁彧祺
@File    : sequence_operations.py
@Time    : 15/04/2026 11:12
@Desc    : 
"""

from __future__ import annotations

from dataclasses import dataclass
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


def concatenate_sequences(
    data, var: list[str] | None = None, sep: str = "-", vname: str = "Sequence", void=np.nan
) -> pd.DataFrame:
    """
    Equivalent to TraMineR::seqconc().

    Concatenate each row of sequence states/events into one string.
    """
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


def decompose_concatenated_sequences(
    data, var: list[str] | None = None, sep: str = "-", miss: str = "NA", vnames: list[str] | None = None
) -> pd.DataFrame:
    """
    Equivalent to TraMineR::seqdecomp().

    Decompose concatenated sequence strings into one column per state/event.
    """
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


def split_fixed_width_sequences(seqdata: Iterable[str], sl: int = 1, sep: str = "-") -> list[str]:
    """
    Equivalent to TraMineR::seqsep().

    Split fixed-width sequence strings and insert a separator.
    """
    out = []
    for i, s in enumerate(seqdata):
        s = str(s)
        n = len(s)
        if n % sl != 0:
            raise ValueError(f"Number of characters does not match sequence length for element {i}.")
        parts = [s[j:j + sl] for j in range(0, n, sl)] if n > 0 else []
        out.append(sep.join(parts))
    return out


def shift_sequence_with_missing_padding(seq: Iterable[Any], nbshift: int):
    """
    Equivalent to TraMineR::seqshift().

    Shift sequence by keeping the first nbshift values at the end and padding the front with NA.
    """
    seq = list(seq)
    seql = len(seq)
    if nbshift < 0 or nbshift > seql:
        raise ValueError("nbshift must be in [0, len(seq)].")
    return [np.nan] * (seql - nbshift) + seq[:nbshift]


def recode_sequence_states(
    seqdata,
    recodes: dict[str, Iterable[Any]],
    otherwise=None,
    labels: list[str] | None = None,
    cpal: list[str] | None = None,
):
    """
    Equivalent to TraMineR::seqrecode().

    Recode sequence states using user-defined mapping rules.

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
        recoded_df = recode_sequence_states(decoded_with_id, recodes, otherwise=otherwise)

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


def convert_sequences_to_numeric_matrix(seqdata, with_missing: bool = False) -> np.ndarray:
    """
    Equivalent to TraMineR::seqasnum().

    Convert sequence states to a numeric coding matrix where coding starts at 0.
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("convert_sequences_to_numeric_matrix currently expects a SequenceData object.")

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


def _validate_same_alphabet(seq1: SequenceData, seq2: SequenceData) -> None:
    if len(seq1.alphabet) != len(seq2.alphabet) or any(a != b for a, b in zip(seq1.alphabet, seq2.alphabet)):
        raise ValueError("[!] The alphabet of both sequences have to be same.")


def longest_common_prefix_length(seq1: SequenceData, seq2: SequenceData, index1: int = 0, index2: int = 0) -> int:
    """
    Equivalent to TraMineR::seqLLCP().

    Length of the longest common prefix between two sequences.
    """
    if not isinstance(seq1, SequenceData) or not isinstance(seq2, SequenceData):
        raise TypeError("[!] sequences must be sequence objects")
    _validate_same_alphabet(seq1, seq2)
    if not isinstance(index1, int) or not isinstance(index2, int):
        raise TypeError("[!] 'index1' and 'index2' must be int.")
    if index1 < 0 or index1 >= seq1.seqdata.shape[0] or index2 < 0 or index2 >= seq2.seqdata.shape[0]:
        raise ValueError("[!] 'seq1' or 'seq2' has no such index.")

    a = seq1.seqdata.iloc[index1].to_numpy()
    b = seq2.seqdata.iloc[index2].to_numpy()
    boundary = min(len(a), len(b))
    length = 0
    while length < boundary and a[length] == b[length]:
        length += 1
    return length


def longest_common_subsequence_length(seq1: SequenceData, seq2: SequenceData, index1: int = 0, index2: int = 0) -> int:
    """
    Equivalent to TraMineR::seqLLCS().

    Length of the longest common subsequence between two sequences.
    """
    if not isinstance(seq1, SequenceData) or not isinstance(seq2, SequenceData):
        raise TypeError("[!] sequences must be sequence objects")
    _validate_same_alphabet(seq1, seq2)
    if not isinstance(index1, int) or not isinstance(index2, int):
        raise TypeError("[!] 'index1' and 'index2' must be int.")
    if index1 < 0 or index1 >= seq1.seqdata.shape[0] or index2 < 0 or index2 >= seq2.seqdata.shape[0]:
        raise ValueError("[!] 'seq1' or 'seq2' has no such index.")

    a = seq1.seqdata.iloc[index1].to_numpy()
    b = seq2.seqdata.iloc[index2].to_numpy()
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[n, m])


def find_sequence_occurrences(x: SequenceData, y: SequenceData) -> list[int]:
    """
    Equivalent to TraMineR::seqfind().

    Find all occurrences in `y` of each sequence from `x`.

    Returns 1-based indices, matching TraMineR `which(...)` semantics.
    """
    if not isinstance(x, SequenceData):
        raise TypeError("x is not a sequence object, use `SequenceData` to create one")
    if not isinstance(y, SequenceData):
        raise TypeError("y is not a sequence object, use `SequenceData` to create one")

    xconc = concatenate_sequences(x).iloc[:, 0].tolist()
    yconc = concatenate_sequences(y).iloc[:, 0].tolist()

    occ: list[int] = []
    for seq in xconc:
        occ.extend([idx + 1 for idx, target in enumerate(yconc) if target == seq])
    return occ


@dataclass
class SeqAlignResult:
    operation: list[str]
    seq1: list[str]
    seq2: list[str]
    cost: list[float]
    opmatrix: np.ndarray
    costmatrix: np.ndarray
    stsseq: SequenceData

    def __str__(self) -> str:
        total = float(np.sum(self.cost))
        return f"SeqAlignResult(steps={len(self.operation)}, om_distance={total:g})"


def _state_label_from_code(seqdata: SequenceData, code: int) -> str:
    return str(seqdata.inverse_state_mapping.get(int(code), code))


def _lookup_substitution_cost(sm: Any, seqdata: SequenceData, c1: int, c2: int) -> float:
    if c1 == c2:
        return 0.0
    if isinstance(sm, pd.DataFrame):
        if c1 in sm.index and c2 in sm.columns:
            return float(sm.loc[c1, c2])
        l1 = _state_label_from_code(seqdata, c1)
        l2 = _state_label_from_code(seqdata, c2)
        if l1 in sm.index and l2 in sm.columns:
            return float(sm.loc[l1, l2])
        raise KeyError(f"Substitution matrix does not contain keys for states {l1!r} and {l2!r}.")
    arr = np.asarray(sm)
    if arr.ndim != 2:
        raise ValueError("`sm` must be a 2D substitution matrix.")
    if c1 < arr.shape[0] and c2 < arr.shape[1]:
        return float(arr[c1, c2])
    return float(arr[c1 - 1, c2 - 1])


def pairwise_sequence_alignment(
    seqdata: SequenceData,
    indices: Iterable[int],
    indel: float = 1.0,
    sm: Any = None,
    with_missing: bool = False,
) -> SeqAlignResult:
    """
    Equivalent to TraMineR::seqalign().

    Pairwise sequence alignment details for two sequences.

    This follows TraMineR `seqalign()` dynamic-programming behavior and tie-breaking:
    substitution/match first, then insertion, then deletion.
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("`seqdata` must be a SequenceData object.")
    if sm is None:
        raise ValueError("`sm` (substitution matrix) must be provided.")

    idx = list(indices)
    if len(idx) != 2:
        raise ValueError("`indices` must contain exactly two sequence indices.")
    i1, i2 = idx
    if min(i1, i2) < 0 or max(i1, i2) >= seqdata.n_sequences:
        raise ValueError("`indices` out of range.")

    s1 = seqdata.seqdata.iloc[i1].to_numpy()
    s2 = seqdata.seqdata.iloc[i2].to_numpy()
    l1, l2 = len(s1), len(s2)

    leven = np.zeros((l1 + 1, l2 + 1), dtype=float)
    operation = np.full((l1 + 1, l2 + 1), "", dtype=object)
    leven[:, 0] = np.arange(l1 + 1) * float(indel)
    leven[0, :] = np.arange(l2 + 1) * float(indel)
    operation[:, 0] = "D"
    operation[0, :] = "I"

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            c1, c2 = int(s1[i - 1]), int(s2[j - 1])
            sub_cost = _lookup_substitution_cost(sm, seqdata, c1, c2)
            sub_val = leven[i - 1, j - 1] + sub_cost
            ins_val = leven[i, j - 1] + float(indel)
            del_val = leven[i - 1, j] + float(indel)
            best = min(sub_val, ins_val, del_val)
            leven[i, j] = best
            if best == sub_val:
                operation[i, j] = "E" if sub_cost == 0 else "S"
            elif best == ins_val:
                operation[i, j] = "I"
            else:
                operation[i, j] = "D"

    i, j = l1, l2
    operations: list[str] = []
    seq1c: list[str] = []
    seq2c: list[str] = []
    costs: list[float] = []
    while i > 0 or j > 0:
        op = str(operation[i, j])
        old_cost = float(leven[i, j])
        operations.append(op)
        if op in {"S", "E"}:
            seq1c.append(_state_label_from_code(seqdata, int(s1[i - 1])))
            seq2c.append(_state_label_from_code(seqdata, int(s2[j - 1])))
            i -= 1
            j -= 1
        elif op == "I":
            seq1c.append("-")
            seq2c.append(_state_label_from_code(seqdata, int(s2[j - 1])))
            j -= 1
        elif op == "D":
            seq1c.append(_state_label_from_code(seqdata, int(s1[i - 1])))
            seq2c.append("-")
            i -= 1
        else:
            raise RuntimeError(f"Unexpected alignment operation {op!r}.")
        costs.append(old_cost - float(leven[i, j]))

    operations.reverse()
    seq1c.reverse()
    seq2c.reverse()
    costs.reverse()
    _ = with_missing  # Kept for API compatibility with TraMineR signature.
    return SeqAlignResult(
        operation=operations,
        seq1=seq1c,
        seq2=seq2c,
        cost=costs,
        opmatrix=operation,
        costmatrix=leven,
        stsseq=seqdata,
    )
