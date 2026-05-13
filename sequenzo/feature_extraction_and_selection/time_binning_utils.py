"""
@Author  : Yuqi Liang 梁彧祺
@File    : time_binning_utils.py
@Time    : 18/03/2026 10:12
@Desc    :
    Time utilities for numeric coercion and equal-width binning.

    Important: ``timing_bin_width`` is always expressed in the **same unit as**
    ``seqdata.time`` labels (months, years, or position indices). A value of
    ``12.0`` means twelve time-label units per bin—not necessarily twelve
    calendar months unless your time grid is monthly.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Literal

TimeUnitHint = Literal["month", "year", "same_as_labels"]


def coerce_numeric_time_labels(time_labels: Iterable) -> List[float]:
    out: List[float] = []
    for x in time_labels:
        if x is None:
            raise ValueError("Time labels cannot contain None.")
        try:
            out.append(float(x))
            continue
        except (TypeError, ValueError):
            pass

        sx = str(x)
        m = re.search(r"-?\d+(\.\d+)?", sx)
        if not m:
            raise ValueError(f"Cannot coerce time label {x!r} to numeric.")
        out.append(float(m.group(0)))
    return out


def make_equal_width_bins(
    min_value: float,
    max_value: float,
    bin_width: float,
    *,
    include_left: bool = True,
) -> List[tuple[float, float]]:
    """
    Build half-open bins ``[a, b)`` from ``min_value`` up to ``max_value``.

    Parameters
    ----------
    min_value, max_value
        Range of the observed time labels (same unit as ``seqdata.time``).
    bin_width
        Width of each bin in **that same unit**. For monthly position indices
        ``1..172``, ``bin_width=12`` yields roughly one-year bins. For age
        labels ``15..30``, use ``bin_width=1`` for one-year bins—not ``12``.
    include_left
        Reserved for API symmetry with ``in_bin``; bins are always ``[a, b)``.
    """
    del include_left  # bins are always [start, end) via in_bin
    if bin_width <= 0:
        raise ValueError("bin_width must be > 0.")
    if max_value < min_value:
        raise ValueError("max_value must be >= min_value.")

    bins: List[tuple[float, float]] = []
    a = min_value
    while a <= max_value + 1e-12:
        b = a + bin_width
        bins.append((a, b))
        a = b
    return bins


def in_bin(value: float, start: float, end: float, *, include_left: bool = True) -> bool:
    if include_left:
        return (value >= start) and (value < end)
    return (value > start) and (value <= end)


def suggest_timing_bin_width(time_unit_hint: TimeUnitHint = "same_as_labels") -> float:
    """
    Suggested default bin width for common FES setups.

    - ``month``: monthly position grid (e.g. TREE month indices) → 12 units per bin
    - ``year``: yearly age labels (e.g. 15, 16, …) → 1 unit per bin
    - ``same_as_labels``: no transformation; caller sets ``timing_bin_width`` explicitly
    """
    if time_unit_hint == "month":
        return 12.0
    if time_unit_hint == "year":
        return 1.0
    raise ValueError(
        "time_unit_hint='same_as_labels' has no default bin width; "
        "set timing_bin_width to match your seqdata.time unit."
    )
