"""
@Author  : Yuqi Liang 梁彧祺
@File    : time_binning_utils.py
@Time    : 18/03/2026 10:12
@Desc    :
    Time utilities for numeric coercion and equal-width binning.
"""

from __future__ import annotations

import re
from typing import Iterable, List


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

