"""Resolve weighted permutation mode for discrepancy analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np

VALID_WEIGHT_PERMUTATION = frozenset({"none", "replicate", "diss", "group"})


def resolve_weight_permutation(
    weights: Optional[np.ndarray],
    weight_permutation: Optional[str],
) -> str:
    """Choose a TraMineR-compatible weight permutation mode.

    Unweighted analyses use ``"none"``. When weights are supplied and the caller
    does not override the mode, default to ``"replicate"`` to match TraMineR
    ``dissassoc()``, ``disstree()``, and ``seqtree()``.
    """
    if weights is None:
        return "none"
    if weight_permutation is None:
        return "replicate"
    if weight_permutation not in VALID_WEIGHT_PERMUTATION:
        raise ValueError(
            "[!] 'weight_permutation' must be one of "
            f"{sorted(VALID_WEIGHT_PERMUTATION)}. Got {weight_permutation!r}."
        )
    return weight_permutation
