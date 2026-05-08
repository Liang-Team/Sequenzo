"""
@Desc: Basic interpretation utilities for selected features.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def interpret_selected_features(selection_result: Dict[str, Any]) -> pd.DataFrame:
    names = selection_result.get("selected_feature_names", [])
    indices = selection_result.get("selected_indices", [])
    hit_counts = selection_result.get("hit_counts")

    if hit_counts is not None and len(indices) > 0:
        selected_hits = [int(hit_counts[i]) for i in indices]
    else:
        selected_hits = [None] * len(indices)

    return pd.DataFrame(
        {
            "feature": names,
            "index": indices,
            "hit_count": selected_hits,
        }
    ).sort_values(["hit_count", "feature"], ascending=[False, True], na_position="last")


__all__ = ["interpret_selected_features"]
