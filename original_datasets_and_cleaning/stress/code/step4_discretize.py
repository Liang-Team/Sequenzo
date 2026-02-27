"""
@Author  : Yuqi Liang 梁彧祺
@File    : step4_discretize.py
@Time    : 27/02/2026 16:16
@Desc    : 
Step 4: Discretize continuous PSS into stress states (L / M / H).
Thresholds: L < 1.5, M in [1.5, 2.5), H >= 2.5.
"""

import pandas as pd
import json

from config import COL_PSS, COL_STRESS_STATE, DISCRETIZATION, OUTPUT_THRESHOLDS


def pss_to_state(pss, boundaries=None, labels=None):
    """
    Map a single PSS value to state label.
    boundaries = [1.5, 2.5] -> L if pss < 1.5, M if 1.5 <= pss < 2.5, H if pss >= 2.5.
    """
    if boundaries is None:
        boundaries = DISCRETIZATION["boundaries"]
    if labels is None:
        labels = DISCRETIZATION["labels"]
    if pd.isna(pss):
        return None
    if pss < boundaries[0]:
        return labels[0]
    if pss < boundaries[1]:
        return labels[1]
    return labels[2]


def add_stress_state(df):
    """Add column stress_state (L, M, H) from PSS using config thresholds."""
    df = df.copy()
    b = DISCRETIZATION["boundaries"]
    lbl = DISCRETIZATION["labels"]
    df[COL_STRESS_STATE] = df[COL_PSS].map(lambda x: pss_to_state(x, b, lbl))
    return df


def write_thresholds_json(path=OUTPUT_THRESHOLDS):
    """Write discretization method and thresholds to JSON for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(DISCRETIZATION, f, indent=2)


def run_step4(df):
    """
    Add stress_state column and write stress_thresholds.json.
    Returns dataframe with stress_state column.
    """
    df = add_stress_state(df)
    write_thresholds_json()
    return df
