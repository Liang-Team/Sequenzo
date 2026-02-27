"""
@Author  : Yuqi Liang 梁彧祺
@File    : step5_export.py
@Time    : 27/02/2026 17:30
@Desc    : 
Step 5: Export long and wide sequence CSVs.
- Long: one row per (ID, t) with ID, cohort, StartDate, t, PSS, stress_state.
- Wide: one row per ID with ID, cohort, s1..s8 (and optional pss1..pss8, start1..start8).
"""

import pandas as pd

from config import (
    COL_ID,
    COL_COHORT,
    COL_START_DATE,
    COL_T,
    COL_PSS,
    COL_STRESS_STATE,
    OUTPUT_LONG,
    OUTPUT_WIDE,
)


def export_long(df, path=OUTPUT_LONG):
    """
    Write stress_sequence_long.csv: ID, cohort, StartDate, t, PSS, stress_state.
    One row per (ID, t), t in 1..8.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[[COL_ID, COL_COHORT, COL_START_DATE, COL_T, COL_PSS, COL_STRESS_STATE]].copy()
    out = out.sort_values([COL_ID, COL_T])
    out.to_csv(path, index=False)


def export_wide(df, path=OUTPUT_WIDE, include_pss=True, include_dates=True):
    """
    Pivot to one row per ID with s1..s8 (and optionally pss1..pss8, start1..start8).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # One row per ID: take cohort from first row
    ids = df[COL_ID].unique()
    cohort_map = df.groupby(COL_ID)[COL_COHORT].first()

    rows = []
    for id_ in ids:
        sub = df[df[COL_ID] == id_].sort_values(COL_T)
        row = {COL_ID: id_, COL_COHORT: cohort_map[id_]}
        for t in range(1, 9):
            r = sub[sub[COL_T] == t]
            if len(r) == 0:
                row[f"s{t}"] = None
                if include_pss:
                    row[f"pss{t}"] = None
                if include_dates:
                    row[f"start{t}"] = None
            else:
                r = r.iloc[0]
                row[f"s{t}"] = r[COL_STRESS_STATE]
                if include_pss:
                    row[f"pss{t}"] = r[COL_PSS]
                if include_dates:
                    row[f"start{t}"] = r[COL_START_DATE]
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)


def run_step5(df):
    """Write stress_sequence_long.csv and stress_sequence_wide.csv."""
    export_long(df)
    export_wide(df, include_pss=True, include_dates=True)
