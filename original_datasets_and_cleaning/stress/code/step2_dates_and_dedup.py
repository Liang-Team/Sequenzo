"""
@Author  : Yuqi Liang 梁彧祺
@File    : step2_dates_and_dedup.py
@Time    : 26/02/2026 11:17
@Desc    : 
Step 2: Deduplicate by (ID, StartDate).
For each (ID, StartDate), keep one row: prefer the row with non-missing PSS;
if multiple have non-missing PSS, keep the first after stable sort by StartDate.
"""

import pandas as pd

from config import COL_ID, COL_START_DATE, COL_PSS


def deduplicate(df):
    """
    One row per (ID, StartDate). Keep row with non-missing PSS when possible;
    otherwise keep first. Use stable sort: StartDate ascending, then prefer non-null PSS.
    """
    df = df.copy()
    # Restore original order column for stable sort (index order)
    df["_order"] = range(len(df))
    # Prefer rows where PSS is not null (True > False when sorting descending)
    df["_has_pss"] = df[COL_PSS].notna()
    df = df.sort_values(
        [COL_ID, COL_START_DATE, "_has_pss", "_order"],
        ascending=[True, True, False, True],
    )
    df = df.drop_duplicates(subset=[COL_ID, COL_START_DATE], keep="first")
    df = df.drop(columns=["_order", "_has_pss"])
    return df


def run_step2(df, report):
    """
    Deduplicate by (ID, StartDate). Update report with rows_dropped_duplicates.
    Returns the dataframe after step 2.
    """
    before = len(df)
    df = deduplicate(df)
    report["rows_dropped_duplicates"] = before - len(df)
    return df
