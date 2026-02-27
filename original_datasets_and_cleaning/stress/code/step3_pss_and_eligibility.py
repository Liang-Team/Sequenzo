"""
@Author  : Yuqi Liang 梁彧祺
@File    : step3_pss_and_eligibility.py
@Time    : 26/02/2026 13:43
@Desc    : 
Step 3: PSS validity (numeric, in range 0–4), then eligibility (>= 8 observations per ID)
and truncation to first 8 observations per ID (by StartDate).
"""

import pandas as pd
import numpy as np

from config import (
    COL_ID,
    COL_START_DATE,
    COL_PSS,
    COL_COHORT,
    COL_T,
    PSS_MIN,
    PSS_MAX,
    MIN_WEEKS,
)


def pss_to_numeric(df):
    """Convert PSS to numeric; drop rows where PSS is missing or non-numeric."""
    df = df.copy()
    df[COL_PSS] = pd.to_numeric(df[COL_PSS], errors="coerce")
    before = len(df)
    df = df[df[COL_PSS].notna()].copy()
    dropped = before - len(df)
    return df, dropped


def pss_range_filter(df):
    """Drop rows with PSS outside [PSS_MIN, PSS_MAX] (strict option A)."""
    before = len(df)
    df = df[(df[COL_PSS] >= PSS_MIN) & (df[COL_PSS] <= PSS_MAX)].copy()
    dropped = before - len(df)
    return df, dropped


def keep_ids_with_at_least_n_weeks(df, n=MIN_WEEKS):
    """
    Keep only IDs that have at least n valid observations.
    Returns the filtered dataframe and count stats for the report.
    """
    counts = df.groupby(COL_ID).size()
    ids_keep = counts[counts >= n].index
    df_keep = df[df[COL_ID].isin(ids_keep)].copy()

    n_ids_before = counts.shape[0]
    n_ids_after = len(ids_keep)
    n_dropped = n_ids_before - n_ids_after

    stats = {
        "ids_before_8_filter": n_ids_before,
        "ids_after_8_filter": n_ids_after,
        "ids_dropped_insufficient_weeks": n_dropped,
        "min_obs_per_id": int(counts.min()) if len(counts) else 0,
        "median_obs_per_id": int(np.median(counts)) if len(counts) else 0,
        "max_obs_per_id": int(counts.max()) if len(counts) else 0,
    }
    return df_keep, stats


def truncate_to_first_n_per_id(df, n=MIN_WEEKS):
    """
    For each ID, sort by StartDate ascending and take the first n rows.
    Assign position t = 1..n.
    """
    df = df.sort_values([COL_ID, COL_START_DATE]).copy()
    df["_rank"] = df.groupby(COL_ID)[COL_START_DATE].rank(method="first", ascending=True)
    df = df[df["_rank"] <= n].copy()
    df[COL_T] = df["_rank"].astype(int)
    df = df.drop(columns=["_rank"])
    return df


def run_step3(df, report):
    """
    Apply PSS validity, eligibility (>= 8 obs), and truncate to first 8 per ID.
    Updates report with PSS drop counts and eligibility stats.
    Returns dataframe with columns ID, cohort, StartDate, PSS, t (1..8).
    """
    # PSS numeric and drop missing
    df, dropped_pss_missing = pss_to_numeric(df)
    # PSS range (strict drop)
    df, dropped_pss_range = pss_range_filter(df)
    report["rows_dropped_invalid_pss"] = dropped_pss_missing + dropped_pss_range

    # Eligibility: keep IDs with >= MIN_WEEKS observations
    df, eligibility_stats = keep_ids_with_at_least_n_weeks(df, MIN_WEEKS)
    report.update(eligibility_stats)

    # Truncate to first 8 per ID and add position t
    df = truncate_to_first_n_per_id(df, MIN_WEEKS)

    report["final_N"] = df[COL_ID].nunique()
    return df
