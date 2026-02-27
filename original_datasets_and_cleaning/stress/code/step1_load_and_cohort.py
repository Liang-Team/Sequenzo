"""
@Author  : Yuqi Liang 梁彧祺
@File    : step1_load_and_cohort.py
@Time    : 26/02/2026 10:27
@Desc    : 
Step 1: Load the raw CSV, validate required columns, parse dates, and filter by cohort.
Only Fall and Spring cohorts are kept; Summer (Q...) and others are dropped.
"""

import pandas as pd

from config import (
    INPUT_CSV,
    COL_START_DATE,
    COL_ID,
    COL_PSS,
    COL_COHORT,
    COHORT_PREFIXES,
)


def load_raw(csv_path=INPUT_CSV):
    """
    Load the raw CSV. Do not parse dates yet so we can count total rows first.
    """
    df = pd.read_csv(csv_path)
    return df


def validate_columns(df):
    """
    Check that required columns exist. Raises if any are missing.
    """
    required = [COL_START_DATE, COL_ID, COL_PSS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    return df


def add_cohort(df):
    """
    Add a 'cohort' column based on ID prefix (Fall or Spring).
    IDs not starting with Fall/Spring get cohort = None (will be dropped).
    """
    def _cohort(id_val):
        if pd.isna(id_val):
            return None
        s = str(id_val).strip()
        for prefix in COHORT_PREFIXES:
            if s.startswith(prefix):
                return prefix
        return None

    df = df.copy()
    df[COL_COHORT] = df[COL_ID].map(_cohort)
    return df


def filter_cohort(df):
    """
    Keep only rows where cohort is Fall or Spring. Drop the rest.
    """
    before = len(df)
    df = df[df[COL_COHORT].notna()].copy()
    dropped = before - len(df)
    return df, dropped


def parse_dates(df):
    """
    Convert StartDate to datetime. Rows that fail to parse are dropped.
    Returns (filtered_df, n_dropped_invalid_date).
    """
    before = len(df)
    df = df.copy()
    df[COL_START_DATE] = pd.to_datetime(df[COL_START_DATE], errors="coerce")
    df = df[df[COL_START_DATE].notna()].copy()
    dropped = before - len(df)
    return df, dropped


def run_step1(report):
    """
    Run full step 1: load, validate, add cohort, filter cohort, parse dates.
    Updates report with: total_rows_input, rows_dropped_invalid_date, rows_dropped_cohort.
    Returns the dataframe after step 1.
    """
    df = load_raw()
    report["total_rows_input"] = len(df)

    df = validate_columns(df)
    df = add_cohort(df)

    df, dropped_cohort = filter_cohort(df)
    report["rows_dropped_cohort"] = dropped_cohort

    df, dropped_date = parse_dates(df)
    report["rows_dropped_invalid_date"] = dropped_date

    return df
