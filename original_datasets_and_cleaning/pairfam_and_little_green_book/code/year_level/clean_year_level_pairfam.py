"""
@Author  : Yuqi Liang 梁彧祺
@File    : split_and_preprocess_pairfam.py
@Time    : 2026/1/30 15:17
@Desc    : 
Clean and preprocess Pairfam year-level sequence data.

This script reads the year-level family and activity sequence files, adds a new
random ID column (1-based, independent of month_level IDs), and renames time
columns to year indices (1, 2, 3, ...).

Important (state definition):
    For each year, the state recorded is the state in the FIRST MONTH of that
    year, not the state that lasted the longest within the year. For example,
    the value in the "year 2" column is the state at month 13 (start of year 2),
    not the modal or longest-held state across months 13–24.

Column renaming:
    - Activity: activity1 -> 1, activity13 -> 2, activity25 -> 3, ... (each
      column represents one year; the number is the year index).
    - Family: family1 -> 1, family13 -> 2, family25 -> 3, ... (same convention).

IDs:
    New random IDs starting from 1 are assigned so that year-level IDs are
    not aligned with month_level (different units / aggregation), and each
    row gets a unique integer id in {1, ..., n} in random order.
"""

import os
import re
from typing import Optional

import numpy as np
import pandas as pd


# Default paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "data_sources", "year_level")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR

# Input file names
ACTIVITY_INPUT = "mc.act.year.seq.now.csv"
FAMILY_INPUT = "mc.fam.year.seq.now.csv"
# Output file names
ACTIVITY_OUTPUT = "pairfam_activity_by_year.csv"
FAMILY_OUTPUT = "pairfam_family_by_year.csv"

# Random seed for reproducible "random" IDs (change to get different mapping)
RANDOM_ID_SEED = 20260130


def _month_col_to_year_index(col_name: str, prefix: str) -> Optional[int]:
    """
    Map a column name like 'activity13' or 'family25' to a 1-based year index.

    activity1 -> 1, activity13 -> 2, activity25 -> 3, etc.
    Same for family* columns.

    Parameters
    ----------
    col_name : str
        Column name (e.g. 'activity13', 'family1').
    prefix : str
        Expected prefix ('activity' or 'family').

    Returns
    -------
    int or None
        Year index (1, 2, 3, ...) or None if column should not be renamed.
    """
    if not col_name.startswith(prefix):
        return None
    match = re.search(r"\d+", col_name)
    if not match:
        return None
    month_1based = int(match.group(0))
    # Month 1 -> year 1, months 13-24 -> year 2, etc.
    year_index = (month_1based - 1) // 12 + 1
    return year_index


def _rename_time_columns_to_years(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Rename columns like activity1, activity13, ... to 1, 2, ... (year indices).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns named e.g. activity1, activity13, ...
    prefix : str
        'activity' or 'family'.

    Returns
    -------
    pd.DataFrame
        New DataFrame with renamed columns (prefix columns -> 1, 2, 3, ...).
    """
    new_names = {}
    for col in df.columns:
        year_idx = _month_col_to_year_index(col, prefix)
        if year_idx is not None:
            new_names[col] = str(year_idx)
    out = df.rename(columns=new_names)
    return out


def _add_random_ids(df: pd.DataFrame, seed: int = RANDOM_ID_SEED) -> pd.DataFrame:
    """
    Add an 'id' column with a random permutation of 1..n.

    Ensures year-level IDs are independent of month_level and not simply
    row position.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (no 'id' column).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'id' as the first column (values 1..n in random order).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    ids = rng.permutation(n) + 1  # 1-based, shuffled
    out = df.copy()
    out.insert(0, "id", ids)
    return out


def clean_activity(
    input_path: str,
    output_path: str,
    seed: int = RANDOM_ID_SEED,
) -> pd.DataFrame:
    """
    Load year-level activity CSV, add random IDs, rename columns to year indices.

    Parameters
    ----------
    input_path : str
        Path to mc.act.year.seq.now.csv.
    output_path : str
        Path to save cleaned pairfam_activity_by_year.csv.
    seed : int
        Random seed for id assignment.

    Returns
    -------
    pd.DataFrame
        Cleaned activity DataFrame.
    """
    df = pd.read_csv(input_path)
    df = _add_random_ids(df, seed=seed)
    df = _rename_time_columns_to_years(df, "activity")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def clean_family(
    input_path: str,
    output_path: str,
    seed: int = RANDOM_ID_SEED,
) -> pd.DataFrame:
    """
    Load year-level family CSV, add random IDs, rename columns to year indices.

    Parameters
    ----------
    input_path : str
        Path to mc.fam.year.seq.now.csv.
    output_path : str
        Path to save cleaned pairfam_family_by_year.csv.
    seed : int
        Random seed for id assignment (should match activity for same ordering).

    Returns
    -------
    pd.DataFrame
        Cleaned family DataFrame.
    """
    df = pd.read_csv(input_path)
    df = _add_random_ids(df, seed=seed)
    df = _rename_time_columns_to_years(df, "family")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def run_cleaning(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = RANDOM_ID_SEED,
) -> tuple:
    """
    Run full year-level cleaning: activity and family, same random ID order.

    Uses the same seed for both so that row i in activity and row i in family
    correspond to the same case (same random id).

    Parameters
    ----------
    input_dir : str, optional
        Directory containing mc.act.year.seq.now.csv and mc.fam.year.seq.now.csv.
        Defaults to data_sources/year_level.
    output_dir : str, optional
        Directory to write pairfam_activity_by_year.csv and pairfam_family_by_year.csv.
        Defaults to input_dir.
    seed : int, optional
        Random seed for id assignment.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (cleaned_activity_df, cleaned_family_df).
    """
    input_dir = input_dir or DEFAULT_INPUT_DIR
    output_dir = output_dir or input_dir

    activity_path = os.path.join(input_dir, ACTIVITY_INPUT)
    family_path = os.path.join(input_dir, FAMILY_INPUT)

    if not os.path.isfile(activity_path):
        raise FileNotFoundError(f"Activity file not found: {activity_path}")
    if not os.path.isfile(family_path):
        raise FileNotFoundError(f"Family file not found: {family_path}")

    activity_out = os.path.join(output_dir, ACTIVITY_OUTPUT)
    family_out = os.path.join(output_dir, FAMILY_OUTPUT)

    print("Cleaning year-level activity...")
    activity_df = clean_activity(activity_path, activity_out, seed=seed)
    print("Cleaning year-level family...")
    family_df = clean_family(family_path, family_out, seed=seed)

    print(f"Activity: {activity_df.shape} -> {activity_out}")
    print(f"Family:   {family_df.shape} -> {family_out}")
    return activity_df, family_df


if __name__ == "__main__":
    run_cleaning()
