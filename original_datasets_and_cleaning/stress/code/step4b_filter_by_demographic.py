"""
@Author  : Yuqi Liang 梁彧祺
@File    : step4b_filter_by_demographic.py
@Time    : 28/02/2026
@Desc    :
Step 4b: Keep only participants who have demographic data (final N = 76).

**Why this step is required**
  The stress sequence data (from PSS weekly surveys) has 89 participants who met
  eligibility (Fall/Spring cohort, ≥8 valid weeks). The demographic file
  ("Demographic and socioeconomic status.xlsx") contains 140 participants,
  but only 76 of them are in the 89-person stress sample. The remaining 13
  participants have no row in the demographic file (e.g. they did not complete
  the demographic survey, or their ID is not present in that export).

**What this step does**
  - Loads the demographic Excel and reads the list of participant IDs.
  - Filters the current dataframe to keep only rows whose ID appears in the
    demographic file.
  - Updates the report so that final_N = 76 and the number dropped is
    recorded (ids_dropped_no_demographic = 13).

**Result**
  All downstream outputs (stress_sequence_wide, stress_sequence_long,
  students_stress_states_by_week, etc.) will contain exactly 76 participants,
  all with non-missing demographic variables (gender, income_group, race).
"""

import pandas as pd
from pathlib import Path

from config import COL_ID, DATA_SOURCE_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEMOGRAPHIC_XLSX = DATA_SOURCE_DIR / "Demographic and socioeconomic status.xlsx"
EXCEL_SHEET = "Demograph&SES"
EXCEL_ID_COL = "ID"


def get_demographic_ids():
    """Return the set of participant IDs present in the demographic Excel."""
    if not DEMOGRAPHIC_XLSX.exists():
        raise FileNotFoundError(
            f"Demographic file not found: {DEMOGRAPHIC_XLSX}. "
            "Cannot filter by demographic eligibility."
        )
    df = pd.read_excel(DEMOGRAPHIC_XLSX, sheet_name=EXCEL_SHEET)
    return set(df[EXCEL_ID_COL].astype(str).str.strip())


def run_step4b(df, report):
    """
    Keep only participants whose ID appears in the demographic file.
    Updates report with N before/after and ids_dropped_no_demographic.
    Returns the filtered dataframe (final N = 76).
    """
    demo_ids = get_demographic_ids()
    ids_in_df = set(df[COL_ID].astype(str).str.strip())

    n_before = df[COL_ID].nunique()
    ids_to_keep = ids_in_df & demo_ids
    df_filtered = df[df[COL_ID].astype(str).str.strip().isin(ids_to_keep)].copy()
    n_after = df_filtered[COL_ID].nunique()
    n_dropped = n_before - n_after

    report["N_before_demographic_filter"] = n_before
    report["ids_dropped_no_demographic"] = n_dropped
    report["final_N"] = n_after

    if n_dropped > 0:
        dropped_ids = sorted(ids_in_df - demo_ids)
        print(f"[>] Step 4b: Keeping only participants with demographic data.")
        print(f"    N before = {n_before}, N after = {n_after}, dropped = {n_dropped}.")
        print(f"    Dropped IDs (no demographic record): {dropped_ids}.")

    return df_filtered
