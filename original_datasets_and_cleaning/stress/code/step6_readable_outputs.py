"""
@Author  : Yuqi Liang 梁彧祺
@File    : step6_readable_outputs.py
@Time    : 27/02/2026 19:09
@Desc    : 
Step 6: Export readable-named datasets from stress_sequence_wide.csv.

Splits the wide format into two human-readable CSVs with full column names
(suitable for papers and sharing). Naming follows the DEPRESS dataset style:
- Perceived Stress Scale (PSS) -> perceived_stress_week_1, ...
- Stress state (L/M/H) -> stress_state_week_1, ...
- Survey date -> survey_date_week_1, ...

Outputs (in cleaned_data/):
  1. students_stress_states_by_week.csv   — one row per student; stress states only
  2. students_perceived_stress_and_dates_by_week.csv — one row per student; PSS scores + dates
"""

import pandas as pd
from pathlib import Path

from config import OUTPUT_DIR

# Input: the wide file produced by the main pipeline
INPUT_WIDE = OUTPUT_DIR / "stress_sequence_wide.csv"

# Output files with readable names
OUTPUT_STRESS_STATES = OUTPUT_DIR / "students_stress_states_by_week.csv"
OUTPUT_PSS_AND_DATES = OUTPUT_DIR / "students_perceived_stress_and_dates_by_week.csv"

# Column names: use 1, 2, ... 8 for week position (short, not verbose)
PERCEIVED_STRESS_COL = "pss_{}"   # pss_1 .. pss_8
SURVEY_DATE_COL = "date_{}"       # date_1 .. date_8


def load_wide():
    """Load the wide-format sequence CSV."""
    if not INPUT_WIDE.exists():
        raise FileNotFoundError(
            f"Run the main pipeline first to create {INPUT_WIDE}"
        )
    return pd.read_csv(INPUT_WIDE)


def build_stress_states_df(wide):
    """
    Build dataset 1: participant_id, cohort, 1, 2, 3, 4, 5, 6, 7, 8 (stress state per week).
    """
    df = pd.DataFrame()
    df["participant_id"] = wide["ID"]
    df["cohort"] = wide["cohort"]
    for w in range(1, 9):
        df[str(w)] = wide[f"s{w}"]
    return df


def build_pss_and_dates_df(wide):
    """
    Build dataset 2: participant_id, cohort, pss_1..8, date_1..8 (short column names).
    """
    df = pd.DataFrame()
    df["participant_id"] = wide["ID"]
    df["cohort"] = wide["cohort"]
    for w in range(1, 9):
        df[PERCEIVED_STRESS_COL.format(w)] = wide[f"pss{w}"]
        df[SURVEY_DATE_COL.format(w)] = wide[f"start{w}"]
    return df


def run_step6():
    """Read wide CSV and write the two readable-named CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide = load_wide()

    states_df = build_stress_states_df(wide)
    states_df.to_csv(OUTPUT_STRESS_STATES, index=False)

    pss_dates_df = build_pss_and_dates_df(wide)
    pss_dates_df.to_csv(OUTPUT_PSS_AND_DATES, index=False)

    print("Readable outputs written:")
    print(f"  1. {OUTPUT_STRESS_STATES.name}")
    print(f"  2. {OUTPUT_PSS_AND_DATES.name}")
    return states_df, pss_dates_df


if __name__ == "__main__":
    run_step6()
