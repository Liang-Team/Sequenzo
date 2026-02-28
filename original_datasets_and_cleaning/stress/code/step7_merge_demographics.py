"""
@Author  : Yuqi Liang 梁彧祺
@File    : step7_merge_demographics.py
@Time    : 28/02/2026 10:17
@Desc    :
Step 7: Merge interpretation-layer variables into final datasets.

Loads demographic data from "Demographic and socioeconomic status.xlsx"
and daily activity diary from Daily_Activity_Diary.csv, then merges 4 variables
onto the stress sequence outputs:

  1. gender         — from Sex
  2. income_group   — recoded from Income: Lower (<$60k), Middle ($60k–$150k), Higher ($150k+), Unknown
  3. race           — self-reported race/ethnicity
  4. avg_study_time — from Daily_Activity_Diary, discretized by tertiles (33.3%, 66.7% quantiles) into
                      Lower (T1), Middle (T2), Higher (T3). NaN if no diary data for that ID.

Output column names are user-friendly. Raw income brackets are not kept; only income_group.

Note: school_year (GradYear) was intentionally excluded. Index plot by school_year and
regression both showed no meaningful effects; we had expected different years to show
different stress patterns but did not find this in the DEPRESS sample.
"""

import pandas as pd
from pathlib import Path

from config import DATA_SOURCE_DIR, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEMOGRAPHIC_XLSX = DATA_SOURCE_DIR / "Demographic and socioeconomic status.xlsx"
DAILY_DIARY_CSV = DATA_SOURCE_DIR / "Daily_Activity_Diary.csv"

# Daily diary: school-related columns (paper: 0 min=0, 15 min=1, 30 min=2, 45 min=3, >45 min=4)
DIARY_ID_COL = "ID"
DIARY_SCHOOL_COLS = [
    "Yesterday, how much time did you spend: - Speaking in class",
    "Yesterday, how much time did you spend: - Listening in class",
    "Yesterday, how much time did you spend: - Reading school-related material NOT on the Internet",
    "Yesterday, how much time did you spend: - Reading school-related material on the Internet",
]

INPUT_STRESS_STATES = OUTPUT_DIR / "students_stress_states_by_week.csv"
INPUT_PSS_AND_DATES = OUTPUT_DIR / "students_perceived_stress_and_dates_by_week.csv"
INPUT_WIDE = OUTPUT_DIR / "stress_sequence_wide.csv"

OUTPUT_STRESS_STATES = OUTPUT_DIR / "students_stress_states_by_week.csv"
OUTPUT_PSS_AND_DATES = OUTPUT_DIR / "students_perceived_stress_and_dates_by_week.csv"
OUTPUT_WIDE_ENRICHED = OUTPUT_DIR / "stress_sequence_wide_with_demographics.csv"

# ---------------------------------------------------------------------------
# Column mapping: Excel source -> output (readable)
# ---------------------------------------------------------------------------
EXCEL_SHEET = "Demograph&SES"
EXCEL_ID_COL = "ID"
EXCEL_SEX_COL = "Sex"
EXCEL_INCOME_COL = "Income"
EXCEL_RACE_COL = "Race"


def _load_avg_study_time_from_diary():
    """
    Load Daily_Activity_Diary.csv and compute per-participant average school-related
    activity level (0–4 scale). Returns a Series index by participant_id.
    """
    if not DAILY_DIARY_CSV.exists():
        return None
    df = pd.read_csv(DAILY_DIARY_CSV)
    df[DIARY_ID_COL] = df[DIARY_ID_COL].astype(str).str.strip()
    cols = [c for c in DIARY_SCHOOL_COLS if c in df.columns]
    if not cols:
        return None
    # numeric; invalid → NaN
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # per row (per day): mean of the 4 school-related items
    df["_school"] = df[cols].mean(axis=1)
    # per ID: mean over all diary days
    out = df.groupby(DIARY_ID_COL)["_school"].mean()
    return out


def _recode_income_to_group(val):
    """Recode raw income brackets into Lower / Middle / Higher (clear category labels)."""
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    s = str(val).strip()
    if s in ("I really don't know", "Prefer not to answer"):
        return "Unknown"
    if s in (
        "Les than $10,000",  # note: typo in source data
        "$10,000 to $19,999", "$20,000 to $29,999",
        "$40,000 to $49,999", "$50,000 to $59,999",
    ):
        return "Lower (<$60k)"
    if s in (
        "$60,000 to $69,999", "$70,000 to $79,999", "$80,000 to $89,999",
        "$90,000 to $99,999", "$100,000 to $149,999",
    ):
        return "Middle ($60k–$150k)"
    if s == "$150,000 or more":
        return "Higher ($150k+)"
    return "Unknown"


def load_demographic_data():
    """Load and standardize demographic variables from Excel."""
    if not DEMOGRAPHIC_XLSX.exists():
        raise FileNotFoundError(
            f"Demographic file not found: {DEMOGRAPHIC_XLSX}"
        )

    df = pd.read_excel(DEMOGRAPHIC_XLSX, sheet_name=EXCEL_SHEET)

    # Build merge-ready dataframe with readable column names
    demo = pd.DataFrame()
    demo["participant_id"] = df[EXCEL_ID_COL].astype(str).str.strip()

    # gender: normalize to lowercase for consistency
    demo["gender"] = df[EXCEL_SEX_COL].astype(str).str.strip().str.lower()

    # income_group: recode raw income brackets into low / medium / high
    demo["income_group"] = df[EXCEL_INCOME_COL].apply(_recode_income_to_group)

    # race
    demo["race"] = df[EXCEL_RACE_COL].astype(str).str.strip()

    # avg_study_time: from Daily_Activity_Diary, then discretize by tertiles (quantiles 1/3, 2/3) → low / medium / high
    diary_avg = _load_avg_study_time_from_diary()
    if diary_avg is not None:
        demo["avg_study_time"] = demo["participant_id"].map(diary_avg)
        valid = demo["avg_study_time"].notna()
        if valid.sum() >= 3:  # need at least 3 values for qcut
            qcat = pd.qcut(
                demo.loc[valid, "avg_study_time"], q=3,
                labels=["Lower (T1)", "Middle (T2)", "Higher (T3)"], duplicates="drop"
            ).astype(str)
            demo["avg_study_time"] = demo["avg_study_time"].astype(object)
            demo.loc[valid, "avg_study_time"] = qcat.values
    else:
        demo["avg_study_time"] = pd.NA

    return demo


DEMO_COLS = ["gender", "income_group", "race", "avg_study_time"]


def merge_demographics_into_df(target_df, demo_df, target_id_col):
    """
    Left-merge demographic variables onto target dataframe.
    target_id_col: name of ID column in target ("participant_id" or "ID")
    Drops any existing demo columns first so re-running is idempotent.
    """
    to_drop = [c for c in target_df.columns if c in DEMO_COLS]
    to_drop += [c for c in target_df.columns if c == "income"]  # legacy
    to_drop += [c for c in target_df.columns if c == "school_year"]  # dropped: no effects in index plot or regression
    to_drop += [c for c in target_df.columns if ("_x" in c or "_y" in c) and c.rsplit("_", 1)[0] in DEMO_COLS]
    out = target_df.drop(columns=to_drop, errors="ignore")
    demo_sub = demo_df.rename(columns={"participant_id": target_id_col})
    return out.merge(demo_sub, on=target_id_col, how="left")


def run_step7():
    """
    Merge demographics into all final datasets.
    Overwrites students_stress_states_by_week.csv and students_perceived_stress_and_dates_by_week.csv
    with enriched versions. Also writes stress_sequence_wide_with_demographics.csv.
    """
    demo = load_demographic_data()

    # 1. students_stress_states_by_week
    if INPUT_STRESS_STATES.exists():
        states = pd.read_csv(INPUT_STRESS_STATES)
        states_enriched = merge_demographics_into_df(states, demo, "participant_id")
        states_enriched.to_csv(OUTPUT_STRESS_STATES, index=False)

    # 2. students_perceived_stress_and_dates_by_week
    if INPUT_PSS_AND_DATES.exists():
        pss_dates = pd.read_csv(INPUT_PSS_AND_DATES)
        pss_dates_enriched = merge_demographics_into_df(pss_dates, demo, "participant_id")
        pss_dates_enriched.to_csv(OUTPUT_PSS_AND_DATES, index=False)

    # 3. stress_sequence_wide (by ID)
    if INPUT_WIDE.exists():
        wide = pd.read_csv(INPUT_WIDE)
        wide_enriched = merge_demographics_into_df(wide, demo, "ID")
        wide_enriched.to_csv(OUTPUT_WIDE_ENRICHED, index=False)

    print("Demographics merged into:")
    print(f"  {OUTPUT_STRESS_STATES.name}")
    print(f"  {OUTPUT_PSS_AND_DATES.name}")
    print(f"  {OUTPUT_WIDE_ENRICHED.name}")
    print("  Variables: gender, income_group, race, avg_study_time")


if __name__ == "__main__":
    run_step7()
