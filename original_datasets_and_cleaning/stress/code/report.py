"""
@Author  : Yuqi Liang 梁彧祺
@File    : report.py
@Time    : 26/02/2026 08:20
@Desc    : 
    Build and write the cleaning report (cleaning_report.csv).
    Summarizes all drop counts and final sample sizes.
"""

import pandas as pd

from config import OUTPUT_REPORT, COL_ID, COL_COHORT


def build_report_df(report, df_final=None):
    """
    Build a two-column table: metric, value.
    If df_final is provided, add cohort counts (Fall, Spring) from it.
    """
    rows = [
        ("total_rows_input", report.get("total_rows_input", "")),
        ("rows_dropped_invalid_date", report.get("rows_dropped_invalid_date", "")),
        ("rows_dropped_cohort", report.get("rows_dropped_cohort", "")),
        ("rows_dropped_duplicates", report.get("rows_dropped_duplicates", "")),
        ("rows_dropped_invalid_pss", report.get("rows_dropped_invalid_pss", "")),
        ("ids_before_8_filter", report.get("ids_before_8_filter", "")),
        ("ids_dropped_insufficient_weeks", report.get("ids_dropped_insufficient_weeks", "")),
        ("final_N_individuals", report.get("final_N", "")),
        ("final_rows_in_long_output", report.get("final_N", "") * 8 if report.get("final_N") is not None else ""),
        ("min_obs_per_id_after_filter", report.get("min_obs_per_id", "")),
        ("median_obs_per_id_after_filter", report.get("median_obs_per_id", "")),
        ("max_obs_per_id_after_filter", report.get("max_obs_per_id", "")),
    ]
    summary = pd.DataFrame(rows, columns=["metric", "value"])

    # Cohort counts in final sample
    if df_final is not None and len(df_final) > 0:
        cohort_counts = df_final.groupby(COL_COHORT)[COL_ID].nunique().reset_index()
        cohort_counts.columns = ["cohort", "count"]
        return summary, cohort_counts
    return summary, None


def write_report(report, df_final=None, path=OUTPUT_REPORT):
    """
    Write cleaning_report.csv with main metrics and optional cohort table.
    We write one CSV: first block is metric/value, then empty row, then cohort table if present.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    summary, cohort_counts = build_report_df(report, df_final)

    with open(path, "w") as f:
        summary.to_csv(f, index=False)
        if cohort_counts is not None:
            f.write("\n")
            cohort_counts.to_csv(f, index=False)


def run_report(report, df_final):
    """Write cleaning_report.csv."""
    write_report(report, df_final)
