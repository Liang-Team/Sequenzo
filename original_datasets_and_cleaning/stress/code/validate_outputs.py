"""
@Author  : Yuqi Liang 梁彧祺
@File    : validate_outputs.py
@Time    : 27/02/2026 20:31
@Desc    : 
Validate pipeline outputs against the acceptance checklist in requirements.md.
Raises AssertionError if any check fails.
"""

import pandas as pd
from pathlib import Path

from config import OUTPUT_WIDE, OUTPUT_LONG, OUTPUT_THRESHOLDS, OUTPUT_REPORT


def validate_outputs(report):
    """
    Check that all output files exist and meet the spec.
    report: dict with at least final_N.
    """
    final_N = report.get("final_N")
    if final_N is None:
        raise AssertionError("report must contain 'final_N'")

    # --- stress_sequence_wide.csv ---
    if not OUTPUT_WIDE.exists():
        raise AssertionError(f"Missing output: {OUTPUT_WIDE}")
    wide = pd.read_csv(OUTPUT_WIDE)
    assert len(wide) == final_N, (
        f"stress_sequence_wide.csv: expected {final_N} rows, got {len(wide)}"
    )
    required_wide_cols = ["ID", "cohort"] + [f"s{i}" for i in range(1, 9)]
    for c in required_wide_cols:
        assert c in wide.columns, f"stress_sequence_wide.csv: missing column '{c}'"
    state_cols = [f"s{i}" for i in range(1, 9)]
    missing = wide[state_cols].isna().any(axis=1).sum()
    assert missing == 0, (
        f"stress_sequence_wide.csv: s1..s8 must have no missing values (found {missing} rows with missing)"
    )

    # --- stress_sequence_long.csv ---
    if not OUTPUT_LONG.exists():
        raise AssertionError(f"Missing output: {OUTPUT_LONG}")
    long_df = pd.read_csv(OUTPUT_LONG)
    expected_long_rows = final_N * 8
    assert len(long_df) == expected_long_rows, (
        f"stress_sequence_long.csv: expected {expected_long_rows} rows, got {len(long_df)}"
    )
    id_counts = long_df.groupby("ID").size()
    assert (id_counts == 8).all(), (
        f"stress_sequence_long.csv: each ID must appear exactly 8 times (violations: {id_counts[id_counts != 8].to_dict()})"
    )
    for id_val, grp in long_df.groupby("ID"):
        t_vals = grp["t"].sort_values().tolist()
        assert t_vals == list(range(1, 9)), (
            f"stress_sequence_long.csv: ID {id_val} must have t=1..8, got {t_vals}"
        )

    # --- stress_thresholds.json ---
    assert OUTPUT_THRESHOLDS.exists(), f"Missing output: {OUTPUT_THRESHOLDS}"

    # --- cleaning_report.csv ---
    assert OUTPUT_REPORT.exists(), f"Missing output: {OUTPUT_REPORT}"
    report_df = pd.read_csv(OUTPUT_REPORT)
    required_metrics = [
        "total_rows_input",
        "rows_dropped_invalid_date",
        "rows_dropped_cohort",
        "rows_dropped_duplicates",
        "rows_dropped_invalid_pss",
        "final_N_individuals",
    ]
    metrics = set(report_df["metric"].tolist())
    for m in required_metrics:
        assert m in metrics, f"cleaning_report.csv: missing metric '{m}'"

    return True
