"""
@Author  : Yuqi Liang 梁彧祺
@File    : run_pipeline.py
@Time    : 26/02/2026 08:30
@Desc    : 
Main entry point: run the full stress sequence cleaning pipeline.

Usage:
    python run_pipeline.py

Runs steps in order:
  1. Load CSV, validate columns, filter Fall/Spring cohort, parse dates
  2. Deduplicate by (ID, StartDate)
  3. PSS validity (numeric, range 0-4), keep IDs with >= 8 obs, truncate to first 8
  4. Discretize PSS to L/M/H, write stress_thresholds.json
  5. Export stress_sequence_long.csv and stress_sequence_wide.csv
  6. Write cleaning_report.csv
  7. Export readable-named datasets (students_stress_states_by_week, students_perceived_stress_and_dates_by_week)
  8. Validate outputs

Outputs are written to stress/cleaned_data/.
"""

import sys
from pathlib import Path

# Ensure we can import from the same directory (config, step1_*, etc.)
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import OUTPUT_WIDE, OUTPUT_LONG, OUTPUT_THRESHOLDS, OUTPUT_REPORT
from step1_load_and_cohort import run_step1
from step2_dates_and_dedup import run_step2
from step3_pss_and_eligibility import run_step3
from step4_discretize import run_step4
from step5_export import run_step5
from report import run_report
from step6_readable_outputs import run_step6
from validate_outputs import validate_outputs


def main():
    report = {}

    # Step 1: load, cohort filter, parse dates
    df = run_step1(report)

    # Step 2: deduplicate by (ID, StartDate)
    df = run_step2(df, report)

    # Step 3: PSS validity, eligibility (>= 8 weeks), truncate to first 8
    df = run_step3(df, report)

    # Step 4: discretize PSS to L/M/H, write thresholds JSON
    df = run_step4(df)

    # Step 5: export long and wide CSVs
    run_step5(df)

    # Step 6: write cleaning report
    run_report(report, df)

    # Step 7: export readable-named datasets (full column names for papers/sharing)
    run_step6()

    # Step 8: validate outputs against acceptance checklist
    validate_outputs(report)

    print("Pipeline finished successfully.")
    print(f"  Output directory: {OUTPUT_WIDE.parent}")
    print(f"  {OUTPUT_LONG.name}")
    print(f"  {OUTPUT_WIDE.name}")
    print(f"  {OUTPUT_THRESHOLDS.name}")
    print(f"  {OUTPUT_REPORT.name}")
    print("  students_stress_states_by_week.csv")
    print("  students_perceived_stress_and_dates_by_week.csv")
    print(f"  Final N = {report.get('final_N', 'N/A')}")
    print("  Validation: all checks passed.")


if __name__ == "__main__":
    main()
