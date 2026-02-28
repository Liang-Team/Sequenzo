"""
@Author  : Yuqi Liang 梁彧祺
@File    : config.py
@Time    : 26/02/2026 08:15
@Desc    : 
Configuration for the stress sequence cleaning pipeline.
All paths and constants in one place for easy editing.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
STRESS_DIR = CODE_DIR.parent
DATA_DIR = STRESS_DIR
DATA_SOURCE_DIR = STRESS_DIR / "data_source"

INPUT_CSV = DATA_SOURCE_DIR / "PANAS_PSS.csv"
OUTPUT_DIR = DATA_DIR / "cleaned_data"  # all pipeline outputs go here

OUTPUT_WIDE = OUTPUT_DIR / "stress_sequence_wide.csv"
OUTPUT_LONG = OUTPUT_DIR / "stress_sequence_long.csv"
OUTPUT_THRESHOLDS = OUTPUT_DIR / "stress_thresholds.json"
OUTPUT_REPORT = OUTPUT_DIR / "cleaning_report.csv"

# ---------------------------------------------------------------------------
# Column names (must match PANAS_PSS.csv)
# ---------------------------------------------------------------------------
COL_START_DATE = "StartDate"
COL_ID = "ID"
COL_PSS = "PSS"

# Optional columns we may keep in outputs
COL_COHORT = "cohort"
COL_T = "t"
COL_STRESS_STATE = "stress_state"

# ---------------------------------------------------------------------------
# Cohort rules: keep only IDs whose prefix is in this set
# ---------------------------------------------------------------------------
COHORT_PREFIXES = ("Fall", "Spring")

# ---------------------------------------------------------------------------
# PSS validity
# ---------------------------------------------------------------------------
PSS_MIN = 0.0
PSS_MAX = 4.0
# Option A (strict): drop rows outside [PSS_MIN, PSS_MAX]

# ---------------------------------------------------------------------------
# Eligibility: minimum number of valid weekly observations per person
# ---------------------------------------------------------------------------
MIN_WEEKS = 8

# ---------------------------------------------------------------------------
# Discretization: convert continuous PSS to states L / M / H
# boundaries: [low_max_exclusive, medium_max_exclusive] -> L, M, H
# L: PSS < 1.5,  M: 1.5 <= PSS < 2.5,  H: PSS >= 2.5
# ---------------------------------------------------------------------------
DISCRETIZATION = {
    "method_name": "fixed_thresholds_v1",
    "boundaries": [1.5, 2.5],
    "labels": ["L", "M", "H"],
}
