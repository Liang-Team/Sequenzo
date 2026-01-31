"""
@Author  : Yuqi Liang 梁彧祺
@File    : clean_biofam_columns.py
@Time    : 31/01/2026 20:23
@Desc    :

Clean biofam time column names using clean_time_columns_auto.

- biofam.csv: columns like a15, a16, ..., a30 -> 15, 16, ..., 30
- biofam_child_domain.csv, biofam_left_domain.csv, biofam_married_domain.csv:
  columns like age_15, age_16, ..., age_30 -> 15, 16, ..., 30

Rows with missing (NA) IDs get randomly generated unique IDs that do not conflict
with existing ones. The total count of NA-ID rows is printed at the end (results: 244 rows).

Reads from data_sources/ and writes to cleaned_data/ (does not overwrite originals).
"""

import os
import random

import pandas as pd

from sequenzo.data_preprocessing import clean_time_columns_auto

# Paths: script in code/, data in data_sources/, output in cleaned_data/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIOFAM_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SOURCES = os.path.join(BIOFAM_ROOT, "data_sources")
OUTPUT_DIR = os.path.join(BIOFAM_ROOT, "cleaned_data")


def fill_na_ids(df: pd.DataFrame, id_col: str) -> tuple[pd.DataFrame, int]:
    """
    Replace missing (NA) IDs with randomly generated unique IDs that do not
    conflict with existing IDs. Returns (modified df, count of rows that had NA IDs).
    """
    na_mask = df[id_col].isna()
    na_count = int(na_mask.sum())
    if na_count == 0:
        return df, 0

    # Collect existing IDs as integers for comparison
    existing = set()
    for v in df[id_col].dropna():
        try:
            existing.add(int(float(v)))
        except (ValueError, TypeError):
            pass

    # Use a high range (900001–999999) to avoid collisions with typical idhous values
    candidate_pool = [i for i in range(900001, 1000000) if i not in existing]
    if len(candidate_pool) < na_count:
        extra = [i for i in range(1000000, 1000000 + na_count * 2) if i not in existing]
        candidate_pool.extend(extra)

    new_ids = random.sample(candidate_pool, na_count)
    df = df.copy()
    df.loc[na_mask, id_col] = new_ids

    return df, na_count


def main():
    # Create output directory if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_na_id_rows = 0

    # 1. biofam.csv: a15, a16, ..., a30 -> 15, 16, ..., 30
    #    ID column: idhous; fill NA IDs with random unique values
    biofam_path = os.path.join(DATA_SOURCES, "biofam.csv")
    df_biofam = pd.read_csv(biofam_path)
    df_biofam, na_count = fill_na_ids(df_biofam, id_col="idhous")
    total_na_id_rows += na_count
    if na_count > 0:
        print(f"biofam.csv: filled {na_count} rows with missing idhous")
    df_biofam = clean_time_columns_auto(df_biofam, prefix_patterns=["a"])
    output_biofam = os.path.join(OUTPUT_DIR, "biofam.csv")
    df_biofam.to_csv(output_biofam, index=False)
    print(f"Written: {output_biofam}")

    # 2. biofam_child_domain.csv, biofam_left_domain.csv, biofam_married_domain.csv:
    #    age_15, age_16, ..., age_30 -> 15, 16, ..., 30
    #    ID column: id; fill NA IDs with random unique values
    domain_files = [
        "biofam_child_domain.csv",
        "biofam_left_domain.csv",
        "biofam_married_domain.csv",
    ]
    for filename in domain_files:
        input_path = os.path.join(DATA_SOURCES, filename)
        df = pd.read_csv(input_path)
        df, na_count = fill_na_ids(df, id_col="id")
        total_na_id_rows += na_count
        if na_count > 0:
            print(f"{filename}: filled {na_count} rows with missing id")
        df = clean_time_columns_auto(df, prefix_patterns=["age_"])
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(output_path, index=False)
        print(f"Written: {output_path}")

    print(f"\nTotal rows with NA IDs that were filled: {total_na_id_rows}")
    print("Done. Original files in data_sources/ were not modified.")


if __name__ == "__main__":
    main()
