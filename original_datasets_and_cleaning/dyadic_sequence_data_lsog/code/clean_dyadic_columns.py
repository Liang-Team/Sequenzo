"""
@Author  : Yuqi Liang 梁彧祺
@File    : clean_dyadic_columns.py
@Time    : 31/01/2026 10:01
@Desc    : 
Clean polyadic sequence data (LSOG): rename time columns from status15 -> 15, pstatus15 -> 15, etc.

Uses clean_time_columns_auto from sequenzo.data_preprocessing. Reads original CSVs from
data_sources/, applies it, and writes dyadic_children.csv and dyadic_parents.csv.
"""

import os

import pandas as pd

from sequenzo.data_preprocessing import clean_time_columns_auto

# Paths: script in code/, data in data_sources/, output in repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_SOURCES = os.path.join(REPO_ROOT, "data_sources")
OUTPUT_DIR = REPO_ROOT

INPUT_CHILDREN = os.path.join(DATA_SOURCES, "original_dyadic_children.csv")
INPUT_PARENTS = os.path.join(DATA_SOURCES, "original_dyadic_parents.csv")
OUTPUT_CHILDREN = os.path.join(OUTPUT_DIR, "dyadic_children.csv")
OUTPUT_PARENTS = os.path.join(OUTPUT_DIR, "dyadic_parents.csv")


def main():
    # Children: status15, status16, ... -> 15, 16, ...
    df_children = pd.read_csv(INPUT_CHILDREN)
    df_children = clean_time_columns_auto(df_children, prefix_patterns=["status"])
    df_children.to_csv(OUTPUT_CHILDREN, index=False)
    print(f"Written: {OUTPUT_CHILDREN}")

    # Parents: pstatus15, pstatus16, ... -> 15, 16, ...
    df_parents = pd.read_csv(INPUT_PARENTS)
    df_parents = clean_time_columns_auto(df_parents, prefix_patterns=["pstatus"])
    df_parents.to_csv(OUTPUT_PARENTS, index=False)
    print(f"Written: {OUTPUT_PARENTS}")


if __name__ == "__main__":
    main()
