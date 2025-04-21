"""
@Author  : Yuqi Liang 梁彧祺
@File    : combt.py
@Time    : 21/04/2025 13:30
@Desc    : 
"""
"""
# CombT (Combined Typology) Analysis Tutorial

This tutorial demonstrates how to use the CombT strategy for multi-domain sequence analysis.
CombT allows you to analyze relationships between multiple life domains (such as education,
employment, family formation) and identify meaningful combined typologies.

NOTE: Please run this as a Python file (.py file) rather than copying into a Jupyter Notebook (.ipynb),
as it includes interactive components that require user input during execution.
"""

from sequenzo import *

from sequenzo import *
import pandas as pd
import numpy as np
import hdbscan

family_df = pd.read_csv('/Users/lei/Documents/japan_romance/multidomain_algorithm/family_15_35.csv')
happiness_df = pd.read_csv('/Users/lei/Documents/japan_romance/multidomain_algorithm/happiness_15_35.csv')

family_mapping = {
    1: "Single",
    2: "Romantic Partner",
    3: "Married"
}

# 替换 time columns（15~35）为 label
time_cols = [str(i) for i in range(15, 36)]
family_df[time_cols] = family_df[time_cols].replace(family_mapping)

happiness_mapping = {
    1: "Unhappy",
    2: "Somewhat unhappy",
    3: "Neutral",
    4: "Somewhat happy",
    5: "Happy"
}

happiness_df[time_cols] = happiness_df[time_cols].replace(happiness_mapping)

# --------------

time_cols = []

for i in list(range(15, 36)):
    time_cols.append(str(i))

family_colors = ["#D1C2D3", "#C4473D", "#574266"]

family_sequence = SequenceData(data=family_df,
                               time_type='age',
                               time=time_cols,
                               states=["Single", "Romantic Partner", "Married"],
                               custom_colors=family_colors)

happiness_colors = ["#4263A3", "#8AB7C5", "#D6DEE9", "#E0BA19", "#C69519"]

happiness_sequence = SequenceData(data=happiness_df,
                                  time_type='age',
                                  time=time_cols,
                                  states=["Unhappy", "Somewhat unhappy", "Neutral", "Somewhat happy", "Happy"],
                                  custom_colors=happiness_colors)

domains = [family_sequence, happiness_sequence]
method_params = [
    {"method": "OM", "sm": "CONSTANT", "indel": 1},
    {"method": "OM", "sm": "CONSTANT", "indel": 1},
]

# NOTE: The order of domains is critical - must match between domains list and domain_names
diss_matrices, membership_df = get_interactive_combined_typology(domains,
                                                                 method_params,
                                                                 domain_names=["Family", "Happiness"])

dat_matrix = compute_dat_distance_matrix(domains, method_params=method_params)

# Use CombT as the label
labels = membership_df["CombT"].values

# Merge sparse clusters - important to check the proportions before deciding min_size
merged_labels, merge_info = merge_sparse_combt_types(distance_matrix=dat_matrix,
                                                     labels=labels,
                                                     min_size=10,
                                                     asw_threshold=0.5,
                                                     verbose=True,
                                                     # Optional parameters below, the default is True
                                                     print_merge_details=True,
                                                     visualize_process=True,
                                                     visualization_path="merge_progress_combt.png"
                                                     )

# Update the membership dataframe
membership_df["CombT_Merged"] = merged_labels

# Save results
membership_df.reset_index().rename(columns={"index": "id"}).to_csv("combt_membership_table.csv", index=False)
print("\n[>] combt_membership_table.csv has been saved.")