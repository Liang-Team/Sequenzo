"""
# CombT (Combined Typology) Analysis Tutorial

This tutorial demonstrates how to use the CombT strategy for multi-domain sequence analysis.
CombT allows you to analyze relationships between multiple life domains (such as education,
employment, family formation) and identify meaningful combined typologies.

NOTE: Please run this as a Python file (.py file) rather than copying into a Jupyter Notebook (.ipynb),
as it includes interactive components that require user input during execution.
"""

from sequenzo import *

left_df = load_dataset('biofam_left_domain')
children_df = load_dataset('biofam_child_domain')
married_df = load_dataset('biofam_married_domain')

time_cols = [col for col in children_df.columns if col.startswith("age_")]

seq_left = SequenceData(left_df, time_type="age", time=time_cols, states=[0, 1],
                        labels=["At home", "Left home"], id_col="id")
seq_child = SequenceData(children_df, time_type="age", time=time_cols, states=[0, 1],
                         labels=["No child", "Child"], id_col="id")
seq_marr = SequenceData(married_df, time_type="age", time=time_cols, states=[0, 1],
                        labels=["Not married", "Married"], id_col="id")

domains = [seq_left, seq_child, seq_marr]
method_params = [
    {"method": "OM", "sm": "TRATE", "indel": "auto"},
    {"method": "OM", "sm": "CONSTANT", "indel": "auto"},
    {"method": "OM", "sm": "CONSTANT", "indel": 1},
]

# NOTE: The order of domains is critical - must match between domains list and domain_names
diss_matrices, membership_df = get_interactive_combined_typology(domains,
                                                                 method_params,
                                                                 domain_names=["Left", "Child", "Married"])

dat_matrix = compute_dat_distance_matrix(domains, method_params=method_params)

# Use CombT as the label
labels = membership_df["CombT"].values

# Merge sparse clusters - important to check the proportions before deciding min_size
merged_labels, merge_info = merge_sparse_combt_types(distance_matrix=dat_matrix,
                                                     labels=labels,
                                                     min_size=50,
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