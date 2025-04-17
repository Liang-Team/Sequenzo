"""
@Author  : Yuqi Liang 梁彧祺
@File    : CombT.py
@Time    : 17/04/2025 14:40
@Desc    : 
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
    {"method": "OM", "sm": "CONSTANT", "indel": 1},
    {"method": "OM", "sm": "CONSTANT", "indel": 1},
    {"method": "OM", "sm": "CONSTANT", "indel": 1},
]

membership_df = get_interactive_combined_typology(domains, method_params, domain_names=["Left", "Child", "Married"])