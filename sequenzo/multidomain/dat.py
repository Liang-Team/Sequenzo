"""
@Author  : Yuqi Liang 梁彧祺
@File    : dat.py
@Time    : 15/04/2025 17:28
@Desc    : DAT (Distance Additive Trick) strategy with customizable dissimilarity parameters per domain.
"""
import pandas as pd
import numpy as np
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures import get_distance_matrix


def compute_dat_distance_matrix(
    sequence_objects: list,
    method_params: list[dict] = None
) -> np.ndarray:
    """
    Compute Distance Additive Trick (DAT) distance matrix.
    This sums distance matrices from multiple SequenceData domains, each with its own method config.

    Parameters:
    - sequence_objects: List of SequenceData instances
    - method_params: List of dicts for each domain, e.g.:
        [{"method": "OM", "sm": "TRATE", "indel": "auto"}, ...]
        Each dict will be passed directly into get_distance_matrix as kwargs.

    Returns:
    - A numpy array representing the combined distance matrix
    """
    distance_matrices = []

    if method_params is None or len(method_params) != len(sequence_objects):
        raise ValueError(
            f"[DAT] Please provide a list of dissimilarity measures parameters for each domain.\n"
            f"Expected {len(sequence_objects)} dicts in method_params, but got: {method_params}.\n"
            f"For instance, if you have two domains, try something like this:\n\n"
            f"    method_params = [\n"
            f"        {{'method': 'OM', 'sm': 'CONSTANT', 'indel': 1}},\n"
            f"        {{'method': 'HAM'}}  # if using Hamming, no 'sm' or 'indel' needed\n"
            f"    ]\n\n"
            f"Each dict will be passed directly into get_distance_matrix(seqdata=..., **params)."
        )

    for seq, params in zip(sequence_objects, method_params):
        dist = get_distance_matrix(seqdata=seq, **params)
        distance_matrices.append(dist)

    dat_matrix = sum(distance_matrices)
    return dat_matrix


if __name__ == '__main__':

    from sequenzo import *

    left_df = load_dataset('biofam_left_domain')
    children_df = load_dataset('biofam_child_domain')
    married_df = load_dataset('biofam_married_domain')

    time_cols = [col for col in children_df.columns if col.startswith("age_")]

    seq_left = SequenceData(data=left_df, time_type="age", time=time_cols, states=[0, 1],
                            labels=["At home", "Left home"])
    seq_child = SequenceData(data=children_df, time_type="age", time=time_cols, states=[0, 1],
                             labels=["No child", "Child"])
    seq_marr = SequenceData(data=married_df, time_type="age", time=time_cols, states=[0, 1],
                            labels=["Not married", "Married"])

    domains_seq_list = [seq_left, seq_child, seq_marr]

    domain_params = [
        {"method": "OM", "sm": "TRATE", "indel": "auto"},
        {"method": "OM", "sm": "CONSTANT", "indel": "auto"},
        # {"method": "OM", "sm": "CONSTANT", "indel": 1},
        {"method": "DHD"}
    ]

    dat_matrix = compute_dat_distance_matrix(domains_seq_list, method_params=domain_params)

    print(dat_matrix)