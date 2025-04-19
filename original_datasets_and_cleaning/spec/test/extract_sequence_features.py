"""
@Author  : Yuqi Liang 梁彧祺
@File    : extract_sequence_features.py
@Time    : 19/04/2025 10:51
@Desc    : 
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy


def extract_sequence_features(seqdata) -> pd.DataFrame:
    """
    Extract derived features from a SequenceData object for each individual sequence.

    Features include:
    - num_unique_states: Number of unique states
    - num_transitions: Number of state changes
    - dominant_ratio: Proportion of the most frequent state
    - entropy: Entropy of the state distribution
    - mean_duration: Average duration of consecutive identical states

    :param seqdata: A SequenceData object with `.values` and `.states` attributes.
    :return: A pandas DataFrame with shape (n_sequences, n_features)
    """
    sequences = seqdata.values
    features = []

    for seq in sequences:
        seq = seq[~np.isnan(seq)]  # Remove NaNs if any
        if len(seq) == 0:
            features.append([0, 0, 0, 0, 0])
            continue

        unique, counts = np.unique(seq, return_counts=True)
        num_unique_states = len(unique)
        dominant_ratio = np.max(counts) / len(seq)
        ent = entropy(counts, base=2)

        transitions = np.sum(seq[:-1] != seq[1:])
        num_transitions = int(transitions)

        durations = []
        current_duration = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_duration = 1
        durations.append(current_duration)
        mean_duration = np.mean(durations)

        features.append([
            num_unique_states,
            num_transitions,
            dominant_ratio,
            ent,
            mean_duration
        ])

    df = pd.DataFrame(features, columns=[
        "num_unique_states",
        "num_transitions",
        "dominant_ratio",
        "entropy",
        "mean_duration"
    ])
    import ace_tools as tools; tools.display_dataframe_to_user(name="Derived Sequence Features", dataframe=df)
    return df
