"""
@Author  : Yuqi Liang 梁彧祺
@File    : utils.py
@Time    : 02/05/2025 12:26
@Desc    : 
"""
import pandas as pd
from typing import List, Tuple


def extract_sequences(df: pd.DataFrame, time_cols: List[str]) -> List[List[str]]:
    """
    Efficiently extracts sequences from specified time columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        time_cols (List[str]): Columns representing the sequence over time.

    Returns:
        List[List[str]]: List of sequences (each sequence is a list of states).
    """
    return df[time_cols].values.tolist()


def get_state_space(sequences: List[List[str]]) -> List[str]:
    """
    Efficiently extracts unique states from a list of sequences.

    Parameters:
        sequences (List[List[str]]): Sequence data.

    Returns:
        List[str]: Sorted list of unique states.
    """
    seen = set()
    for seq in sequences:
        seen.update(seq)
    return sorted(seen)


def convert_to_prefix_tree_data(df: pd.DataFrame, time_cols: List[str]) -> Tuple[List[List[str]], List[str]]:
    """
    Wrapper to extract sequences and their state space from a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        time_cols (List[str]): Sequence columns (e.g., ['C1', ..., 'C10'])

    Returns:
        Tuple[List[List[str]], List[str]]: sequences, unique states
    """
    sequences = df[time_cols].values.tolist()
    states = get_state_space(sequences)
    return sequences, states