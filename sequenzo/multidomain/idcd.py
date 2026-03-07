"""
@Author  : Yuqi Liang 梁彧祺
@File    : idcd.py
@Time    : 15/04/2025 16:38
@Desc    :
    IDCD strategy for multidomain sequence analysis in Python, with custom time, states, and labels.
"""
from typing import List, Dict
import pandas as pd
from sequenzo.define_sequence_data import SequenceData


def _generate_combined_sequence_from_csv(csv_paths: List[str],
                                         time_cols: List[str],
                                         id_col: str = "id") -> pd.DataFrame:
    """
    Load multiple CSVs, extract time sequences, and combine into a multidomain sequence.
    Only observed combinations will be used.

    Parameters:
        csv_paths: List of file paths, each containing one domain's sequence data
        time_cols: Time columns to extract and align
        id_col: ID column to align on

    Returns:
        combined_df: DataFrame with combined state sequences

    Raises:
        ValueError: If any CSV is missing required columns
    """
    import os
    domain_dfs = []

    for idx, path in enumerate(csv_paths):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV at '{path}': {str(e)}")

        # Check if ID column exists
        if id_col not in df.columns:
            raise ValueError(
                f"Missing ID column '{id_col}' in file: {path}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Check if all time columns exist
        missing_cols = [col for col in time_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing time columns {missing_cols} in file: {path}\n"
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()
        df.sort_values(by=id_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        domain_dfs.append(df)

    # Combine states row-wise
    combined_matrix = []
    for i in range(domain_dfs[0].shape[0]):
        row = []
        for t in time_cols:
            combo = '+'.join(str(df.at[i, t]) for df in domain_dfs)
            row.append(combo)
        combined_matrix.append(row)

    combined_df = pd.DataFrame(combined_matrix, columns=time_cols)
    combined_df.insert(0, id_col, domain_dfs[0][id_col].values)

    return combined_df


def _generate_combined_sequence_from_dfs(domain_dfs: List[pd.DataFrame],
                                         time_cols: List[str],
                                         id_col: str = "id") -> pd.DataFrame:
    """
    Combine multiple domain DataFrames into a multidomain sequence (same logic as CSV version).
    Each DataFrame must have id_col and all time_cols; rows are aligned by id_col (assumed sorted).
    """
    for idx, df in enumerate(domain_dfs):
        if id_col not in df.columns:
            raise ValueError(
                f"Missing ID column '{id_col}' in domain DataFrame at index {idx}. "
                f"Available columns: {list(df.columns)}"
            )
        missing_cols = [col for col in time_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing time columns {missing_cols} in domain DataFrame at index {idx}. "
                f"Available columns: {list(df.columns)}"
            )
    dfs = []
    for df in domain_dfs:
        d = df.copy()
        d.sort_values(by=id_col, inplace=True)
        d.reset_index(drop=True, inplace=True)
        dfs.append(d)
    combined_matrix = []
    for i in range(dfs[0].shape[0]):
        row = []
        for t in time_cols:
            combo = '+'.join(str(df.at[i, t]) for df in dfs)
            row.append(combo)
        combined_matrix.append(row)
    combined_df = pd.DataFrame(combined_matrix, columns=time_cols)
    combined_df.insert(0, id_col, dfs[0][id_col].values)
    return combined_df


def _build_idcd_sequence_data(combined_df: pd.DataFrame,
                              time_cols: List[str],
                              id_col: str,
                              domain_state_labels: List[Dict] = None) -> SequenceData:
    """Build SequenceData from a combined multidomain DataFrame (used by both CSV and DFS entry points)."""
    flat_vals = combined_df[time_cols].values.ravel()
    observed_states = pd.Series(flat_vals).value_counts()
    proportions = observed_states / len(flat_vals) * 100
    if domain_state_labels:
        pretty_labels = []
        for state in observed_states.index:
            parts = state.split("+")
            label_parts = []
            for i, token in enumerate(parts):
                try:
                    key = int(token) if token.isdigit() else token
                    label = domain_state_labels[i].get(key, str(token))
                except Exception:
                    label = str(token)
                label_parts.append(label)
            pretty_labels.append(' + '.join(label_parts))
    else:
        pretty_labels = observed_states.index.tolist()
    freq_table = pd.DataFrame({
        "State": observed_states.index,
        "Label": pretty_labels,
        "Frequency": observed_states.values,
        "Proportion (%)": proportions.round(2)
    })
    print("\n[IDCD] Observed Combined States Frequency Table:")
    print(freq_table.to_string(index=False))
    return SequenceData(
        data=combined_df,
        time=time_cols,
        states=observed_states.index.tolist(),
        labels=pretty_labels,
        id_col=id_col
    )


def create_idcd_sequence_from_csvs(
    csv_paths: List[str],
    time_cols: List[str],
    id_col: str = "id",
    domain_state_labels: List[Dict] = None
) -> SequenceData:
    """
    Create IDCD-style SequenceData from multiple CSVs.
    Combines real observed joint states and builds sequence data.

    Parameters:
    - csv_paths: List of paths to domain CSVs
    - time_cols: List of time column names to use
    - id_col: ID column name
    - domain_state_labels: List of dictionaries mapping raw state values to labels for each domain

    Returns:
    - SequenceData object with expanded alphabet of observed joint states
    """
    combined_df = _generate_combined_sequence_from_csv(csv_paths, time_cols, id_col=id_col)
    return _build_idcd_sequence_data(combined_df, time_cols, id_col, domain_state_labels)


def create_idcd_sequence_from_dfs(
    domain_dfs: List[pd.DataFrame],
    time_cols: List[str],
    id_col: str = "id",
    domain_state_labels: List[Dict] = None
) -> SequenceData:
    """
    Create IDCD-style SequenceData from multiple domain DataFrames (e.g. from load_dataset).
    Combines real observed joint states and builds sequence data.

    Parameters:
    - domain_dfs: List of DataFrames, each containing one domain's sequence data
    - time_cols: List of time column names to use
    - id_col: ID column name
    - domain_state_labels: List of dictionaries mapping raw state values to labels for each domain

    Returns:
    - SequenceData object with expanded alphabet of observed joint states
    """
    combined_df = _generate_combined_sequence_from_dfs(domain_dfs, time_cols, id_col=id_col)
    return _build_idcd_sequence_data(combined_df, time_cols, id_col, domain_state_labels)




