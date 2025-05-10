"""
@Author  : Yuqi Liang 梁彧祺
@File    : helpers.py
@Time    : 01/05/2025 09:27
@Desc    : 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from typing import Union, List


def assign_unique_ids(df: pd.DataFrame, id_col_name: str = "Entity ID") -> pd.DataFrame:
    """
    Assigns a unique integer ID to each row in the DataFrame and inserts it as the first column.

    :param df: Input DataFrame.
    :param id_col_name: Name of the new ID column (default = "Entity ID").
    :return: DataFrame with the ID column inserted at the first position.
    """
    if id_col_name in df.columns:
        raise ValueError(f"[!] Column '{id_col_name}' already exists in the DataFrame.")

    df = df.copy()
    df.insert(0, id_col_name, np.arange(len(df)))
    return df


def long_to_wide_format_data(df: pd.DataFrame,
                             id_col: str,
                             time_col: str,
                             value_col: Union[str, List[str]]) -> pd.DataFrame:
    """
    Convert a long-format DataFrame to wide format.

    This function pivots the long-format data so that each unique time point becomes
    a separate column, and each row corresponds to one unique sequence (identified by `id_col`).

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame in long format.

    id_col : str
        The name of the column representing unique entity IDs.

    time_col : str
        The name of the column containing time points (must be a string, not a list).

    value_col : Union[str, List[str]]
        The name(s) of the column(s) containing state values.
        Can be a single string or a list of strings.

    Returns:
    -------
    pd.DataFrame
        A wide-format DataFrame with one row per ID and one column per time point for each value column.
        The column names are taken from the unique values in `time_col` combined with value column names.

    Notes:
    -----
    - This function assumes `df` is already in long format.
    - `time_col` must be a column *name* (string), not a list.
    - The top-left "column name" in the output (from pivot) may carry over as a column index name;
      this is removed automatically for clean output.
    - If multiple value columns are provided, the result will have multi-level columns.
    """
    # Ensure value_col is a list for consistency
    if isinstance(value_col, str):
        value_col = [value_col]

    wide_list = []

    for col in value_col:
        pivoted = df.pivot(index=id_col, columns=time_col, values=col).add_prefix(f'{col}_').reset_index()
        wide_list.append(pivoted)

    # Merge all pivoted DataFrames on the ID column
    wide = wide_list[0]
    for w in wide_list[1:]:
        wide = pd.merge(wide, w, on=id_col, how='outer')

    wide.columns.name = None  # Remove residual column group name from pivot
    return wide


def wide_to_long_format_data(df: pd.DataFrame,
                             id_col: str,
                             time_cols: Union[List[str], List[List[str]]],
                             var_name="time",
                             value_name="state") -> pd.DataFrame:
    """
    Convert a wide-format DataFrame to long format.

    :param df: Wide-format DataFrame.
    :param id_col: Column with unique IDs.
    :param time_cols: List of time columns or a list of lists if multiple value columns.
    :param var_name: Name for the time variable in long format.
    :param value_name: Name for the state value.
    :return: Long-format DataFrame.
    """
    if isinstance(time_cols[0], str):
        return df.melt(id_vars=[id_col], value_vars=time_cols, var_name=var_name, value_name=value_name)
    else:
        long_dfs = []
        for cols in time_cols:
            long_df = df.melt(id_vars=[id_col], value_vars=cols, var_name=var_name, value_name=value_name)
            long_dfs.append(long_df)
        return pd.concat(long_dfs, ignore_index=True).reset_index(drop=True)


def summarize_missing_values(df: pd.DataFrame,
                             plot: bool = True,
                             top_n: int = 5,
                             columns: list = None,
                             mode: str = 'matrix',  # 'matrix' or 'bar'
                             figsize=(10, 5),
                             save_as: str = None,
                             show: bool = True) -> None:
    """
    Summarize missing values in a DataFrame, with optional visualization.

    :param df: Input DataFrame
    :param plot: Whether to visualize missing values
    :param top_n: Number of rows with most missing values to show
    :param columns: Columns to limit analysis to
    :param mode: 'matrix' or 'bar' for visualization mode
    :param figsize: Figure size for visualization
    :param save_as: Path to save figure
    :param show: Whether to display the figure
    """
    print("🔍 Missing Value Summary")
    print("-" * 40)

    if columns:
        df = df[columns]

    # 1. Summary per column
    missing_per_column = df.isnull().sum()
    percent_missing = (missing_per_column / len(df)) * 100
    summary_df = pd.DataFrame({
        'Missing Count': missing_per_column,
        'Missing (%)': percent_missing.round(2)
    }).sort_values('Missing Count', ascending=False)

    print("[Columns with Missing Values]")
    print(summary_df[summary_df['Missing Count'] > 0])

    # 2. Summary per row
    row_missing = df.isnull().sum(axis=1)
    if row_missing.max() > 0:
        print(f"\n[Top {top_n} Rows with Most Missing Values]")
        print(row_missing.sort_values(ascending=False).head(top_n).rename("Missing Count").to_frame())

    # 3. Visualization
    if plot and not summary_df.empty:
        plt.figure(figsize=figsize)
        if mode == 'matrix':
            fig = msno.matrix(df)
        elif mode == 'bar':
            fig = msno.bar(df)
        else:
            raise ValueError("mode must be either 'matrix' or 'bar'")

        if save_as:
            fig.figure.savefig(save_as, bbox_inches='tight', dpi=200)
        if not show:
            plt.close()


def replace_cluster_id_by_labels(df, mapping=None, new_cluster_column_name='Cluster', new_id_column_name='Entity ID'):
    """
    Once users have gotten the membership table,
    this function helps replace cluster IDs in a DataFrame with user-defined labels and updates column names.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'Entity ID' and 'Cluster' columns.
        mapping (dict, optional): A dictionary where keys are cluster IDs (e.g., 1, 2, 3, 4)
                                  and values are the corresponding labels. Default is an empty dictionary.
        new_cluster_column_name (str): The name of the new cluster column. Default is 'Cluster'.
        new_id_column_name (str): The name of the new entity ID column. Default is 'Entity ID'.

    Returns:
        pd.DataFrame: A new DataFrame with cluster IDs replaced by labels and updated column names.

    Example:
        original_df = pd.DataFrame({'Entity ID': [1, 2, 3], 'Cluster': [1, 2, 3]})
        mapping = {1: 'A', 2: 'B', 3: 'C'}
        new_df = replace_cluster_id_by_labels(original_df, mapping, 'New Cluster', 'New ID')
    """
    if mapping is None:
        mapping = {}

    # Check if the necessary columns exist in the DataFrame
    if 'Entity ID' not in df.columns or 'Cluster' not in df.columns:
        raise ValueError("The input DataFrame must contain 'Entity ID' and 'Cluster' columns.")

    # Check if all keys in the mapping are valid cluster IDs in the DataFrame
    unique_clusters = set(df['Cluster'].unique())
    for cluster_id in mapping.keys():
        if cluster_id not in unique_clusters:
            raise ValueError(f"Cluster ID {cluster_id} from the mapping does not exist in the DataFrame.")

    # Replace cluster IDs with the specified labels
    df['Cluster'] = df['Cluster'].map(mapping).fillna(df['Cluster'])

    # Rename the columns
    df.rename(columns={'Entity ID': new_id_column_name, 'Cluster': new_cluster_column_name}, inplace=True)

    return df


if __name__ == '__main__':
    # Example long-format data
    data = {
        'id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
        'time': ['T1', 'T2', 'T3', 'T1', 'T2', 'T1', 'T2', 'T3'],
        'value1': [10, 20, 30, 40, 50, 60, 70, 80],
        'value2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
    }

    df = pd.DataFrame(data)
    print(df)

    # Test with a single value column
    print("\nTest with a single value column:")
    first_df = long_to_wide_format_data(df, 'id', 'time', 'value1')
    print(first_df)

    # Test with multiple value columns
    print("\nTest with multiple value columns:")
    second_df = long_to_wide_format_data(df, 'id', 'time', ['value1', 'value2'])
    print(second_df)
    print('end')

    # ------------------------------------
    # data = {
    #     'id': ['A', 'B', 'C'],
    #     'T1_value1': [10, 40, 60],
    #     'T2_value1': [20, 50, 70],
    #     'T3_value1': [30, None, 80],
    #     'T1_value2': [1.1, 4.4, 6.6],
    #     'T2_value2': [2.2, 5.5, 7.7],
    #     'T3_value2': [3.3, None, 8.8]
    # }
    # df = pd.DataFrame(data)
    #
    # print(df)
    #
    # print("\nTest with single value column:")
    # print(wide_to_long_format_data(df, 'id', ['T1_value1', 'T2_value1', 'T3_value1']))
    #
    # print("\nTest with multiple value columns:")
    # print(wide_to_long_format_data(df, 'id',
    #                                [['T1_value1', 'T2_value1', 'T3_value1'], ['T1_value2', 'T2_value2', 'T3_value2']]))
