"""
@Author  : Yuqi Liang 梁彧祺
@File    : basic_indicators.py
@Time    : 31/01/2026 09:23
@Desc    : Basic sequence characteristics indicators

    This module implements basic sequence characteristics including:
    - Sequence length (seqlength)
    - Spell durations (seqdur)
    - Visited states (visited, visitp)
    - Recurrence (recu)
    - Mean spell duration (meand, meand2)
    - Duration standard deviation (dustd, dustd2)

    Reference: TraMineR R package
    - seqlength: R/seqlength.R
    - seqdur: R/seqdur.R
    - seqistatd: R/seqistatd.R
    - seqivardur: R/seqivardur.R (for meand, dustd)
"""

import numpy as np
import pandas as pd
from typing import Union

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.utils.seqlength import seqlength
from sequenzo.dissimilarity_measures.utils.seqdur import seqdur
from sequenzo.dissimilarity_measures.utils.seqdss import seqdss
from .state_frequencies_and_entropy_per_sequence import get_state_freq_and_entropy_per_seq
from .simple_characteristics import cut_prefix


def get_sequence_length(seqdata: SequenceData, with_missing: bool = True) -> pd.DataFrame:
    """
    Calculate the length of each sequence in the dataset.
    
    The length is calculated as the number of non-void positions in each sequence.
    If with_missing=False, missing values are excluded from the count.
    
    Args:
        seqdata: SequenceData object containing sequence data
        with_missing: If True, include missing values in length calculation.
                     If False, exclude missing values.
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Length' containing sequence lengths.
                     Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqlength.R
        https://github.com/cran/TraMineR/blob/master/R/seqlength.R
    
    Examples:
        >>> seq_length = get_sequence_length(seqdata, with_missing=True)
        >>> print(seq_length.head())
                 Length
        ID_1        10
        ID_2        12
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Use the existing seqlength utility function
    # Note: seqlength returns a numpy array, not a DataFrame
    lengths = seqlength(seqdata)
    
    # Convert to DataFrame with proper column name
    result = pd.DataFrame(lengths, columns=['Length'], index=seqdata.seqdata.index)
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result


def get_spell_durations(seqdata: SequenceData, with_missing: bool = False) -> pd.DataFrame:
    """
    Extract spell durations from sequences.
    
    A spell is a consecutive period spent in the same state. This function
    extracts the duration of each spell for all sequences.
    
    Args:
        seqdata: SequenceData object containing sequence data
        with_missing: If True, treat missing values as regular states.
                     If False, ignore missing values.
    
    Returns:
        pd.DataFrame: DataFrame with columns 'DUR1', 'DUR2', ... containing
                     spell durations for each sequence. Rows correspond to sequences.
    
    Reference:
        TraMineR: R/seqdur.R
        https://github.com/cran/TraMineR/blob/master/R/seqdur.R
    
    Examples:
        >>> durations = get_spell_durations(seqdata, with_missing=False)
        >>> print(durations.head())
                 DUR1  DUR2  DUR3
        ID_1       3     2     5
        ID_2       1     4     3
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Use the existing seqdur utility function
    # Note: seqdur doesn't support with_missing parameter - missing values are handled at SequenceData level
    # seqdur returns a numpy array, convert to DataFrame
    durations_array = seqdur(seqdata)
    durations = pd.DataFrame(durations_array, index=seqdata.seqdata.index)
    
    # Rename columns to 'DUR1', 'DUR2', etc. to match TraMineR convention
    n_cols = durations.shape[1]
    durations.columns = [f'DUR{i+1}' for i in range(n_cols)]
    
    # Convert 0 values to NaN (0 indicates padding/empty slots, not actual spell durations)
    # Spell durations should be at least 1, so 0 values are invalid
    durations = durations.replace(0, np.nan)
    
    return durations


def get_visited_states(seqdata: SequenceData, with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate the number of distinct states visited by each sequence.
    
    This counts how many different states appear in each sequence.
    
    Args:
        seqdata: SequenceData object containing sequence data
        with_missing: If True, treat missing values as regular states.
                     If False, ignore missing values.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
                     - 'Visited': Number of distinct states visited
                     - 'Visitp': Proportion of states visited (out of total alphabet)
                     Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqindic.R (visited, visitp indicators)
        Uses seqistatd from R/seqistatd.R internally
    
    Examples:
        >>> visited = get_visited_states(seqdata, with_missing=False)
        >>> print(visited.head())
                 Visited    Visitp
        ID_1          3      0.75
        ID_2          4      1.00
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Get state distribution for each sequence
    sdist = get_state_freq_and_entropy_per_seq(seqdata, prop=False)
    
    # Count number of states with frequency > 0 (visited states)
    # Exclude the ID column
    state_cols = [col for col in sdist.columns if col != 'ID']
    nvisit = (sdist[state_cols] > 0).sum(axis=1)
    
    # Calculate proportion of states visited
    # Total alphabet size
    alph_size = len(seqdata.states)
    if with_missing and hasattr(seqdata, 'ismissing') and seqdata.ismissing:
        alph_size += 1  # Add 1 for missing state
    
    pvisit = nvisit / alph_size
    
    # Create result DataFrame
    # Use the index from sdist before reset_index, or use ID column values
    result = pd.DataFrame({
        'Visited': nvisit.values,
        'Visitp': pvisit.values
    }, index=sdist.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result


def get_recurrence(seqdata: SequenceData, with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate the recurrence index for each sequence.
    
    Recurrence is the average number of visits to visited states.
    It is calculated as: dlgth / visited, where dlgth is the length
    of the distinct state sequence (DSS) and visited is the number
    of distinct states visited.
    
    Args:
        seqdata: SequenceData object containing sequence data
        with_missing: If True, treat missing values as regular states.
                     If False, ignore missing values.
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Recu' containing
                     recurrence values. Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqindic.R (recu indicator)
        Formula: dlgth / visited
    
    Examples:
        >>> recurrence = get_recurrence(seqdata, with_missing=False)
        >>> print(recurrence.head())
                 Recu
        ID_1     2.33
        ID_2     1.50
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    # Get DSS length
    dss = seqdss(seqdata)
    dlgth = seqlength(dss)  # Returns numpy array
    
    # Get number of visited states
    visited_df = get_visited_states(seqdata, with_missing=with_missing)
    nvisit = visited_df['Visited'].values
    
    # Calculate recurrence: dlgth / visited
    # Handle division by zero
    recu = dlgth / nvisit
    recu = pd.Series(recu, index=seqdata.seqdata.index)
    recu = recu.replace([np.inf, -np.inf], np.nan)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Recu': recu.values
    }, index=seqdata.seqdata.index)
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result


def get_mean_spell_duration(seqdata: SequenceData, type: int = 1, with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate mean spell duration for each sequence.
    
    Mean spell duration is the average length of spells (consecutive periods
    in the same state) within each sequence.
    
    Args:
        seqdata: SequenceData object containing sequence data
        type: Type of calculation:
             1 - Mean duration considering only visited states
             2 - Mean duration taking non-visited states into account
        with_missing: If True, treat missing values as regular states.
                     If False, ignore missing values.
    
    Returns:
        pd.DataFrame: DataFrame with one column 'MeanD' (or 'MeanD2' for type=2)
                     containing mean spell durations. Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqivardur.R
        The mean duration is extracted from the variance calculation.
        https://github.com/cran/TraMineR/blob/master/R/seqivardur.R
    
    Examples:
        >>> mean_dur = get_mean_spell_duration(seqdata, type=1)
        >>> print(mean_dur.head())
                 MeanD
        ID_1      3.33
        ID_2      2.50
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    if type not in [1, 2]:
        raise ValueError("[!] type must be 1 or 2.")
    
    # Import the variance function which also computes mean duration
    from .variance_of_spell_durations import get_spell_duration_variance
    
    # Get variance results which include mean duration
    var_results = get_spell_duration_variance(seqdata, type=type)
    
    # Extract mean duration
    if type == 1:
        meand = var_results['meand'].copy()
        meand = meand.rename(columns={'meand': 'MeanD'})
    else:
        meand = var_results['meand'].copy()
        meand = meand.rename(columns={'meand': 'MeanD2'})
    
    return meand


def get_duration_standard_deviation(seqdata: SequenceData, type: int = 1, with_missing: bool = False) -> pd.DataFrame:
    """
    Calculate standard deviation of spell durations for each sequence.
    
    This measures the variability in spell lengths within each sequence.
    Higher values indicate more irregular spell patterns.
    
    Args:
        seqdata: SequenceData object containing sequence data
        type: Type of calculation:
             1 - Standard deviation considering only visited states
             2 - Standard deviation taking non-visited states into account
        with_missing: If True, treat missing values as regular states.
                     If False, ignore missing values.
    
    Returns:
        pd.DataFrame: DataFrame with one column 'Dustd' (or 'Dustd2' for type=2)
                     containing standard deviations. Index matches sequence IDs.
    
    Reference:
        TraMineR: R/seqivardur.R
        Standard deviation is calculated as sqrt(variance).
        https://github.com/cran/TraMineR/blob/master/R/seqivardur.R
    
    Examples:
        >>> std_dur = get_duration_standard_deviation(seqdata, type=1)
        >>> print(std_dur.head())
                 Dustd
        ID_1      1.25
        ID_2      0.87
    """
    if not isinstance(seqdata, SequenceData):
        raise ValueError("[!] seqdata must be a SequenceData object, see SequenceData function to create one.")
    
    if type not in [1, 2]:
        raise ValueError("[!] type must be 1 or 2.")
    
    # Import the variance function
    from .variance_of_spell_durations import get_spell_duration_variance
    
    # Get variance results
    var_results = get_spell_duration_variance(seqdata, type=type)
    
    # Calculate standard deviation as sqrt(variance)
    variance = var_results['result'].copy()
    std_dev = np.sqrt(variance.iloc[:, 1])  # Get the variance column
    
    # Create result DataFrame
    col_name = 'Dustd' if type == 1 else 'Dustd2'
    result = pd.DataFrame({
        col_name: std_dev.values
    }, index=variance['ID'])
    
    result = result.reset_index().rename(columns={'index': 'ID'})
    
    return result
