"""
@Author  : Yuqi Liang 梁彧祺
@File    : sequence_history_analysis.py
@Time    : 30/09/2025 21:08
@Desc    : Sequence History Analysis - Convert person-level sequence data to person-period format
"""

import numpy as np
import pandas as pd


def person_level_to_person_period(data, id_col="id", period_col="time", event_col="event"):
    """
    Convert person-level data to person-period format.
    
    This function expands each person's single row into multiple rows,
    one for each time period they are observed.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data with one row per person
    id_col : str, optional
        Name of the ID column (default: "id")
    period_col : str, optional
        Name of the time period column (default: "time")
    event_col : str, optional
        Name of the event indicator column (default: "event")
    
    Returns
    -------
    pandas.DataFrame
        Expanded data with one row per person-period
    
    Examples
    --------
    >>> data = pd.DataFrame({'id': [1, 2], 'time': [3, 2], 'event': [True, False]})
    >>> person_level_to_person_period(data)
       id  time  event
    0   1     1  False
    1   1     2  False
    2   1     3   True
    3   2     1  False
    4   2     2  False
    """
    # Check for missing values in critical columns
    if data[[id_col, period_col, event_col]].isna().any().any():
        raise ValueError("Cannot handle missing data in the time or event variables")
    
    # Create an index that repeats each row based on the time value
    # For example, if time=3, that row will be repeated 3 times
    index = np.repeat(np.arange(len(data)), data[period_col].values)
    
    # Find the cumulative sum to identify which rows should have the event
    idmax = np.cumsum(data[period_col].values) - 1
    
    # Expand the data by repeating rows
    dat = data.iloc[index].copy()
    dat.reset_index(drop=True, inplace=True)
    
    # Create sequential time periods for each ID (1, 2, 3, ...)
    dat[period_col] = dat.groupby(id_col).cumcount() + 1
    
    # Set all events to False initially
    dat[event_col] = False
    
    # Set events to True only at the final period for each person
    # Convert to bool to avoid dtype incompatibility warning
    dat.loc[idmax, event_col] = data[event_col].values.astype(bool)
    
    return dat


def _extract_sequence_dataframe(seqdata):
    """
    Extract sequence DataFrame from various input types.
    
    Parameters
    ----------
    seqdata : SequenceData, pandas.DataFrame, or numpy.ndarray
        Input sequence data
    
    Returns
    -------
    pandas.DataFrame
        Sequence data as a DataFrame
    """
    # Check if input is a SequenceData object
    if hasattr(seqdata, 'seqdata'):
        # This is a SequenceData object
        return seqdata.seqdata.copy()
    elif isinstance(seqdata, pd.DataFrame):
        return seqdata.copy()
    else:
        # Assume it's array-like
        return pd.DataFrame(seqdata)


def seqsha(seqdata, time, event, include_present=False, align_end=False, covar=None):
    """
    Sequence History Analysis: Create person-period format with sequence history.
    
    This function converts sequence data into a person-period format where each
    row represents a time point for a person, with columns showing their sequence
    history up to that point.
    
    Parameters
    ----------
    seqdata : SequenceData, pandas.DataFrame, or numpy.ndarray
        Sequence data where each row is a person and each column is a time point.
        Can be a SequenceData object, DataFrame, or array.
    time : array-like
        Duration or time until event for each person. Length should equal the 
        number of sequences. Each value indicates how many time periods that 
        person is observed. For example, if all persons are observed for the 
        full sequence length, use: np.full(n_persons, sequence_length)
    event : array-like
        Event indicator for each person (True/False or 1/0). Length should 
        equal the number of sequences.
    include_present : bool, optional
        If True, include the current time point in the history (default: False)
        If False, only include past time points (recommended for most analyses)
    align_end : bool, optional
        If True, align sequences from the end (right-aligned) (default: False)
        If False, align sequences from the start (left-aligned)
    covar : pandas.DataFrame or numpy.ndarray, optional
        Additional covariates to merge with the output (default: None)
        Should have the same number of rows as seqdata
    
    Returns
    -------
    pandas.DataFrame
        Person-period data with the following columns:
        - id: Person identifier
        - time: Time period within person
        - event: Event indicator (True only at the final period for each person)
        - Sequence history columns (varies based on align_end parameter)
        - Additional covariate columns (if covar is provided)
    
    Raises
    ------
    ValueError
        If maximum time exceeds the length of the longest sequence
    
    Examples
    --------
    Example 1: Basic usage with DataFrame
    >>> import pandas as pd
    >>> import numpy as np
    >>> seqdata = pd.DataFrame([[1, 2, 3, 4], [1, 1, 2, 2]])
    >>> time = np.array([3, 2])
    >>> event = np.array([True, False])
    >>> result = seqsha(seqdata, time, event)
    
    Example 2: Usage with SequenceData object (recommended)
    >>> from sequenzo import SequenceData, load_dataset
    >>> df = load_dataset('pairfam_family')
    >>> time_cols = [str(i) for i in range(1, 265)]
    >>> seq_data = SequenceData(df, time=time_cols, id_col='id', 
    ...                          states=list(range(1, 10)))
    >>> # All persons observed for 264 months
    >>> time = np.full(len(df), 264)
    >>> event = df['highschool'].values
    >>> result = seqsha(seq_data, time, event)
    
    Example 3: With covariates
    >>> covar = df[['sex', 'yeduc', 'east']]
    >>> result = seqsha(seq_data, time, event, covar=covar)
    
    Example 4: Right-aligned sequences
    >>> result = seqsha(seq_data, time, event, align_end=True)
    
    Notes
    -----
    - The time parameter represents observation duration, not calendar time
    - When include_present=False (default), only past states are included
    - Use align_end=True when analyzing sequences leading up to an event
    - Missing values in the original sequence are converted to "NA_orig"
    """
    # Extract sequence DataFrame from input (handles SequenceData, DataFrame, or array)
    seq_df = _extract_sequence_dataframe(seqdata)
    
    # Convert time and event to numpy arrays for consistency
    time_array = np.asarray(time)
    event_array = np.asarray(event)
    
    # Check that dimensions match
    n_sequences = len(seq_df)
    if len(time_array) != n_sequences:
        raise ValueError(
            f"Length of 'time' ({len(time_array)}) must match number of sequences ({n_sequences})"
        )
    if len(event_array) != n_sequences:
        raise ValueError(
            f"Length of 'event' ({len(event_array)}) must match number of sequences ({n_sequences})"
        )
    
    # Create base time data: one row per person with their time and event
    basetime = pd.DataFrame({
        'id': np.arange(1, n_sequences + 1),
        'time': time_array,
        'event': event_array
    })
    
    # Convert to person-period format (expand rows)
    persper = person_level_to_person_period(basetime, "id", "time", "event")
    
    # Convert sequence data to matrix and handle missing values
    sdata = seq_df.values.astype(str)
    sdata[pd.isna(seq_df.values)] = "NA_orig"
    
    # Get the time periods for each row in person-period data
    age = persper['time'].values
    ma = int(np.max(age))
    
    # Check if time values are valid
    if ma > seq_df.shape[1]:
        raise ValueError("Maximum time of event occurrence is higher than the longest sequence!")
    
    # Create empty matrix to store past sequence states
    past = np.full((len(persper), seq_df.shape[1]), np.nan, dtype=object)
    
    if align_end:
        # Right-align the sequences (align from the end)
        start = 1 if include_present else 2
        
        for aa in range(start, ma + 1):
            # Find rows where time equals aa
            cond = age == aa
            # Get the person IDs for these rows
            ids_a = persper.loc[cond, 'id'].values - 1  # Subtract 1 for 0-based indexing
            
            if include_present:
                # Include current time point: fill from (ncol-aa) to end
                past[cond, (seq_df.shape[1] - aa):seq_df.shape[1]] = sdata[ids_a, 0:aa]
            else:
                # Exclude current time point: fill from (ncol-aa+1) to end
                past[cond, (seq_df.shape[1] - aa + 1):seq_df.shape[1]] = sdata[ids_a, 0:(aa - 1)]
        
        # Create column names counting backwards
        col_names = [f"Tm{i}" for i in range(seq_df.shape[1], 0, -1)]
    else:
        # Left-align the sequences (align from the start)
        for aa in range(1, ma + 1):
            if include_present:
                # Include present: use time > aa
                cond = age > aa
            else:
                # Exclude present: use time >= aa
                cond = age >= aa
            
            # Get the person IDs for these rows
            ids_a = persper.loc[cond, 'id'].values - 1  # Subtract 1 for 0-based indexing
            
            # Fill in the sequence state at position aa-1 (0-based)
            past[cond, aa - 1] = sdata[ids_a, aa - 1]
        
        # Use original column names or create default ones
        if seq_df.columns is not None and len(seq_df.columns) > 0:
            col_names = [str(col) for col in seq_df.columns[:ma]]
            # Pad with additional column names if needed
            col_names += [f"col_{i}" for i in range(ma, seq_df.shape[1])]
        else:
            col_names = [f"col_{i}" for i in range(seq_df.shape[1])]
    
    # Convert past matrix to DataFrame
    past_df = pd.DataFrame(past, columns=col_names)
    
    # Combine person-period data with sequence history
    alldata = pd.concat([persper.reset_index(drop=True), past_df], axis=1)
    
    # Add covariates if provided
    if covar is not None:
        # Merge covariates based on the ID (subtract 1 for 0-based indexing)
        if isinstance(covar, pd.DataFrame):
            covar_subset = covar.iloc[alldata['id'].values - 1].reset_index(drop=True)
            alldata = pd.concat([alldata, covar_subset], axis=1)
        else:
            covar_array = np.array(covar)
            covar_subset = covar_array[alldata['id'].values - 1]
            alldata = pd.concat([alldata, pd.DataFrame(covar_subset)], axis=1)
    
    return alldata