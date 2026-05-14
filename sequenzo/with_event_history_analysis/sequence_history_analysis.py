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

    period_values = data[period_col].values
    if pd.api.types.is_bool_dtype(data[period_col]):
        raise ValueError("'time' must contain integer durations, not boolean values.")
    try:
        period_values_int = period_values.astype(int)
    except (TypeError, ValueError):
        raise ValueError("'time' must contain integer durations.") from None
    if not np.all(np.equal(period_values, period_values_int)):
        raise ValueError("'time' must contain integer durations.")
    period_values = period_values_int
    if (period_values < 1).any():
        raise ValueError("'time' must be at least 1 for every sequence.")

    event_values = data[event_col].values
    unique_events = set(pd.unique(event_values))
    if not unique_events.issubset({0, 1, False, True}):
        raise ValueError("'event' must be boolean or 0/1.")

    # Create an index that repeats each row based on the time value
    # For example, if time=3, that row will be repeated 3 times
    index = np.repeat(np.arange(len(data)), period_values)
    
    # Find the cumulative sum to identify which rows should have the event
    idmax = np.cumsum(period_values) - 1
    
    # Expand the data by repeating rows
    dat = data.iloc[index].copy()
    dat.reset_index(drop=True, inplace=True)
    
    # Create sequential time periods for each ID (1, 2, 3, ...)
    dat[period_col] = dat.groupby(id_col).cumcount() + 1
    
    # Set all events to False initially
    dat[event_col] = False
    
    # Set events to True only at the final period for each person
    dat.loc[idmax, event_col] = event_values.astype(bool)
    
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


def get_sequence_history_data(seqdata, time, event, include_present=False, align_end=False, covar=None):
    """
    Build person-period data with sequence history for Sequence History Analysis (SHA).
    
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
    >>> result = get_sequence_history_data(seqdata, time, event)
    
    Example 2: Usage with SequenceData object (recommended)
    >>> from sequenzo import SequenceData, load_dataset
    >>> df = load_dataset('pairfam_family')
    >>> time_cols = [str(i) for i in range(1, 265)]
    >>> seq_data = SequenceData(df, time=time_cols, id_col='id', 
    ...                          states=list(range(1, 10)))
    >>> # All persons observed for 264 months
    >>> time = np.full(len(df), 264)
    >>> event = df['highschool'].values
    >>> result = get_sequence_history_data(seq_data, time, event)
    
    Example 3: With covariates
    >>> covar = df[['sex', 'yeduc', 'east']]
    >>> result = get_sequence_history_data(seq_data, time, event, covar=covar)
    
    Example 4: Right-aligned sequences
    >>> result = get_sequence_history_data(seq_data, time, event, align_end=True)
    
    Notes
    -----
    This function implements the first step of Sequence History Analysis (SHA):
    reconstructing person-period data with past trajectories at each time point.
    Subsequent steps (sequence typology and discrete-time event-history modeling)
    are not performed here.

    - The time parameter represents observation duration, not calendar time
    - When include_present=False (default), the current state is excluded from history
    - When align_end=False, only the first ``ma`` sequence columns are returned
      (``ma`` is the maximum observation duration across persons)
    - Use align_end=True when analyzing sequences leading up to an event
    - Missing values in the original sequence are converted to "NA_orig"
    - Sequence states are treated as categorical labels and converted to strings
    - Left-aligned ``include_present`` follows SHA semantics and differs from
      TraMineRextras::seqsha, where the parameter meaning is inverted
    """
    # Extract sequence DataFrame from input (handles SequenceData, DataFrame, or array)
    seq_df = _extract_sequence_dataframe(seqdata)

    if seq_df.shape[0] == 0:
        raise ValueError("'seqdata' must contain at least one sequence.")
    if seq_df.shape[1] == 0:
        raise ValueError("'seqdata' must contain at least one time column.")
    
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
    seq_values = seq_df.to_numpy(dtype=object)
    missing_mask = pd.isna(seq_values)
    seq_values = seq_values.copy()
    seq_values[missing_mask] = "NA_orig"
    sdata = seq_values.astype(str)
    
    # Get the time periods for each row in person-period data
    age = persper['time'].values
    ma = int(np.max(age))
    
    # Check if time values are valid
    if ma > seq_df.shape[1]:
        raise ValueError("Maximum time of event occurrence is higher than the longest sequence!")
    
    n_cols = seq_df.shape[1]
    history_cols = n_cols if align_end else ma

    # Create empty matrix to store past sequence states
    past = np.full((len(persper), history_cols), np.nan, dtype=object)
    
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
                past[cond, (n_cols - aa):n_cols] = sdata[ids_a, 0:aa]
            else:
                # Exclude current time point: fill from (ncol-aa+1) to end
                past[cond, (n_cols - aa + 1):n_cols] = sdata[ids_a, 0:(aa - 1)]
        
        # Create column names counting backwards
        col_names = [f"Tm{i}" for i in range(n_cols, 0, -1)]
    else:
        # Left-align the sequences (align from the start)
        for aa in range(1, ma + 1):
            if include_present:
                cond = age >= aa
            else:
                cond = age > aa

            ids_a = persper.loc[cond, 'id'].values - 1
            past[cond, aa - 1] = sdata[ids_a, aa - 1]

        if seq_df.columns is not None and len(seq_df.columns) > 0:
            col_names = [str(col) for col in seq_df.columns[:ma]]
        else:
            col_names = [f"col_{i}" for i in range(ma)]

    base_cols = {"id", "time", "event"}
    overlap = base_cols.intersection(col_names)
    if overlap:
        raise ValueError(
            f"Sequence history columns duplicate base columns: {sorted(overlap)}. "
            "Please rename sequence time columns before calling this function."
        )

    # Convert past matrix to DataFrame
    past_df = pd.DataFrame(past, columns=col_names)
    
    # Combine person-period data with sequence history
    alldata = pd.concat([persper.reset_index(drop=True), past_df], axis=1)
    
    # Add covariates if provided
    if covar is not None:
        if isinstance(covar, pd.DataFrame):
            if len(covar) != n_sequences:
                raise ValueError(
                    f"Number of rows in 'covar' ({len(covar)}) must match "
                    f"number of sequences ({n_sequences})."
                )
            covar_subset = covar.iloc[alldata['id'].values - 1].reset_index(drop=True)
            overlap = set(covar_subset.columns).intersection(alldata.columns)
            if overlap:
                raise ValueError(
                    f"Covariate columns duplicate existing columns: {sorted(overlap)}"
                )
            alldata = pd.concat([alldata, covar_subset], axis=1)
        else:
            covar_array = np.asarray(covar)
            if covar_array.shape[0] != n_sequences:
                raise ValueError(
                    f"Number of rows in 'covar' ({covar_array.shape[0]}) must match "
                    f"number of sequences ({n_sequences})."
                )
            covar_subset = covar_array[alldata['id'].values - 1]
            alldata = pd.concat([alldata, pd.DataFrame(covar_subset)], axis=1)

    return alldata