"""
@Author  : Yuqi Liang 梁彧祺
@File    : sequence_analysis_multi_state_model.py
@Time    : 30/09/2025 20:27
@Desc    : Sequence Analysis Multi-state Model (SAMM) for event history analysis
           
           This module provides tools for analyzing sequences through a multi-state perspective,
           creating person-period datasets that can be used for event history analysis.
           
           Based on the TraMineR package's SAMM functionality.

IMPORTANT DIFFERENCES FROM R'S TraMineR IMPLEMENTATION:

Plotting Approach Differences:

R's plot.SAMM() function:
  - Uses TraMineR's seqplot() function with grouping
  - Original R code: plot.SAMM <- function(x, type="d", ...){
                       seqdata <- attr(x, "stslist")[x$transition,]
                       group <- x[x$transition, attr(x, "sname")[1]]
                       levels(group) <- paste("Transition out of", levels(group))
                       seqplot(seqdata, group=group, type=type, ...)
                     }
  - Creates grouped sequence plots where sequences are grouped by starting state
  - Relies on TraMineR's built-in plotting system
  - Links: 
        Source code: https://rdrr.io/cran/TraMineRextras/src/R/seqsamm.R
        Documentation: https://cran.r-project.org/web/packages/TraMineRextras/refman/TraMineRextras.html#seqsha 

Our Python plot_samm() function:
  - Uses matplotlib's imshow() with sequence index plot approach
  - Creates separate subplots for each starting state (one subplot per transition state)
  - Each subplot shows all subsequences that start with a specific state as colored horizontal bars
  - Displays actual sequence patterns using a color-coded matrix visualization
  - Automatically handles varying numbers of sequences per state with dynamic subplot heights

Why We Made This Choice:
  1. Better Visual Separation: Each starting state gets its own dedicated subplot,
     making it easier to compare patterns across different states
  2. Scalability: Works well with large numbers of sequences and states
  3. Clarity: Direct visualization of subsequence patterns without grouping artifacts
  4. Python Ecosystem: Leverages matplotlib's powerful visualization capabilities
  5. Detail Preservation: Shows individual sequence patterns rather than aggregate summaries

Both approaches show transition patterns effectively, but our Python implementation 
provides more detailed, subplot-based visualizations that are particularly suitable
for exploratory data analysis and detailed pattern inspection.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple
import matplotlib.pyplot as plt

# Import the SequenceData class from the parent package
from sequenzo.define_sequence_data import SequenceData


class SAMM:
    """
    Sequence Analysis Multi-state Model (SAMM) object.
    
    This class stores a person-period dataset generated from sequence data,
    where each row represents one time point for one person, along with
    information about subsequences, transitions, and spell characteristics.
    
    Attributes:
        data (pd.DataFrame): The person-period dataset
        alphabet (list): The state space (unique states in the sequences)
        labels (list): Labels for the states
        color_map (dict): Color mapping for visualization
        sname (list): Column names for subsequence variables (e.g., ['s.1', 's.2', 's.3'])
        sublength (int): Length of the subsequences being tracked
    """
    
    def __init__(self, data: pd.DataFrame, alphabet: list, labels: list, 
                 color_map: dict, sname: list, sublength: int):
        """
        Initialize a SAMM object.
        
        Args:
            data: Person-period dataset
            alphabet: List of unique states
            labels: Labels for states
            color_map: Dictionary mapping states to colors
            sname: List of subsequence column names
            sublength: Length of subsequences
        """
        self.data = data
        self.alphabet = alphabet
        self.labels = labels
        self.color_map = color_map
        self.sname = sname
        self.sublength = sublength
        
        # Initialize typology column (will be set later using set_typology)
        if 'typology' not in self.data.columns:
            self.data['typology'] = 'None'
    
    def __repr__(self):
        """String representation of SAMM object."""
        return f"SAMM(n_rows={len(self.data)}, sublength={self.sublength})"
    
    def __len__(self):
        """Return number of rows in the person-period dataset."""
        return len(self.data)


def sequence_analysis_multi_state_model(seqdata: SequenceData, sublength: int, covar: Optional[pd.DataFrame] = None) -> SAMM:
    """
    Generate a person-period dataset from sequence data for multi-state analysis.
    
    This function transforms sequence data into a "person-period" format where each row 
    represents one time point for one individual. At each time position, it also extracts
    the subsequence for the next 'sublength' time units.
    
    **What is person-period data?**
    Instead of having one row per person with all their time points as columns,
    person-period data has one row for each person-time combination. For example,
    if we track 3 people over 5 time periods, we get 15 rows (3 x 5).
    
    **What are subsequences?**
    At each time point, we look ahead and record what happens in the next few time periods.
    For example, if sublength=3 and we're at time 2, we record states at time 2, 3, and 4.
    
    Args:
        seqdata (SequenceData): A SequenceData object containing your sequence data.
                                This should be created using the SequenceData class.
        sublength (int): The length of the subsequence to extract at each time point.
                        For example, if sublength=3, we look 3 steps ahead from each position.
        covar (pd.DataFrame, optional): Time-invariant covariates (variables that don't change over time).
                                       For example: gender, education level, birth year, etc.
                                       The row index should match the sequence IDs.
    
    Returns:
        SAMM: A SAMM object containing the person-period dataset with the following variables:
            - id: Identifier for each sequence/person
            - time: Time elapsed since the beginning of the sequence (starts at 1)
            - begin: Time when the current spell began
            - spell_time: Time elapsed since the beginning of the current spell
            - transition: Boolean indicator (True if there's a state transition at this point)
            - s.1, s.2, ..., s.X: The subsequence values (number depends on sublength)
            - Additional covariate columns (if covar was provided)
    
    Example:
        >>> # Suppose we have sequence data tracking employment states
        >>> # States: 'employed', 'unemployed', 'education'
        >>> # We want to analyze what happens in the next 3 time periods
        >>> samm_obj = sequence_analysis_multi_state_model(my_seqdata, sublength=3)
        >>> # Now we can use this for event history analysis
    """
    
    # Extract the sequence data as a numpy array (rows=individuals, columns=time points)
    # Each cell contains a numeric code representing a state (1, 2, 3, etc.)
    seqdata_array = seqdata.values
    n_individuals = seqdata_array.shape[0]  # Number of sequences/people
    n_timepoints = seqdata_array.shape[1]   # Length of each sequence
    
    # Create column names for the subsequence variables
    # For example, if sublength=3, this creates ['s.1', 's.2', 's.3']
    sname = [f's.{i+1}' for i in range(sublength)]
    
    # Get the IDs for each sequence
    # If the SequenceData has an ID column, use it; otherwise use row numbers
    if seqdata.id_col is not None:
        id_values = seqdata.ids
    else:
        id_values = np.arange(1, n_individuals + 1)
    
    # This will store all the person-period rows as we process each time point
    all_subseq_list = []
    
    # Track when each individual's current spell began
    # A "spell" is a continuous period in the same state
    # Initialize: everyone's spell begins at time 1
    spell_begin = np.ones(n_individuals, dtype=int)
    
    # Loop through each time point (but stop before the end to ensure we have enough future points)
    # For example, if sublength=3 and we have 10 time points, we only go up to time point 7
    # because from time 8, 9, 10 we can't look 3 steps ahead
    for tt in range(n_timepoints - sublength + 1):
        
        # Extract the subsequence starting at time 'tt' and going for 'sublength' time units
        # For example, if tt=2 and sublength=3, extract columns 2, 3, 4
        subseq = seqdata_array[:, tt:(tt + sublength)]
        
        # Create a DataFrame for this subsequence with proper column names
        subseq_df = pd.DataFrame(subseq, columns=sname)
        
        # Detect transitions: A transition occurs when the state changes from this time to the next
        # Compare the first column (current state) with the second column (next state)
        transition = (subseq_df['s.1'].values != subseq_df['s.2'].values)
        
        # Update spell begin times
        # If this isn't the first time point, check if there was a state change from previous time
        if tt > 0:
            # Get the state at the previous time point and current time point
            prev_state = seqdata_array[:, tt - 1]
            curr_state = seqdata_array[:, tt]
            
            # Find where the state changed (spell reset)
            spell_reset_mask = (prev_state != curr_state)
            
            # For those individuals, update their spell begin time to current time (tt + 1, since time starts at 1)
            spell_begin[spell_reset_mask] = tt + 1
        
        # Calculate spell duration: how long has the current spell lasted?
        # This is the current time minus when the spell began
        spell_time = (tt + 1) - spell_begin
        
        # Create the person-period dataset for this time point
        # Each row represents one individual at this specific time point
        subseq_record = pd.DataFrame({
            'id': id_values,                    # Individual identifier
            'time': tt + 1,                     # Current time point (1-indexed)
            'begin': spell_begin,               # When current spell began
            'spell_time': spell_time,           # Duration of current spell
            'transition': transition            # Whether transition occurs
        })
        
        # Add the subsequence columns (s.1, s.2, ..., s.X)
        subseq_record = pd.concat([subseq_record, subseq_df], axis=1)
        
        # Add this time point's data to our collection
        all_subseq_list.append(subseq_record)
    
    # Combine all time points into one large person-period dataset
    # Stack all the DataFrames on top of each other
    result = pd.concat(all_subseq_list, ignore_index=True)
    
    # If time-invariant covariates were provided, merge them in
    # These are variables that don't change over time (e.g., gender, birth year)
    if covar is not None:
        # Match covariates to IDs
        covar_with_id = covar.copy()
        covar_with_id['id'] = id_values
        
        # Merge the covariates into our person-period data based on ID
        result = result.merge(covar_with_id, on='id', how='left')
    
    # Sort the data by ID and time for easier reading
    result = result.sort_values(['id', 'time']).reset_index(drop=True)
    
    # Map numeric state codes back to their readable labels
    # First map to states, then to labels for better interpretability
    inverse_mapping = seqdata.inverse_state_mapping  # Maps numeric codes to states
    state_to_label = seqdata.state_to_label  # Maps states to descriptive labels
    
    for col in sname:
        # First convert numeric codes to states
        result[col] = result[col].map(inverse_mapping)
        # Then convert states to labels for better readability
        result[col] = result[col].map(state_to_label)
    
    # Create and return the SAMM object
    samm_obj = SAMM(
        data=result,
        alphabet=seqdata.alphabet,
        labels=seqdata.labels,
        color_map=seqdata.color_map,
        sname=sname,
        sublength=sublength
    )
    
    return samm_obj


def plot_samm(samm: SAMM, plot_type: str = "d", base_width: int = 15, 
              title: Optional[str] = None, save_as: Optional[str] = None, 
              dpi: int = 200, fontsize: int = 10):
    """
    Plot subsequences following transitions in the SAMM data using sequence index plots.
    
    This function creates sequence index visualizations showing what subsequences occur 
    after transitions out of each state. Similar to R's TraMineR seqplot function.
    
    **What does this show?**
    For each state, this displays the actual subsequence patterns (as colored bars) 
    that occur when individuals transition OUT of that state. Each row is one sequence,
    and colors represent different states in the subsequence.
    
    Args:
        samm (SAMM): A SAMM object created by sequence_analysis_multi_state_model()
        plot_type (str): Type of plot to create (currently supports 'd' for sequence index plot)
        base_width (int): Base width for the figure. Default 15 (wider for better proportions).
        title (str, optional): Custom title for the plot
        save_as (str, optional): File path to save the plot (if None, plot is displayed)
        dpi (int): Resolution for saved images
        fontsize (int): Base font size for labels and titles
    
    Example:
        >>> samm_obj = sequence_analysis_multi_state_model(my_seqdata, sublength=3)
        >>> plot_samm(samm_obj, title="Transition Patterns")
    """
    
    # Import visualization utilities
    from io import BytesIO
    from sequenzo.visualization.utils import (
        create_standalone_legend, 
        combine_plot_with_legend,
        save_figure_to_buffer
    )
    from matplotlib.colors import ListedColormap
    
    # Filter to only rows where a transition occurs
    transition_rows = samm.data[samm.data['transition'] == True].copy()
    
    if len(transition_rows) == 0:
        print("No transitions found in the data.")
        return
    
    # Group by the starting state (s.1) to see transitions out of each state
    starting_states = sorted(transition_rows['s.1'].unique())
    
    # Create subplots: one for each starting state
    n_states = len(starting_states)
    ncols = min(3, n_states)  # Maximum 3 columns
    nrows = int(np.ceil(n_states / ncols))
    
    # Calculate dynamic heights for each subplot based on number of sequences
    # We'll use gridspec to allow different heights
    from matplotlib import gridspec
    
    # First, count sequences for each state to determine heights
    state_seq_counts = {}
    for state in starting_states:
        state_seq_counts[state] = len(transition_rows[transition_rows['s.1'] == state])
    
    # Calculate height ratios - base height per sequence, min 2.5, max 5 for better aspect ratio
    height_ratios = []
    for i in range(nrows):
        row_states = starting_states[i*ncols : (i+1)*ncols]
        if row_states:
            max_seqs_in_row = max([state_seq_counts[s] for s in row_states])
            # Height: 2.5-5 inches, scaled by number of sequences
            # Use smaller scaling factor (0.01 instead of 0.015) to make plots less stretched
            height = min(5, max(2.5, max_seqs_in_row * 0.01))
            height_ratios.append(height)
    
    # Calculate total figure height with more spacing
    total_height = sum(height_ratios) + (nrows - 1) * 2.0  # Add more spacing between rows
    
    # Create figure with GridSpec for flexible heights
    fig = plt.figure(figsize=(base_width, total_height))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, height_ratios=height_ratios,
                          hspace=0.5, wspace=0.25)  # Adjusted spacing for better layout
    
    # Create a reverse mapping from labels back to numeric codes for plotting
    label_to_numeric = {label: i + 1 for i, label in enumerate(samm.labels)}
    
    # Use the color map from the original sequence data
    cmap = ListedColormap([samm.color_map[i] for i in sorted(samm.color_map.keys())])
    
    # For each starting state, create a sequence index plot
    for idx, state in enumerate(starting_states):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        
        # Get all subsequences that start with this state and have a transition
        state_data = transition_rows[transition_rows['s.1'] == state].copy()
        
        # Extract subsequence columns and convert labels to numeric codes
        subseq_cols = samm.sname
        subseq_matrix = state_data[subseq_cols].values
        
        # Convert label strings to numeric codes for plotting
        numeric_matrix = np.zeros_like(subseq_matrix, dtype=float)
        for i in range(subseq_matrix.shape[0]):
            for j in range(subseq_matrix.shape[1]):
                label = subseq_matrix[i, j]
                if pd.notna(label) and label in label_to_numeric:
                    numeric_matrix[i, j] = label_to_numeric[label]
                else:
                    numeric_matrix[i, j] = np.nan
        
        # Plot with masked array for NaN handling
        ax.imshow(np.ma.masked_invalid(numeric_matrix), 
                 aspect='auto', 
                 cmap=cmap, 
                 interpolation='nearest',
                 vmin=1, 
                 vmax=len(samm.labels))
        
        # Disable grid
        ax.grid(False)
        
        # Set title showing the starting state with count
        num_seqs = numeric_matrix.shape[0]
        title_text = f'Transitions out of: {state} (n={num_seqs})'
        
        # Break long titles into multiple lines
        if len(title_text) > 35:  # If title is too long
            # Try to break at a natural point
            if 'Transitions out of:' in title_text:
                parts = title_text.split('Transitions out of:')
                if len(parts) == 2:
                    title_text = f'Transitions out of:\n{parts[1].strip()}'
        
        ax.set_title(title_text, fontsize=fontsize+1, pad=12, color='black')
        
        # X-axis: time steps in subsequence
        ax.set_xlabel('Subsequence Position', fontsize=fontsize, labelpad=8, color='black')
        xticks = np.arange(len(subseq_cols))
        ax.set_xticks(xticks)
        ax.set_xticklabels([f't+{i}' for i in range(len(subseq_cols))], 
                          fontsize=fontsize-2, color='gray')
        
        # Y-axis: sequence count
        ax.set_ylabel('Sequences', fontsize=fontsize, labelpad=8, color='black')
        
        # Smart y-tick display based on sequence count
        if num_seqs <= 10:
            yticks = np.arange(num_seqs)
            ax.set_yticks(yticks)
            ax.set_yticklabels(range(1, num_seqs + 1), fontsize=fontsize-2, color='gray')
        elif num_seqs <= 50:
            # Show every 5th or 10th
            step = 5 if num_seqs <= 25 else 10
            yticks = np.arange(0, num_seqs, step)
            if yticks[-1] != num_seqs - 1:
                yticks = np.append(yticks, num_seqs - 1)
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(y + 1) for y in yticks], fontsize=fontsize-2, color='gray')
        else:
            # Show quartiles for large numbers
            ytick_positions = [0, num_seqs // 4, num_seqs // 2, 3 * num_seqs // 4, num_seqs - 1]
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels([str(pos + 1) for pos in ytick_positions], 
                              fontsize=fontsize-2, color='gray')
        
        # Style axis spines and ticks like index plot
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('gray')
            ax.spines[spine].set_linewidth(0.8)
        
        # Tick parameters matching index plot style
        ax.tick_params(axis='x', colors='gray', length=4, width=0.7, which='major')
        ax.tick_params(axis='y', colors='gray', length=4, width=0.7, which='major')
        ax.tick_params(axis='both', which='major', direction='out')
    
    # Adjust layout first
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave less space at top for title
    
    # Add overall title if provided (after tight_layout to prevent overlap)
    if title:
        fig.suptitle(title, fontsize=fontsize+4, y=0.93, color='black')
    
    # Save main figure to buffer
    main_buffer = save_figure_to_buffer(fig, dpi=dpi)
    
    # Create standalone legend using the same style as index plot
    colors = {samm.labels[i]: samm.color_map[i+1] for i in range(len(samm.labels))}
    legend_buffer = create_standalone_legend(
        colors=colors,
        labels=samm.labels,
        ncol=min(5, len(samm.labels)),
        figsize=(base_width, 1),
        fontsize=fontsize,
        dpi=dpi
    )
    
    # Combine plot with legend
    if save_as and not save_as.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        save_as = save_as + '.png'
    
    combined_img = combine_plot_with_legend(
        main_buffer,
        legend_buffer,
        output_path=save_as,
        dpi=dpi,
        padding=20
    )
    
    # Display combined image
    plt.figure(figsize=(base_width, total_height + 1))
    plt.imshow(combined_img)
    plt.axis('off')
    plt.show()
    plt.close('all')


def seqsammseq(samm: SAMM, spell: str) -> pd.DataFrame:
    """
    Extract subsequences that follow a specific state (spell).
    
    This function returns all the subsequences that occur after a given state,
    specifically when there is a transition OUT of that state.
    
    **Why is this useful?**
    It helps you analyze what happens after a particular state. For example,
    if you're studying employment sequences, you might want to know:
    "What happens after someone becomes unemployed?" or
    "What patterns follow graduation?"
    
    Args:
        samm (SAMM): A SAMM object created by sequence_analysis_multi_state_model()
        spell (str): The state you want to analyze transitions from
                     (e.g., 'employed', 'single', 'education')
    
    Returns:
        pd.DataFrame: A DataFrame containing only the subsequence columns (s.1, s.2, ...)
                     for rows where:
                     1. The starting state (s.1) matches the specified spell
                     2. A transition occurs at that point
    
    Example:
        >>> # Get all subsequences following unemployment
        >>> unemployed_subsequences = seqsammseq(samm_obj, spell='unemployed')
        >>> print(unemployed_subsequences.head())
        # This shows what typically happens after someone becomes unemployed
    """
    
    # Filter for rows that:
    # 1. Start with the specified state (s.1 == spell)
    # 2. Have a transition occurring (transition == True)
    condition = (samm.data['s.1'] == spell) & (samm.data['transition'] == True)
    
    # Extract only the subsequence columns
    subsequences = samm.data.loc[condition, samm.sname].copy()
    
    # Reset index for cleaner output
    subsequences = subsequences.reset_index(drop=True)
    
    return subsequences


def _expand_typology_for_transitions(
    samm: SAMM,
    spell: str,
    mapping: Union[Dict, pd.Series],
    by: Optional[str] = None,
    cluster_to_name: Optional[Dict] = None
) -> np.ndarray:
    """
    Build a row-aligned typology vector for transition rows given a mapping.

    Parameters
    ----------
    samm : SAMM
        The SAMM object.
    spell : str
        The state to analyze transitions out of.
    mapping : dict or pandas.Series
        Either a mapping of id -> cluster/label, or (id, begin) -> cluster/label.
        Values can be final label strings, or cluster ids to be mapped via cluster_to_name.
    by : {"id", "id_begin"}, optional
        If None, auto-detect by inspecting mapping keys/index. Use "id_begin" when
        mapping is keyed by (id, begin).
    cluster_to_name : dict, optional
        Mapping from cluster id to human-readable label. Required if mapping values
        are cluster ids rather than label strings.

    Returns
    -------
    numpy.ndarray
        A vector of labels aligned to samm.data.loc[(s.1==spell) & transition].
    """
    condition = (samm.data['s.1'] == spell) & (samm.data['transition'] == True)
    trans_df = samm.data.loc[condition, ['id', 'begin']].copy()

    # Normalize mapping to a dict for fast lookup
    if isinstance(mapping, pd.Series):
        if mapping.index.nlevels == 1:
            normalized: Dict = mapping.to_dict()
            inferred_by = 'id'
        elif mapping.index.nlevels == 2:
            normalized = {tuple(idx): val for idx, val in mapping.items()}
            inferred_by = 'id_begin'
        else:
            raise ValueError("Mapping Series index must be 1 or 2 levels: id or (id, begin)")
    else:
        normalized = dict(mapping)
        # Auto-detect key type when by is not provided
        if by is None:
            if len(normalized) == 0:
                inferred_by = 'id'  # default
            else:
                sample_key = next(iter(normalized.keys()))
                inferred_by = 'id_begin' if isinstance(sample_key, tuple) and len(sample_key) == 2 else 'id'
        else:
            inferred_by = by

    labels: List[str] = []
    missing_keys: List[Union[int, Tuple[int, int]]] = []

    if inferred_by == 'id':
        for pid in trans_df['id'].tolist():
            if pid not in normalized:
                missing_keys.append(pid)
                labels.append(None)
                continue
            val = normalized[pid]
            # If val is numeric-like and cluster_to_name is provided, map to name
            if cluster_to_name is not None and pd.notna(val):
                try:
                    labels.append(cluster_to_name[val])
                except KeyError:
                    raise ValueError(f"cluster_to_name is missing key {val!r} for id {pid}")
            else:
                labels.append(val)
    elif inferred_by == 'id_begin':
        ids = trans_df['id'].to_list()
        begins = trans_df['begin'].to_list()
        for pid, b in zip(ids, begins):
            key = (pid, b)
            if key not in normalized:
                missing_keys.append(key)
                labels.append(None)
                continue
            val = normalized[key]
            if cluster_to_name is not None and pd.notna(val):
                try:
                    labels.append(cluster_to_name[val])
                except KeyError:
                    raise ValueError(f"cluster_to_name is missing key {val!r} for (id, begin) {key}")
            else:
                labels.append(val)
    else:
        raise ValueError("Parameter 'by' must be one of {'id', 'id_begin'}")

    if missing_keys:
        sample = missing_keys[:5]
        raise ValueError(
            f"Missing {len(missing_keys)} keys in mapping for transitions from '{spell}'. "
            f"Examples: {sample}. You can provide (id, begin) or id mappings, "
            f"and use cluster_to_name to map cluster ids to names."
        )

    return np.asarray(labels, dtype=object)


def set_typology(
    samm: SAMM,
    spell: str,
    typology: Union[pd.Series, np.ndarray, list, None] = None,
    *,
    clusters: Optional[Union[pd.Series, np.ndarray, list]] = None,
    cluster_to_name: Optional[Dict] = None,
    mapping: Optional[Union[Dict, pd.Series]] = None,
    by: Optional[str] = None
) -> SAMM:
    """
    Assign a typology classification to subsequences following a specific state.
    
    This function allows you to categorize the different patterns that occur
    after transitioning out of a particular state. This is useful for creating
    meaningful groups for further analysis.
    
    **What is a typology?**
    A typology is a classification system. For example, after unemployment,
    you might classify subsequences as:
    - "Quick reemployment" (gets job within 3 months)
    - "Long-term unemployment" (stays unemployed > 6 months)
    - "Exit labor force" (moves to education or retirement)
    
    Args:
        samm (SAMM): A SAMM object created by sequence_analysis_multi_state_model()
        spell (str): The state for which you're setting typologies
        typology (array-like, optional): Final labels for each transition row (length = n_transitions).
        clusters (array-like, optional): Cluster ids per transition row (length = n_transitions).
        cluster_to_name (dict, optional): Mapping from cluster id -> label name. Used with clusters
                                          or when mapping values are cluster ids.
        mapping (dict or pandas.Series, optional): id -> cluster/label or (id, begin) -> cluster/label.
        by (str, optional): 'id' or 'id_begin'. If None, auto-detect from mapping keys.
    
    Returns:
        SAMM: The updated SAMM object with typology column filled in
    
    Example:
        >>> # First, identify transitions from unemployment
        >>> unemployed_transitions = (samm_obj.data['s.1'] == 'unemployed') & samm_obj.data['transition']
        >>> # Create your typology based on some logic
        >>> my_typology = ['quick_return', 'education', 'long_term', ...]  # One label per transition
        >>> # Apply the typology
        >>> samm_obj = set_typology(samm_obj, spell='unemployed', typology=my_typology)
    """
    
    # Find rows where: state is the specified spell AND there's a transition
    condition = (samm.data['s.1'] == spell) & (samm.data['transition'] == True)

    n_transitions = int(condition.sum())

    labels_array: Optional[np.ndarray] = None

    # Case 1: direct typology vector
    if typology is not None:
        if isinstance(typology, pd.Series):
            labels_array = typology.values
        else:
            labels_array = np.asarray(typology, dtype=object)
        if len(labels_array) != n_transitions:
            raise ValueError(
                f"Length mismatch: provided length {len(labels_array)} but there are {n_transitions} "
                f"transitions from state '{spell}'. You should provide a typology vector of length n_transitions "
                f"(one label per transition row), not a list of unique type names. Use clusters+cluster_to_name "
                f"or mapping parameters instead."
            )

    # Case 2: clusters aligned to transition rows + mapping dict
    elif clusters is not None:
        clusters_array = clusters.values if isinstance(clusters, pd.Series) else np.asarray(clusters)
        if len(clusters_array) != n_transitions:
            raise ValueError(
                f"Length mismatch: clusters length {len(clusters_array)} must match n_transitions={n_transitions}"
            )
        if cluster_to_name is not None:
            try:
                labels_array = np.asarray([cluster_to_name[c] for c in clusters_array], dtype=object)
            except KeyError as e:
                raise ValueError(f"cluster_to_name is missing key {e.args[0]!r}")
        else:
            # Assume clusters are already label strings
            labels_array = clusters_array.astype(object)

    # Case 3: mapping keyed by id or (id, begin)
    elif mapping is not None:
        labels_array = _expand_typology_for_transitions(
            samm=samm, spell=spell, mapping=mapping, by=by, cluster_to_name=cluster_to_name
        )

    else:
        raise ValueError(
            "You must provide one of: typology (row-aligned), clusters+cluster_to_name (row-aligned), "
            "or mapping (id or (id, begin) to cluster/label)."
        )

    # Assign the typology labels to the corresponding rows
    samm.data.loc[condition, 'typology'] = labels_array
    
    return samm


def seqsammeha(
    samm: SAMM,
    spell: str,
    typology: Union[pd.Series, np.ndarray, list, None] = None,
    *,
    clusters: Optional[Union[pd.Series, np.ndarray, list]] = None,
    cluster_to_name: Optional[Dict] = None,
    mapping: Optional[Union[Dict, pd.Series]] = None,
    by: Optional[str] = None,
    persper: bool = True
) -> pd.DataFrame:
    """
    Generate a dataset for Event History Analysis (EHA) with typology outcomes.
    
    This function prepares your data for statistical models (like logistic regression
    or survival analysis) that estimate the probability of different outcomes
    following a specific state.
    
    **What is Event History Analysis?**
    EHA examines the timing and nature of events. For example:
    - "What factors predict returning to work after unemployment?"
    - "How long do people stay in education before entering the labor force?"
    
    **Person-period vs. Spell-level data:**
    - person-period (persper=True): One row for EACH time point in the spell
      Good for: Time-varying effects, duration dependence
    - spell-level (persper=False): One row per spell (only the last observation)
      Good for: Simpler models, overall spell outcomes
    
    Args:
        samm (SAMM): A SAMM object created by sequence_analysis_multi_state_model()
        spell (str): The state you're analyzing (e.g., 'unemployed', 'single')
        typology (array-like, optional): Final labels for each transition row (length = n_transitions)
        clusters (array-like, optional): Cluster ids per transition row (length = n_transitions)
        cluster_to_name (dict, optional): Mapping from cluster id -> label name
        mapping (dict or pandas.Series, optional): id -> cluster/label or (id, begin) -> cluster/label
        by (str, optional): 'id' or 'id_begin'. If None, auto-detect
        persper (bool): If True, return person-period data (multiple rows per spell).
                        If False, return spell-level data (one row per spell).
    
    Returns:
        pd.DataFrame: A dataset ready for event history analysis with:
            - All original SAMM variables (id, time, spell_time, etc.)
            - SAMMtypology: The typology classification (with "None" for non-events)
            - lastobs: Boolean indicating if this is the last observation of a spell
            - SAMM[type1], SAMM[type2], ...: Binary indicators for each typology category
              (these are your outcome variables for analysis)
    
    Example:
        >>> # Define typologies for transitions from unemployment
        >>> typology = ['reemployed', 'education', 'reemployed', 'retired', ...]
        >>> # Create EHA dataset
        >>> eha_data = seqsammeha(samm_obj, spell='unemployed', typology=typology, persper=True)
        >>> # Now you can use this with logistic regression, Cox models, etc.
        >>> # For example: predict probability of reemployment vs. other outcomes
    """
    
    # First, set the typology in the SAMM object using any of the supported inputs
    samm = set_typology(
        samm,
        spell=spell,
        typology=typology,
        clusters=clusters,
        cluster_to_name=cluster_to_name,
        mapping=mapping,
        by=by
    )
    
    # Filter data to only include rows in the specified spell
    spell_condition = (samm.data['s.1'] == spell)
    ppdata = samm.data[spell_condition].copy()
    
    # Identify the last observation for each spell
    # Group by individual ID and spell begin time, then mark the maximum spell_time
    ppdata['lastobs'] = ppdata.groupby(['id', 'begin'])['spell_time'].transform('max') == ppdata['spell_time']
    
    # Create binary indicator variables for each typology category
    # This creates dummy variables that statistical models can use
    # Determine unique typologies from the rows where typology is set
    typology_series = samm.data.loc[(samm.data['s.1'] == spell) & (samm.data['transition'] == True), 'typology']
    unique_types = typology_series.dropna().unique()
    
    # Create a column for each unique typology
    for type_label in unique_types:
        col_name = f'SAMM{type_label}'
        ppdata[col_name] = (ppdata['typology'] == type_label).astype(int)
    
    # Ensure 'SAMMtypology' column exists and is properly named
    ppdata = ppdata.rename(columns={'typology': 'SAMMtypology'})
    
    # If persper=False, return only the last observation of each spell
    if not persper:
        ppdata = ppdata[ppdata['lastobs']].copy()
    
    # Reset index for clean output
    ppdata = ppdata.reset_index(drop=True)
    
    return ppdata


# Define what gets imported with "from module import *"
__all__ = [
    'SAMM',
    'sequence_analysis_multi_state_model',
    'plot_samm',
    'seqsammseq',
    'set_typology',
    'seqsammeha',
    '_expand_typology_for_transitions',
    # Keep old names for backward compatibility
    'seqsamm'
]

# Backward compatibility aliases
seqsamm = sequence_analysis_multi_state_model
