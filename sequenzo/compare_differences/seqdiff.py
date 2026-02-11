"""
@Author  : Yuqi Liang 梁彧祺
@File    : seqdiff.py
@Time    : 2026-02-10 16:35
@Desc    : Position-wise discrepancy analysis between groups of sequences.
           
           This module implements the seqdiff function which analyzes how differences
           between groups of sequences evolve along the positions. It runs a sequence
           of discrepancy analyses on sliding windows.
           
           Corresponds to TraMineR function: seqdiff()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Any
import warnings
import gc

from ..tree_analysis.tree_utils import compute_distance_association
from ..dissimilarity_measures.get_distance_matrix import get_distance_matrix
from ..define_sequence_data import SequenceData


def compare_groups_across_positions(
    seqdata: Union[pd.DataFrame, SequenceData],
    group: Union[np.ndarray, pd.Series, pd.DataFrame],
    cmprange: tuple = (0, 1),
    seqdist_args: Optional[dict] = None,
    with_missing: bool = False,
    weighted: bool = True,
    squared: bool = False
) -> dict:
    """
    Position-wise discrepancy analysis between groups of sequences.
    
    The function analyzes how the part of discrepancy explained by the group variable
    evolves along the position axis. It runs successively discrepancy analyses within
    a sliding time-window of range cmprange. At each position t, the method uses
    get_distance_matrix to compute a distance matrix over the time-window
    (t + cmprange[0], t + cmprange[1]) and then derives the explained discrepancy
    on that window with compute_distance_association.
    
    **Corresponds to TraMineR function: `seqdiff()`**
    
    **TraMineR Equivalent:**
    ```r
    # In R (TraMineR package):
    seqdiff(seqdata, group, cmprange = c(0, 1), 
            seqdist.args = list(method = "LCS", norm = "auto"),
            with.missing = FALSE, weighted = TRUE, squared = FALSE)
    ```
    
    Parameters
    ----------
    seqdata : pd.DataFrame
        State sequence data. Each row is a sequence, each column is a time position.
        This should be a sequence object created with define_sequence_data().
        
    group : np.ndarray, pd.Series, or pd.DataFrame
        The grouping variable. Can be:
        - A 1D array/Series: Group labels for each sequence
        - A pd.DataFrame: Sequence data where each position defines groups
        
    cmprange : tuple, optional
        Vector of two integers: Time range of the sliding windows.
        Comparison at position t is computed on the window
        (t + cmprange[0], t + cmprange[1]).
        Default: (0, 1) - compares position t with position t+1
        
    seqdist_args : dict, optional
        Dictionary of arguments passed to get_distance_matrix() for computing distances.
        Common arguments include:
        - method: Distance method (default: "LCS")
        - norm: Normalization method (default: "auto")
        Default: {"method": "LCS", "norm": "auto"}
        
    with_missing : bool, optional
        If True, missing values are considered as an additional state.
        If False, subsequences with missing values are removed from the analysis.
        Default: False
        
    weighted : bool, optional
        If True, seqdiff uses the weights from seqdata.
        Default: True
        
    squared : bool, optional
        If True, the dissimilarities are squared for computing the discrepancy.
        Default: False
        
    Returns
    -------
    dict
        A dictionary with the following items:
        
        - 'stat': pd.DataFrame with five statistics for each time position:
            - 'Pseudo F': Pseudo F-statistic
            - 'Pseudo Fbf': Pseudo F with Bonferroni correction
            - 'Pseudo R2': Proportion of discrepancy explained by groups
            - 'Bartlett': Bartlett test statistic for homogeneity of variances
            - 'Levene': Levene test statistic for homogeneity of variances
            
        - 'discrepancy': pd.DataFrame with discrepancy values at each time position
            for each group and the total discrepancy
            
        - 'xtstep': Time step from seqdata (for plotting)
        - 'tick_last': Whether to tick last position (for plotting)
        
    Notes
    -----
    - The function performs discrepancy analysis at each position using a sliding window
    - At each position t, distances are computed over subsequences spanning
      (t + cmprange[0]) to (t + cmprange[2])
    - Missing values handling:
        - If with_missing=False: sequences with any missing values in the window
          are excluded from that position's analysis
        - If with_missing=True: missing values are treated as a valid state
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sequenzo.compare_differences import seqdiff
    >>> 
    >>> # Create sample sequence data
    >>> seqdata = pd.DataFrame({
    ...     'pos1': ['A', 'A', 'B', 'B'],
    ...     'pos2': ['A', 'B', 'B', 'C'],
    ...     'pos3': ['B', 'B', 'C', 'C']
    ... })
    >>> 
    >>> # Define groups
    >>> group = np.array([1, 1, 2, 2])
    >>> 
    >>> # Run position-wise analysis using centered sliding windows of length 5
    >>> result = seqdiff(seqdata, group=group, cmprange=(-2, 2))
    >>> 
    >>> # Print results
    >>> print(result['stat'])
    >>> print(result['discrepancy'])
    >>> 
    >>> # Plot results
    >>> plot_seqdiff(result, stat='Pseudo R2')
    
    References
    ----------
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2011).
    Discrepancy analysis of state sequences.
    Sociological Methods and Research, Vol. 40(3), 471-510.
    doi:10.1177/0049124111415372
    
    Studer, M., G. Ritschard, A. Gabadinho and N. S. Müller (2010).
    Discrepancy analysis of complex objects using dissimilarities.
    In F. Guillet, G. Ritschard, D. A. Zighed and H. Briand (Eds.),
    Advances in Knowledge Discovery and Management,
    Studies in Computational Intelligence, Volume 292, pp. 3-19. Berlin: Springer.
    """
    # Set default seqdist_args
    if seqdist_args is None:
        seqdist_args = {"method": "LCS", "norm": "auto"}
    else:
        # Make a copy to avoid modifying the input
        seqdist_args = seqdist_args.copy()
    
    # Handle SequenceData objects
    if isinstance(seqdata, SequenceData):
        # Extract DataFrame with time columns
        seqdata_df = seqdata.data[seqdata.time].copy()
    elif isinstance(seqdata, pd.DataFrame):
        seqdata_df = seqdata.copy()
    else:
        raise TypeError(
            "[!] 'seqdata' should be a DataFrame or SequenceData object. "
            f"Got {type(seqdata)}"
        )
    
    # Set with.missing in seqdist_args
    seqdist_args['with_missing'] = with_missing
    
    # Get sequence length
    slenE = seqdata_df.shape[1]
    
    # Determine range where we compare
    startAt = 1  # 1-based indexing to match R
    totrange = range(
        max(startAt, 1 - cmprange[0]),
        min(slenE, slenE - cmprange[1]) + 1
    )
    totrange_list = list(totrange)
    
    if len(totrange_list) == 0:
        raise ValueError(
            f"[!] Invalid cmprange {cmprange} for sequence length {slenE}. "
            "No valid positions to analyze."
        )
    
    # Determine group column names
    is_group_seqdata = isinstance(group, pd.DataFrame)
    if is_group_seqdata:
        # Group is a sequence object - use its alphabet as column names
        if hasattr(group, 'alphabet'):
            name_column = group.alphabet
        else:
            # Try to infer states
            unique_states = []
            for col in group.columns:
                unique_states.extend(group[col].unique())
            name_column = sorted(set(unique_states))
    else:
        # Group is a factor/array
        group = pd.Series(group) if not isinstance(group, pd.Series) else group
        group = pd.Categorical(group)
        name_column = group.categories.tolist()
    
    num_column = len(name_column)
    
    # Initialize result matrices
    num_positions = len(totrange_list)
    stat_matrix = np.full((num_positions, 5), np.nan)
    stat_df = pd.DataFrame(
        stat_matrix,
        columns=['Pseudo F', 'Pseudo Fbf', 'Pseudo R2', 'Bartlett', 'Levene'],
        index=[seqdata_df.columns[i - 1] for i in totrange_list]  # 0-based indexing for columns
    )
    
    discrepancy_matrix = np.zeros((num_positions, num_column + 1))
    discrepancy_df = pd.DataFrame(
        discrepancy_matrix,
        columns=name_column + ['Total'],
        index=[seqdata_df.columns[i - 1] for i in totrange_list]
    )
    
    # Get weights
    weights = None
    if weighted:
        if isinstance(seqdata, SequenceData) and hasattr(seqdata, 'weights'):
            weights = seqdata.weights
        elif hasattr(seqdata, 'weights'):
            weights = seqdata.weights
    
    # Process each position in the range
    offtot = min(totrange_list) - 1
    
    for idx, i in enumerate(totrange_list):
        # Run garbage collection
        gc.collect()
        
        # Define the subsequence range (1-based to 0-based conversion)
        srange = list(range(i + cmprange[0] - 1, i + cmprange[1]))
        
        # Ensure srange is valid
        srange = [s for s in srange if 0 <= s < slenE]
        
        if len(srange) == 0:
            continue
        
        # Get comparison base (group variable at this position)
        if is_group_seqdata:
            cmpbase = group.iloc[:, i - 1].copy()  # 0-based indexing
            # Handle missing values in group sequence
            if hasattr(group, 'void'):
                cmpbase[cmpbase == group.void] = np.nan
            if hasattr(group, 'nr'):
                cmpbase[cmpbase == group.nr] = np.nan
        else:
            cmpbase = group.copy()
        
        # Get subsequence for this range
        subseq = seqdata_df.iloc[:, srange]
        
        # Determine which sequences are complete cases
        if not with_missing:
            # Check for missing values in subsequence
            subseq2 = subseq.copy()
            # Mark missing values as NaN
            if isinstance(seqdata, SequenceData):
                if hasattr(seqdata, 'void'):
                    subseq2[subseq2 == seqdata.void] = np.nan
                if hasattr(seqdata, 'nr'):
                    subseq2[subseq2 == seqdata.nr] = np.nan
            elif hasattr(seqdata, 'void'):
                subseq2[subseq2 == seqdata.void] = np.nan
            elif hasattr(seqdata, 'nr'):
                subseq2[subseq2 == seqdata.nr] = np.nan
            
            # Get complete cases (no NaN in subsequence or group)
            seqok = (~subseq2.isna().any(axis=1)) & (~cmpbase.isna())
        else:
            # Only check for missing values in group variable
            seqok = ~cmpbase.isna()
        
        # Skip if no valid sequences
        if not seqok.any():
            warnings.warn(
                f"[!] No valid sequences at position {i}. Skipping.",
                UserWarning
            )
            continue
        
        # Filter subsequence and group
        subseq_filtered = subseq[seqok].copy()
        cmpbase_filtered = cmpbase[seqok].copy()
        weights_filtered = weights[seqok] if weights is not None else None
        
        # Create temporary SequenceData object for get_distance_matrix
        # Extract states from original seqdata if available
        temp_states = None
        if isinstance(seqdata, SequenceData):
            temp_states = seqdata.states
        else:
            # Infer states from data
            temp_states = sorted(subseq_filtered.stack().dropna().unique().tolist())
        
        # Create temporary SequenceData
        temp_seqdata = SequenceData(
            subseq_filtered,
            time=list(subseq_filtered.columns),
            states=temp_states,
            weights=weights_filtered
        )
        
        # Compute distance matrix on the filtered range
        seqdist_args_temp = seqdist_args.copy()
        seqdist_args_temp['seqdata'] = temp_seqdata
        
        try:
            # Suppress messages from get_distance_matrix
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sdist = get_distance_matrix(**seqdist_args_temp)
        except Exception as e:
            warnings.warn(
                f"[!] Failed to compute distance matrix at position {i}: {e}",
                UserWarning
            )
            continue
        
        # Compute association with group variable
        try:
            # Use R=0 for no permutation test (faster)
            # Convert Categorical to numpy array if needed
            if isinstance(cmpbase_filtered, pd.Categorical):
                group_values = np.asarray(cmpbase_filtered)
            else:
                group_values = np.asarray(cmpbase_filtered.values if hasattr(cmpbase_filtered, 'values') else cmpbase_filtered)
            
            result = compute_distance_association(
                distance_matrix=sdist,
                group=group_values,
                weights=weights_filtered,
                R=0,
                weight_permutation='diss',
                squared=squared
            )
            
            # Extract all 5 statistics from compute_distance_association result
            # TraMineR's dissassoc returns: Pseudo F, Pseudo Fbf, Pseudo R2, Bartlett, Levene
            # Now all statistics are computed and returned in result['stat']
            if 'stat' in result:
                for stat_name in result['stat'].index:
                    if stat_name in stat_df.columns:
                        stat_df.loc[stat_df.index[idx], stat_name] = result['stat'].loc[stat_name, 'Value']
            else:
                # Fallback to individual values (backward compatibility)
                stat_df.loc[stat_df.index[idx], 'Pseudo F'] = result.get('pseudo_f', np.nan)
                stat_df.loc[stat_df.index[idx], 'Pseudo Fbf'] = result.get('pseudo_fbf', np.nan)
                stat_df.loc[stat_df.index[idx], 'Pseudo R2'] = result.get('pseudo_r2', np.nan)
                stat_df.loc[stat_df.index[idx], 'Bartlett'] = result.get('bartlett', np.nan)
                stat_df.loc[stat_df.index[idx], 'Levene'] = result.get('levene', np.nan)
            
            # Extract discrepancies from groups DataFrame
            for group_name in result['groups'].index:
                if group_name in discrepancy_df.columns:
                    discrepancy_df.loc[discrepancy_df.index[idx], group_name] = \
                        result['groups'].loc[group_name, 'discrepancy']
        
        except Exception as e:
            warnings.warn(
                f"[!] Failed to compute association at position {i}: {e}",
                UserWarning
            )
            continue
    
    # Build result dictionary
    result_dict = {
        'stat': stat_df,
        'discrepancy': discrepancy_df,
        'xtstep': getattr(seqdata, 'xtstep', 1),
        'tick_last': getattr(seqdata, 'tick_last', False),
    }
    
    return result_dict


def print_group_differences_across_positions(result: dict) -> None:
    """
    Print position-wise group difference results.
    
    **Corresponds to TraMineR function: `print.seqdiff()`**
    
    **TraMineR Equivalent:**
    ```r
    # In R (TraMineR package):
    print(seqdiff_result)
    ```
    
    Parameters
    ----------
    result : dict
        Result dictionary from compare_groups_across_positions()
    """
    print("\nStatistics:")
    print(result['stat'])
    print("\nDiscrepancies:")
    print(result['discrepancy'])


def plot_group_differences_across_positions(
    result: dict,
    stat: Union[str, List[str]] = 'Pseudo R2',
    plot_type: str = 'l',
    ylab: Optional[str] = None,
    xlab: str = '',
    legend_pos: str = 'upper center',
    ylim: Optional[tuple] = None,
    xaxis: bool = True,
    col: Optional[Union[str, List[str]]] = None,
    xtstep: Optional[int] = None,
    tick_last: Optional[bool] = None,
    figsize: tuple = (10, 6),
    **kwargs
) -> plt.Figure:
    """
    Plot position-wise group difference results.
    
    **Corresponds to TraMineR function: `plot.seqdiff()`**
    
    **TraMineR Equivalent:**
    ```r
    # In R (TraMineR package):
    plot(seqdiff_result, stat = "Pseudo R2", type = "l")
    ```
    
    Parameters
    ----------
    result : dict
        Result dictionary from compare_groups_across_positions()
        
    stat : str or list of str, optional
        Statistic(s) to plot. Can be:
        - One of: 'Pseudo F', 'Pseudo Fbf', 'Pseudo R2', 'Bartlett', 'Levene'
        - 'Variance' or 'discrepancy' or 'Residuals': plot discrepancy per group
        - List of two statistics to plot on dual y-axes
        Default: 'Pseudo R2'
        
    plot_type : str, optional
        Plot type: 'l' for line, 'p' for points, etc.
        Default: 'l'
        
    ylab : str, optional
        Y-axis label. Default: uses stat name
        
    xlab : str, optional
        X-axis label. Default: ''
        
    legend_pos : str, optional
        Legend position. Default: 'top'
        
    ylim : tuple, optional
        Y-axis limits as (min, max). Default: None (auto)
        
    xaxis : bool, optional
        Whether to show x-axis. Default: True
        
    col : str or list of str, optional
        Color(s) for lines. Default: None (auto)
        
    xtstep : int, optional
        Step for x-axis ticks. Default: uses result['xtstep']
        
    tick_last : bool, optional
        Whether to tick the last position. Default: uses result['tick_last']
        
    figsize : tuple, optional
        Figure size as (width, height). Default: (10, 6)
        
    **kwargs
        Additional arguments passed to matplotlib plot()
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Set defaults
    if ylab is None:
        if isinstance(stat, list):
            ylab = ' / '.join(stat)
        else:
            ylab = stat
    
    if xtstep is None:
        xtstep = result.get('xtstep', 1)
    
    if tick_last is None:
        tick_last = result.get('tick_last', False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different stat types
    if isinstance(stat, str) and stat in ['Variance', 'discrepancy', 'Residuals', 'residuals']:
        # Plot discrepancy per group
        discrepancy_df = result['discrepancy']
        num_groups = discrepancy_df.shape[1]
        
        # Set colors
        if col is None:
            # Use default color palette
            if num_groups <= 8:
                # Use new matplotlib API (3.7+) - colormaps is available as module attribute
                try:
                    from matplotlib import colormaps
                    cmap = colormaps['Accent']
                except (AttributeError, KeyError, ImportError):
                    # Fallback for older matplotlib versions (< 3.7)
                    from matplotlib.cm import get_cmap
                    cmap = get_cmap('Accent')
                colors = [cmap(i / num_groups) for i in range(num_groups)]
            else:
                colors = plt.cm.Set3(np.linspace(0, 1, num_groups))
        else:
            colors = col if isinstance(col, list) else [col] * num_groups
        
        # Compute values to plot
        if stat in ['Residuals', 'residuals']:
            toplot = discrepancy_df.values * (1 - result['stat']['Pseudo R2'].values[:, np.newaxis])
        else:
            toplot = discrepancy_df.values
        
        # Set y-limits
        if ylim is None:
            ylim = (np.nanmin(toplot), np.nanmax(toplot))
        
        # Plot each group
        x_values = np.arange(len(discrepancy_df))
        
        # Plot total first (last column)
        ax.plot(x_values, toplot[:, -1], 
                color=colors[-1], 
                linestyle='-' if plot_type == 'l' else '',
                marker='o' if plot_type == 'p' else '',
                label=discrepancy_df.columns[-1],
                **kwargs)
        
        # Plot other groups
        for i in range(num_groups - 1):
            ax.plot(x_values, toplot[:, i], 
                    color=colors[i], 
                    linestyle='-' if plot_type == 'l' else '',
                    marker='o' if plot_type == 'p' else '',
                    label=discrepancy_df.columns[i],
                    **kwargs)
        
        ax.set_ylim(ylim)
        ax.legend(loc=legend_pos)
        
    elif isinstance(stat, str):
        # Plot a single statistic
        if stat not in result['stat'].columns:
            raise ValueError(
                f"[!] 'stat' argument should be one of "
                f"{', '.join(['discrepancy'] + result['stat'].columns.tolist())}"
            )
        
        if col is None:
            col = 'black'
        
        x_values = np.arange(len(result['stat']))
        y_values = result['stat'][stat].values
        
        ax.plot(x_values, y_values,
                color=col,
                linestyle='-' if plot_type == 'l' else '',
                marker='o' if plot_type == 'p' else '',
                **kwargs)
    
    elif isinstance(stat, list) and len(stat) == 2:
        # Plot two statistics on dual y-axes
        if not all(s in result['stat'].columns for s in stat):
            raise ValueError(
                f"[!] The two values of 'stat' should be one of "
                f"{', '.join(result['stat'].columns.tolist())}"
            )
        
        if col is None:
            col = ['red', 'blue']
        
        x_values = np.arange(len(result['stat']))
        
        # Plot first statistic
        ax.plot(x_values, result['stat'][stat[0]].values,
                color=col[0],
                linestyle='-' if plot_type == 'l' else '',
                marker='o' if plot_type == 'p' else '',
                label=stat[0],
                **kwargs)
        ax.set_ylabel(stat[0], color=col[0])
        ax.tick_params(axis='y', labelcolor=col[0])
        
        # Create second y-axis
        ax2 = ax.twinx()
        ax2.plot(x_values, result['stat'][stat[1]].values,
                 color=col[1],
                 linestyle='-' if plot_type == 'l' else '',
                 marker='o' if plot_type == 'p' else '',
                 label=stat[1],
                 **kwargs)
        ax2.set_ylabel(stat[1], color=col[1])
        ax2.tick_params(axis='y', labelcolor=col[1])
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc=legend_pos)
    else:
        raise ValueError("[!] Too many values for 'stat' argument (max 2)")
    
    # Set labels
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    
    # Set x-axis ticks
    if xaxis:
        seql = len(result['stat'])
        tpos = list(range(0, seql, xtstep))
        if tick_last and tpos[-1] < seql - 1:
            tpos.append(seql - 1)
        ax.set_xticks(tpos)
        ax.set_xticklabels([result['stat'].index[i] for i in tpos])
    else:
        ax.set_xticks([])
    
    plt.tight_layout()
    return fig
