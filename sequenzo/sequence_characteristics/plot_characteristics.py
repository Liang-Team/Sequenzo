"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_characteristics.py
@Time    : 2025/9/24 23:22
@Desc    : Plot longitudinal characteristics of sequences with elegant visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib.font_manager import FontProperties

# Import the correct functions from sequence characteristics modules
from .simple_characteristics import get_number_of_transitions
from .within_sequence_entropy import get_within_sequence_entropy
from .turbulence import get_turbulence
from .complexity_index import get_complexity_index

def plot_longitudinal_characteristics(seqdata,
                                      pick_ids=None,
                                      k=9,
                                      selection='first',
                                      order_by="complexity",
                                      figsize=(8, 6),
                                      fontsize=12,
                                      title=None,
                                      xlabel="Normalized Values",
                                      ylabel="Sequence ID",
                                      custom_colors=None,
                                      show_sequence_ids=False):
    """
    Create a horizontal bar chart showing four key characteristics for selected sequences.
    
    This function calculates and visualizes four important sequence characteristics:
    - Transitions: How many times sequences change from one state to another
    - Entropy: How diverse/varied the sequences are
    - Turbulence: How chaotic or unpredictable the sequences are  
    - Complexity: How complex the overall pattern is
    
    All values are normalized to 0-1 scale for easy comparison.
    
    Parameters
    ----------
    seqdata : SequenceData
        Your sequence data object containing the sequences to analyze.
        
    pick_ids : list, optional (default=None)
        Specific sequence IDs you want to plot. If provided, only these sequences
        will be shown. If None, the function will automatically select sequences
        based on the 'selection' and 'k' parameters.
        Example: [1, 5, 10, 23] to show sequences with IDs 1, 5, 10, and 23
        
    k : int, optional (default=9)
        Number of sequences to display when pick_ids is None. 
        Warning: Using more than 15 may make the plot hard to read.
        
    selection : str, optional (default='first')
        How to select sequences when pick_ids is None:
        - 'first': Show the k sequences with highest values for the order_by metric
        - 'last': Show the k sequences with lowest values for the order_by metric
        
    order_by : str, optional (default='complexity')
        Which metric to use for sorting sequences when pick_ids is None:
        - 'transitions': Sort by number of state changes
        - 'entropy': Sort by sequence diversity
        - 'turbulence': Sort by sequence unpredictability
        - 'complexity': Sort by overall sequence complexity
        
    figsize : tuple, optional (default=(8, 6))
        Size of the plot as (width, height) in inches.
        Example: (10, 8) for a larger plot, (6, 4) for a smaller one

    fontsize : int, optional (default=12)
        Base font size for labels, ticks, and legend. Title uses fontsize+2.
        
    title : str, optional (default=None)
        Title to display at the top of the plot. If None, no title is shown.
        Example: "Sequence Characteristics Comparison"
        
    xlabel : str, optional (default="Normalized Values")
        Label for the horizontal axis (x-axis).
        
    ylabel : str, optional (default="Sequence ID")
        Label for the vertical axis (y-axis).

    custom_colors : dict or list, optional (default=None)
        Colors used for the four bars. If dict, keys can include
        {'Transitions', 'Entropy', 'Turbulence', 'Complexity'} to override defaults.
        If list/tuple of length 4, it maps to the above order.

    show_sequence_ids : bool, optional (default=False)
        If True, y-axis shows actual sequence IDs (when available).
        If False, shows 1..N index positions.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated metrics for all plotted sequences.
        Columns: ['Transitions', 'Entropy', 'Turbulence', 'Complexity']
        Index: The sequence IDs that were plotted
        
    Warnings
    --------
    - If you try to plot more than 15 sequences, you'll get a warning about
      potential overplotting (too crowded to read clearly)
    - All metric values are automatically normalized to 0-1 scale
    
    Examples
    --------
    Basic usage - plot 9 most complex sequences:
    >>> metrics = plot_longitudinal_characteristics(my_seqdata)
    
    Plot specific sequences by ID:
    >>> metrics = plot_longitudinal_characteristics(my_seqdata, 
    ...                                           pick_ids=[1, 5, 10, 15])
    
    Plot 5 sequences with highest number of transitions:
    >>> metrics = plot_longitudinal_characteristics(my_seqdata, 
    ...                                           k=5, 
    ...                                           order_by='transitions')
    
    Customize the plot appearance:
    >>> metrics = plot_longitudinal_characteristics(my_seqdata,
    ...                                           k=6,
    ...                                           figsize=(12, 8),
    ...                                           title="My Sequence Analysis",
    ...                                           xlabel="Characteristic Scores",
    ...                                           ylabel="Person ID")
    
    Notes
    -----
    The four characteristics help you understand different aspects of your sequences:
    
    - **Transitions**: Higher values mean sequences change states frequently
    - **Entropy**: Higher values mean sequences have more diverse states
    - **Turbulence**: Higher values mean sequences are more unpredictable
    - **Complexity**: Higher values mean sequences have more complex patterns
    
    All values range from 0 to 1, making them easy to compare across different
    types of sequences and datasets.
    """
    # Calculate four metrics (all should be 0-1 normalized)
    df_t = get_number_of_transitions(seqdata=seqdata, norm=True).iloc[:, 0]   # Series
    df_e = get_within_sequence_entropy(seqdata=seqdata, norm=True)           # Series or single-column DataFrame
    if isinstance(df_e, pd.DataFrame): df_e = df_e.iloc[:, 0]

    df_tb = get_turbulence(seqdata=seqdata, norm=True, type=2)               # Normalized turbulence
    if isinstance(df_tb, pd.DataFrame): df_tb = df_tb.iloc[:, 0]

    df_c = get_complexity_index(seqdata=seqdata)                             # Already 0-1 normalized
    if isinstance(df_c, pd.DataFrame): df_c = df_c.iloc[:, 0]

    # Create metrics DataFrame with actual sequence IDs as index
    metrics = pd.DataFrame({
        "Transitions": df_t,
        "Entropy": df_e,
        "Turbulence": df_tb,
        "Complexity": df_c
    })
    
    # Set the index to actual sequence IDs if available
    if hasattr(seqdata, 'ids') and seqdata.ids is not None:
        metrics.index = seqdata.ids

    # Check for overplotting and issue warning if needed
    if pick_ids is not None:
        num_sequences = len(pick_ids)
        if num_sequences > 15:
            warnings.warn(f"Plotting {num_sequences} sequences may cause overplotting issues. "
                         f"Consider reducing to 15 or fewer sequences for better visualization.",
                         UserWarning)
    elif k > 15:
        warnings.warn(f"Plotting {k} sequences may cause overplotting issues. "
                     f"Consider reducing to 15 or fewer sequences for better visualization.",
                     UserWarning)

    # Select sequences to display
    if pick_ids is not None:
        # Custom ID selection
        metrics = metrics.loc[pick_ids]
    else:
        # Sort by specified metric and select first/last k sequences
        key = order_by.capitalize()
        if key not in metrics.columns:
            key = "Complexity"
        
        metrics_sorted = metrics.sort_values(key, ascending=False)
        
        if selection == 'first':
            metrics = metrics_sorted.head(k)
        elif selection == 'last':
            metrics = metrics_sorted.tail(k)
        else:
            # Default to first k sequences
            metrics = metrics_sorted.head(k)

    # Create horizontal grouped bar chart
    # Use the DataFrame index which now contains the actual sequence IDs
    if show_sequence_ids:
        labels = list(metrics.index)
    else:
        labels = list(range(1, len(metrics) + 1))
    y = np.arange(len(metrics))
    bar_h = 0.18

    # Basic matplotlib styling
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Add simple background grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Axis/text color theme
    axis_gray = '#666666'

    # Add title only if provided
    if title is not None:
        plt.title(title, fontsize=fontsize + 2, color=axis_gray)

    # Color palette with optional overrides
    default_colors = {
        'Transitions': '#74C9B4',  # Soft green
        'Entropy': '#A6E3D0',      # Light green  
        'Turbulence': '#F9E79F',   # Light yellow
        'Complexity': '#F6CDA3'    # Light orange
    }

    if isinstance(custom_colors, dict):
        colors = {**default_colors, **custom_colors}
    elif isinstance(custom_colors, (list, tuple)) and len(custom_colors) == 4:
        ordered_keys = ['Transitions', 'Entropy', 'Turbulence', 'Complexity']
        colors = {k: v for k, v in zip(ordered_keys, custom_colors)}
    else:
        colors = default_colors

    plt.barh(y + 0.30, metrics["Transitions"].values, height=bar_h, 
             label="Transitions", color=colors['Transitions'])
    plt.barh(y + 0.10, metrics["Entropy"].values, height=bar_h, 
             label="Entropy", color=colors['Entropy'])
    plt.barh(y - 0.10, metrics["Turbulence"].values, height=bar_h, 
             label="Turbulence", color=colors['Turbulence'])
    plt.barh(y - 0.30, metrics["Complexity"].values, height=bar_h, 
             label="Complexity", color=colors['Complexity'])

    # Use actual sequence IDs as y-tick labels
    plt.yticks(y, labels)
    plt.xlim(0, 1)
    
    # Use custom labels with refined spacing
    ax.set_xlabel(xlabel, labelpad=8, fontsize=fontsize, color=axis_gray)
    # Slightly expand y-axis label letter spacing
    ylabel_props = FontProperties(stretch='expanded')
    ax.set_ylabel(ylabel, labelpad=6, fontproperties=ylabel_props, fontsize=fontsize, color=axis_gray)
    
    # Simple legend
    plt.legend(loc="lower right", fontsize=max(6, fontsize - 1))
    
    # Style axes like index plot - clean and minimal
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color(axis_gray)
    ax.spines['bottom'].set_color(axis_gray)
    
    # Move spines slightly away from the plot area (but keep y-axis closer than before)
    ax.spines['left'].set_position(('outward', 2))
    ax.spines['bottom'].set_position(('outward', 4))
    
    # Ticks styling and subtle padding
    ax.tick_params(axis='x', which='major', colors=axis_gray, length=4, width=0.7, direction='out', pad=4, labelsize=max(6, fontsize - 1))
    ax.tick_params(axis='y', which='major', colors=axis_gray, length=4, width=0.7, direction='out', pad=3, labelsize=max(6, fontsize - 1))
    
    # Extend axes slightly beyond the data range for better spacing
    ax.set_ylim(-0.5, len(metrics) - 0.5)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()

    return metrics  # Return the data used for plotting for inspection
