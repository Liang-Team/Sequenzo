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
from .overall_cross_sectional_entropy import get_cross_sectional_entropy

# Import visualization utilities
try:
    from ..visualization.utils.utils import set_up_time_labels_for_x_axis
except ImportError:
    # Fallback function if import fails
    def set_up_time_labels_for_x_axis(seqdata, ax, color="gray"):
        time_labels = np.array(seqdata.cleaned_time)
        num_time_steps = len(time_labels)
        
        if num_time_steps <= 10:
            xtick_positions = np.arange(num_time_steps)
        elif num_time_steps <= 20:
            xtick_positions = np.arange(0, num_time_steps, step=2)
        else:
            xtick_positions = np.linspace(0, num_time_steps - 1, num=10, dtype=int)
        
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(time_labels[xtick_positions], fontsize=10, rotation=0, ha="center", color=color)

def plot_longitudinal_characteristics(seqdata,
                                      pick_ids=None,
                                      k=9,
                                      selection='first',
                                      order_by="complexity",
                                      figsize=(8, 6),
                                      fontsize=12,
                                      title=None,
                                      show_title=True,
                                      xlabel="Normalized Values",
                                      ylabel="Sequence ID",
                                      save_as=None,
                                      dpi=200,
                                      custom_colors=None,
                                      show_sequence_ids=False,
                                      id_as_column=True):
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
        
    show_title : bool, optional (default=True)
        Whether to display the title. If False, no title will be shown regardless
        of the title parameter value. This provides consistent control with other plots.
        
    xlabel : str, optional (default="Normalized Values")
        Label for the horizontal axis (x-axis).
        
    ylabel : str, optional (default="Sequence ID")
        Label for the vertical axis (y-axis).

    save_as : str, optional (default=None)
        File path to save the plot. If None, plot will only be displayed.
        Supported formats: .png, .jpg, .jpeg, .pdf, .svg
        If no extension provided, .png will be added automatically.
        
    dpi : int, optional (default=200)
        Resolution (dots per inch) for saved image. Higher values result in
        better quality but larger file sizes.

    custom_colors : dict or list, optional (default=None)
        Colors used for the four bars. If dict, keys can include
        {'Transitions', 'Entropy', 'Turbulence', 'Complexity'} to override defaults.
        If list/tuple of length 4, it maps to the above order.

    show_sequence_ids : bool, optional (default=False)
        If True, y-axis shows actual sequence IDs (when available).
        If False, shows 1..N index positions.
        
    id_as_column : bool, optional (default=True)
        If True, the returned DataFrame will include ID as a separate column on the same level as other columns.
        If False, IDs will be used as the DataFrame index.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated metrics for all plotted sequences.
        If id_as_column=True: Columns: ['ID', 'Transitions', 'Entropy', 'Turbulence', 'Complexity'] (all columns at same level)
        If id_as_column=False: Columns: ['Transitions', 'Entropy', 'Turbulence', 'Complexity'], Index: The sequence IDs
        
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
    
    Plot without title:
    >>> metrics = plot_longitudinal_characteristics(my_seqdata, show_title=False)
    
    Save plot to file:
    >>> metrics = plot_longitudinal_characteristics(my_seqdata,
    ...                                           save_as="sequence_characteristics.png",
    ...                                           dpi=300)
    
    Save as PDF:
    >>> metrics = plot_longitudinal_characteristics(my_seqdata,
    ...                                           save_as="characteristics_analysis.pdf")
    
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
    df_t = get_number_of_transitions(seqdata=seqdata, norm=True).iloc[:, 1]   # Series
    df_e = get_within_sequence_entropy(seqdata=seqdata, norm=True)           # Series or single-column DataFrame
    if isinstance(df_e, pd.DataFrame): df_e = df_e.iloc[:, 1]

    df_tb = get_turbulence(seqdata=seqdata, norm=True, type=2, id_as_column=True)               # Normalized turbulence
    if isinstance(df_tb, pd.DataFrame): df_tb = df_tb.iloc[:, 1]

    df_c = get_complexity_index(seqdata=seqdata)                             # Already 0-1 normalized
    if isinstance(df_c, pd.DataFrame): df_c = df_c.iloc[:, 1]

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

    # Add title only if provided and show_title is True
    if show_title and title is not None:
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
    
    # Handle saving and display
    if save_as:
        if not any(save_as.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
            save_as += '.png'
        plt.savefig(save_as, dpi=dpi, bbox_inches='tight')
    
    plt.show()
    plt.close()

    # Handle ID display options for returned DataFrame
    if id_as_column:
        # Add ID as a separate column and reset index to numeric
        metrics_result = metrics.copy()
        metrics_result['ID'] = metrics_result.index
        metrics_result = metrics_result[['ID', 'Transitions', 'Entropy', 'Turbulence', 'Complexity']].reset_index(drop=True)
        return metrics_result
    else:
        # Return with ID as index (traditional format)
        metrics.index.name = 'ID'
        return metrics


def plot_cross_sectional_characteristics(seqdata,
                                          figsize=(10, 6),
                                          fontsize=12,
                                          title="Cross-sectional entropy over time",
                                          show_title=True,
                                          xlabel="Time",
                                          ylabel="Entropy (0-1)",
                                          line_color="#74C9B4",
                                          save_as=None,
                                          dpi=200,
                                          return_data=False,
                                          custom_state_colors=None):
    """
    Visualize cross-sectional entropy across time points.
    
    This function shows how diverse the population is at each time point,
    providing a complementary view to longitudinal analysis which tracks
    individual sequences over time.
    
    The plot displays cross-sectional entropy across time points.
    
    Parameters
    ----------
    seqdata : SequenceData
        Your sequence data object containing the sequences to analyze.
        
    figsize : tuple, optional (default=(10, 6))
        Size of the plot as (width, height) in inches.
        
    fontsize : int, optional (default=12)
        Base font size for labels, ticks, and axes. Title uses fontsize+1.
        
    title : str, optional (default="Cross-sectional entropy over time")
        Title for the entropy plot. If show_title=False, this is ignored.
        
    show_title : bool, optional (default=True)
        Whether to display the title. If False, no title will be shown regardless
        of the title parameter value.
        
    xlabel : str, optional (default="Time")
        Label for the x-axis.
        
    ylabel : str, optional (default="Entropy (0-1)")
        Label for the y-axis (main entropy axis).
        
    line_color : str, optional (default="#74C9B4")
        Color for the entropy line. Can be any valid matplotlib color including
        hex colors like "#FF5733", named colors like "red", or RGB tuples.
        
    save_as : str, optional (default=None)
        File path to save the plot. If None, plot will only be displayed.
        Supported formats: .png, .jpg, .jpeg, .pdf, .svg
        If no extension provided, .png will be added automatically.
        
    dpi : int, optional (default=200)
        Resolution (dots per inch) for saved image. Higher values result in
        better quality but larger file sizes.
        
    return_data : bool, optional (default=False)
        Whether to return the computed data. If False, only displays the plot.
        If True, returns a dictionary with frequencies, entropy, and valid states.
        
    custom_state_colors : dict, optional (default=None)
        Custom color mapping for states. Keys should match your state labels.
        If None, uses the colors defined in your SequenceData object.
        Example: {"Education": "#A7D8DE", "Employment": "#F6CDA3"}
        Note: This parameter is maintained for compatibility but not used in entropy plot.
        
    Returns
    -------
    dict or None
        If return_data=True, returns a dictionary containing the computed data:
        - "Frequencies": DataFrame with states as rows and time points as columns
        - "Entropy": Series with entropy values for each time point (0-1 normalized)
        - "ValidStates": Series with number of valid observations per time point
        
        If return_data=False (default), returns None to keep output clean and focus on visualization.
        
    Notes
    -----
    **Cross-sectional entropy** measures how diverse the population is at each
    time point. Values range from 0 to 1:
    - 0: Everyone is in the same state (no diversity)
    - 1: Population is equally distributed across all possible states (maximum diversity)
    
    The plot uses index plot styling with clean borders. For state distribution 
    visualization, use the dedicated `plot_state_distribution` function.
    
    Examples
    --------
    Basic usage (displays plot only, no data returned):
    >>> plot_cross_sectional_characteristics(my_seqdata)
    
    Custom title and size:
    >>> plot_cross_sectional_characteristics(my_seqdata,
    ...                                     figsize=(12, 6),
    ...                                     title="Population Diversity Over Time")
    
    Plot without title:
    >>> plot_cross_sectional_characteristics(my_seqdata, show_title=False)
    
    Custom labels and colors:
    >>> plot_cross_sectional_characteristics(my_seqdata,
    ...                                     xlabel="Years",
    ...                                     ylabel="Indicators",
    ...                                     line_color="#FF5733")
    
    Custom hex color:
    >>> plot_cross_sectional_characteristics(my_seqdata, line_color="#2E86AB")
    
    Save plot to file:
    >>> plot_cross_sectional_characteristics(my_seqdata, 
    ...                                     save_as="entropy_plot.png",
    ...                                     dpi=300)
    
    Save with custom format:
    >>> plot_cross_sectional_characteristics(my_seqdata,
    ...                                     save_as="entropy_analysis.pdf")
    
    Get data when needed (only when explicitly requested):
    >>> result = plot_cross_sectional_characteristics(my_seqdata, return_data=True)
    >>> entropy_values = result['Entropy']      # Access entropy data
    >>> frequencies = result['Frequencies']     # State frequencies by time
    >>> valid_n = result['ValidStates']         # Sample sizes by time
    """
    # Get cross-sectional data using the existing function
    res = get_cross_sectional_entropy(seqdata, weighted=True, norm=True, return_format="dict")
    
    freq = res["Frequencies"]          # rows: states, cols: time points
    # Get normalized or raw entropy (check which key exists)
    if "per_time_entropy_norm" in res and res["per_time_entropy_norm"] is not None:
        ent = res["per_time_entropy_norm"]
    else:
        ent = res["Entropy"]
    N = res.get("ValidStates", None)   # valid sample sizes per time point

    # Sort time axis if possible (handles both numeric and string time labels)
    try:
        # Try to sort columns as integers first
        sorted_cols = sorted(freq.columns, key=lambda x: int(x))
        freq = freq[sorted_cols]
        ent = ent.loc[sorted_cols]
        if N is not None:
            N = N.loc[sorted_cols]
    except (ValueError, TypeError):
        try:
            # If that fails, sort as strings
            sorted_cols = sorted(freq.columns)
            freq = freq[sorted_cols]
            ent = ent.loc[sorted_cols]
            if N is not None:
                N = N.loc[sorted_cols]
        except Exception:
            # If all sorting fails, keep original order
            pass

    # Prepare color scheme - use SequenceData's color mapping
    if custom_state_colors is not None:
        # Use custom colors if provided
        colors = [custom_state_colors.get(s, None) for s in freq.index]
        colors = [c for c in colors if c is not None] or None
    else:
        # Use SequenceData's built-in color mapping (this is the standard way)
        colors = None
        if hasattr(seqdata, 'color_map') and seqdata.color_map:
            # Map state labels to colors using the sequence data's color mapping
            colors = []
            for state_label in freq.index:
                # Find the state index for this label
                if hasattr(seqdata, 'state_mapping') and seqdata.state_mapping:
                    state_idx = seqdata.state_mapping.get(state_label)
                    if state_idx is not None and state_idx in seqdata.color_map:
                        colors.append(seqdata.color_map[state_idx])
                    else:
                        colors.append(None)
                else:
                    # Fallback: try direct label lookup
                    colors.append(seqdata.color_map.get(state_label, None))
            
            # Filter out None values
            colors = [c for c in colors if c is not None] or None

    # Color scheme consistent with existing plot style
    axis_gray = '#666666'
    
    # Create entropy plot with optional valid N secondary axis
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    
    ax1.plot(ent.index, ent.values, marker='o', color=line_color, linewidth=2, markersize=4)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(ylabel, fontsize=fontsize, color=axis_gray)
    
    # Set title only if show_title is True
    if show_title and title:
        ax1.set_title(title, fontsize=fontsize+1, color=axis_gray)
    
    
    # Set up x-axis labels using the utility function
    set_up_time_labels_for_x_axis(seqdata, ax1, color=axis_gray)
    
    # Style consistent with index plot design
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    # Use index plot style borders - only show left and bottom spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('gray')
    ax1.spines['bottom'].set_color('gray')
    ax1.spines['left'].set_linewidth(0.7)
    ax1.spines['bottom'].set_linewidth(0.7)
    
    # Move spines slightly away from the plot area for better aesthetics
    ax1.spines['left'].set_position(('outward', 5))
    ax1.spines['bottom'].set_position(('outward', 5))
    
    ax1.tick_params(axis='both', colors=axis_gray, labelsize=max(6, fontsize-1), length=4, width=0.7)
    
    # Add x-axis label
    ax1.set_xlabel(xlabel, fontsize=fontsize, color=axis_gray)

    plt.tight_layout()
    
    # Handle saving and display
    if save_as:
        if not any(save_as.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
            save_as += '.png'
        plt.savefig(save_as, dpi=dpi, bbox_inches='tight')
    
    plt.show()
    plt.close()

    # Only return data if explicitly requested
    if return_data:
        return {"Frequencies": freq, "Entropy": ent, "ValidStates": N}
    else:
        return None
