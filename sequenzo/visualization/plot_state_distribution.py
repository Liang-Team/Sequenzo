"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_state_distribution.py
@Time    : 15/02/2025 22:03
@Desc    : 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sequenzo import SequenceData
from sequenzo.visualization.utils import (
    set_up_time_labels_for_x_axis,
    save_figure_to_buffer,
    create_standalone_legend,
    combine_plot_with_legend,
    save_and_show_results,
    determine_layout,
    show_plot_title,
    show_group_title
)


def smart_sort_groups(groups):
    """
    Smart sorting: prioritize numeric prefix, fallback to string sorting
    
    :param groups: List of group names
    :return: Sorted list of group names
    """
    import re
    
    # Compile regex once for better performance
    numeric_pattern = re.compile(r'^(\d+)')
    
    def sort_key(item):
        match = numeric_pattern.match(str(item))
        return (int(match.group(1)), str(item)) if match else (float('inf'), str(item))
    
    return sorted(groups, key=sort_key)


def plot_state_distribution(seqdata: SequenceData,
                            # Grouping parameters
                            group_by_column=None,
                            group_dataframe=None,
                            group_column_name=None,
                            group_labels=None,
                            # Other parameters
                            weights="auto",
                            figsize=(12, 7),
                            plot_style="standard",
                            title=None,
                            xlabel="Time",
                            ylabel="State Distribution (%)",
                            save_as=None,
                            dpi=200,
                            layout='column',
                            nrows: int = None,
                            ncols: int = None,
                            stacked=True,
                            show=True,
                            include_legend=True,
                            group_order=None,
                            fontsize=12,
                            sort_groups='auto',
                            show_group_titles: bool = True,
                            show_title: bool = True) -> None:
    """
    Creates state distribution plots for different groups, showing how state
    prevalence changes over time within each group.

    **Two API modes for grouping:**
    
    1. **Simplified API** (when grouping info is already in the data):
       ```python
       plot_state_distribution(seqdata, group_by_column="Cluster", group_labels=cluster_labels)
       ```
    
    2. **Complete API** (when grouping info is in a separate dataframe):
       ```python
       plot_state_distribution(seqdata, group_dataframe=membership_df, 
                              group_column_name="Cluster", group_labels=cluster_labels)
       ```

    :param seqdata: (SequenceData) A SequenceData object containing sequences
    
    **Grouping parameters:**
    :param group_by_column: (str, optional) Column name from seqdata.data to group by.
                           Use this when grouping information is already in your data.
                           Example: "Cluster", "sex", "education"
    :param group_dataframe: (pd.DataFrame, optional) Separate dataframe containing grouping information.
                           Use this when grouping info is in a separate table (e.g., clustering results).
                           Must contain ID column and grouping column.
    :param group_column_name: (str, optional) Name of the grouping column in group_dataframe.
                             Required when using group_dataframe.
    :param group_labels: (dict, optional) Custom labels for group values.
                        Example: {1: "Late Family Formation", 2: "Early Partnership"}
                        Maps original values to display labels.
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param figsize: (tuple) Size of the figure (only used when plot_style="custom")
    :param plot_style: Plot aspect style:
        - 'standard': Standard proportions (12, 7) - balanced view
        - 'compact': Compact/vertical proportions (10, 8) - more vertical like R plots
        - 'wide': Wide proportions (14, 5) - emphasizes time progression  
        - 'narrow': Narrow/tall proportions (9, 11) - moderately vertical
        - 'custom': Use the provided figsize parameter
    :param title: (str) Optional title for the plot
    :param xlabel: (str) Label for the x-axis
    :param ylabel: (str) Label for the y-axis
    :param save_as: (str) Optional file path to save the plot
    :param dpi: (int) Resolution of the saved plot
    :param layout: (str) Layout style - 'column' (default, 3xn), 'grid' (nxn)
    :param stacked: (bool) Whether to create stacked area plots (True) or line plots (False)
    :param group_order: List, manually specify group order (overrides sort_groups)
    :param sort_groups: String, sorting method: 'auto'(smart numeric), 'numeric'(numeric prefix), 'alpha'(alphabetical), 'none'(original order)
    :param show_title: (bool, default: True) If False, suppresses the main title display even if title parameter is provided.
                      This allows you to control title visibility separately from providing a title string.

    :return: None
    """
    # Determine figure size based on plot style
    style_sizes = {
        'standard': (12, 7),   # Balanced view
        'compact': (10, 8),    # More square, like R plots  
        'wide': (14, 5),       # Wide, emphasizes time
        'narrow': (9, 11),     # Moderately vertical
        'custom': figsize      # User-provided
    }
    
    if plot_style not in style_sizes:
        raise ValueError(f"Invalid plot_style '{plot_style}'. "
                        f"Supported styles: {list(style_sizes.keys())}")
    
    # Special validation for custom plot style
    if plot_style == 'custom' and figsize == (12, 7):
        raise ValueError(
            "When using plot_style='custom', you must explicitly provide a figsize parameter "
            "that differs from the default (12, 7). "
            "Suggested custom sizes:\n"
            "  - For wide plots: figsize=(16, 6)\n"
            "  - For tall plots: figsize=(8, 12)\n"
            "  - For square plots: figsize=(10, 10)\n"
            "  - For small plots: figsize=(8, 5)\n"
            "Example: plot_state_distribution(data, plot_style='custom', figsize=(14, 9))"
        )
    
    actual_figsize = style_sizes[plot_style]
    
    # Handle the simplified API: group_by_column
    if group_by_column is not None:
        # Validate that the column exists in the original data
        if group_by_column not in seqdata.data.columns:
            available_cols = [col for col in seqdata.data.columns if col not in seqdata.time and col != seqdata.id_col]
            raise ValueError(
                f"Column '{group_by_column}' not found in the data. "
                f"Available columns for grouping: {available_cols}"
            )
        
        # Automatically create group_dataframe and group_column_name from the simplified API
        group_dataframe = seqdata.data[[seqdata.id_col, group_by_column]].copy()
        group_dataframe.columns = ['Entity ID', 'Category']
        group_column_name = 'Category'
        
        # Handle group labels - flexible and user-controllable
        unique_values = seqdata.data[group_by_column].unique()
        
        if group_labels is not None:
            # User provided custom labels - use them
            missing_keys = set(unique_values) - set(group_labels.keys())
            if missing_keys:
                raise ValueError(
                    f"group_labels missing mappings for values: {missing_keys}. "
                    f"Please provide labels for all unique values in '{group_by_column}': {sorted(unique_values)}"
                )
            group_dataframe['Category'] = group_dataframe['Category'].map(group_labels)
        else:
            # No custom labels provided - use smart defaults
            if all(isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) for v in unique_values):
                # Numeric values - keep as is (user can provide group_labels if they want custom names)
                pass
            # For string/categorical values, keep original values
            # This handles cases where users already have meaningful labels like "Male"/"Female"
        
        print(f"[>] Creating grouped plots by '{group_by_column}' with {len(unique_values)} categories")
    
    # If no grouping information, create a single plot
    if group_dataframe is None or group_column_name is None:
        return _plot_state_distribution_single(
            seqdata=seqdata, weights=weights, figsize=actual_figsize,
            plot_style=plot_style, title=title, xlabel=xlabel, ylabel=ylabel,
            save_as=save_as, dpi=dpi, stacked=stacked,
            show=show, include_legend=include_legend, fontsize=fontsize,
            show_title=show_title
        )

    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    # Ensure ID columns match (convert if needed)
    id_col_name = "Entity ID" if "Entity ID" in group_dataframe.columns else group_dataframe.columns[0]

    # Apply group_labels if provided (for group_dataframe API)
    if group_labels is not None and group_column_name in group_dataframe.columns:
        # Validate that all values in the group column have labels
        unique_values = group_dataframe[group_column_name].unique()
        missing_keys = set(unique_values) - set(group_labels.keys())
        if missing_keys:
            raise ValueError(
                f"group_labels missing mappings for values: {missing_keys}. "
                f"Please provide labels for all unique values in '{group_column_name}': {sorted(unique_values)}"
            )
        # Apply the labels mapping
        group_dataframe = group_dataframe.copy()  # Avoid modifying original
        group_dataframe[group_column_name] = group_dataframe[group_column_name].map(group_labels)

    # Get unique groups and sort them based on user preference
    if group_order:
        # Use manually specified order, filter out non-existing groups
        groups = [g for g in group_order if g in group_dataframe[group_column_name].unique()]
        missing_groups = [g for g in group_dataframe[group_column_name].unique() if g not in group_order]
        if missing_groups:
            print(f"[Warning] Groups not in group_order will be excluded: {missing_groups}")
    elif group_labels is not None:
        # If group_labels is provided, use its key order to determine groups order
        # This ensures subplot order matches the order in group_labels dictionary
        # Note: group_labels keys are original values, values are labels (which become groups)
        mapped_labels = []
        available_labels = set(group_dataframe[group_column_name].unique())
        
        # Iterate through group_labels in order (Python 3.7+ dicts maintain insertion order)
        for original_key, label_value in group_labels.items():
            # Check if this label exists in the mapped dataframe
            if label_value in available_labels:
                mapped_labels.append(label_value)
        
        # Also check for any labels in dataframe that weren't in group_labels
        missing_in_labels = available_labels - set(mapped_labels)
        if missing_in_labels:
            print(f"[Warning] Some groups in data are not in group_labels and will be excluded: {missing_in_labels}")
        
        groups = mapped_labels
    elif sort_groups == 'numeric' or sort_groups == 'auto':
        groups = smart_sort_groups(group_dataframe[group_column_name].unique())
    elif sort_groups == 'alpha':
        groups = sorted(group_dataframe[group_column_name].unique())
    elif sort_groups == 'none':
        groups = list(group_dataframe[group_column_name].unique())
    else:
        raise ValueError(f"Invalid sort_groups value: {sort_groups}. Use 'auto', 'numeric', 'alpha', or 'none'.")
    
    num_groups = len(groups)

    # Calculate figure size and layout based on number of groups and specified layout
    nrows, ncols = determine_layout(num_groups, layout=layout, nrows=nrows, ncols=ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(actual_figsize[0] * ncols, actual_figsize[1] * nrows),
        gridspec_kw={'wspace': 0.15, 'hspace': 0.25}  # Reduced spacing for tighter layout
    )
    axes = axes.flatten()

    # Create state mapping from numerical values back to state names
    inv_state_mapping = {v: k for k, v in seqdata.state_mapping.items()}

    # Process each group
    for i, group in enumerate(groups):
        # Get IDs for this group
        group_ids = group_dataframe[group_dataframe[group_column_name] == group][id_col_name].values

        # Match IDs with sequence data
        mask = np.isin(seqdata.ids, group_ids)
        if not np.any(mask):
            print(f"Warning: No matching sequences found for group '{group}'")
            continue

        # Get sequences for this group
        group_seq_df = seqdata.to_dataframe().loc[mask]
        
        # Get weights for this group
        if weights is None:
            w = np.ones(len(group_seq_df))
        else:
            w = np.asarray(weights)[mask]

        # Broadcast weights to each time point
        W = np.repeat(w[:, None], group_seq_df.shape[1], axis=1)

        # Calculate weighted state distributions at each time point
        distributions = []
        for t, col in enumerate(group_seq_df.columns):
            col_vals = group_seq_df[col].to_numpy()
            
            # Calculate weighted counts for each state
            sums = {s: float(W[col_vals == s, t].sum()) for s in range(1, len(seqdata.states)+1)}
            totw = float(W[:, t].sum())
            
            # Convert to weighted percentages
            dist = {inv_state_mapping.get(s, 'Missing'): 100.0 * (sums[s] / totw if totw > 0 else 0.0) 
                    for s in range(1, len(seqdata.states) + 1)}

            # Add time point and distribution to the list
            distributions.append(dict({"time": col, **{str(k): v for k, v in dist.items()}}))

        # Ensure percentages sum to exactly 100% to avoid gaps
        for j in range(len(distributions)):
            total_percentage = sum(distributions[j][str(state)] for state in seqdata.states)
            if total_percentage < 100:
                top_state = str(seqdata.states[-1])
                distributions[j][str(top_state)] += (100 - total_percentage)

        # Convert to DataFrame for plotting
        dist_df = pd.DataFrame(distributions)

        # Plot on the corresponding axis
        ax = axes[i]

        # Get colors for each state
        # seqdata.states are integer encodings (e.g., 1, 2, ...)
        # seqdata.state_mapping[state] maps integers to labels (e.g., 'Married', 'Single')
        # seqdata.color_map[...] gets color by label
        base_colors = [seqdata.color_map[seqdata.state_mapping[state]] for state in seqdata.states]

        # Plot the data
        if stacked:
            # Create a stacked area plot
            ax.stackplot(range(len(dist_df)),
                         [dist_df[str(state)] for state in seqdata.states],
                         labels=seqdata.labels,
                         colors=base_colors,
                         alpha=1.0)

            # Add grid lines behind the stack plot
            ax.grid(axis='y', linestyle='-', alpha=0.2)
            ax.set_axisbelow(True)
        else:
            # Create a line plot
            for state, label, color in zip(seqdata.states, seqdata.labels, base_colors):
                ax.plot(range(len(dist_df)), dist_df[str(state)],
                        label=label, color=color,
                        linewidth=2.5, marker='o', markersize=5)

            # Add grid lines
            ax.grid(True, linestyle='-', alpha=0.2)

        # Set group title with weighted sample size
        # Check if we have effective weights (not all 1.0) and they were provided by user
        original_weights = getattr(seqdata, "weights", None)
        if original_weights is not None and not np.allclose(original_weights, 1.0):
            sum_w = float(w.sum())
            group_title = f"{group} (n = {len(group_seq_df)}, total weight = {sum_w:.1f})"
        else:
            group_title = f"{group} (n = {len(group_seq_df)})"
        if show_group_titles:
            show_group_title(ax, group_title, show=True, fontsize=fontsize)

        # Set y-axis limits from 0 to 100%
        ax.set_ylim(0, 100)

        # Clean up axis aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)
        
        # Move spines slightly away from the plot area for better aesthetics (same as index plot)
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        
        ax.tick_params(axis='x', colors='gray', length=4, width=0.7)
        ax.tick_params(axis='y', colors='gray', length=4, width=0.7)

        # Set x-axis labels
        set_up_time_labels_for_x_axis(seqdata, ax)
        
        # Set x-axis range to prevent over-extension like in the reference image
        ax.set_xlim(-0.5, len(seqdata.cleaned_time) - 0.5)

        # Add axis labels
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10, color='black')

        # if i >= num_groups - ncols:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color='black')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add a common title if provided and show_title is True
    if title and show_title:
        fig.suptitle(title, fontsize=fontsize+2, y=1.02)

    # Adjust layout to remove tight_layout warning and eliminate extra right space
    fig.subplots_adjust(wspace=0.15, hspace=0.25, bottom=0.1, top=0.9, right=0.98, left=0.08)

    # Save main figure to memory
    main_buffer = save_figure_to_buffer(fig, dpi=dpi)

    if include_legend:
        # Create standalone legend
        colors = seqdata.color_map_by_label
        legend_buffer = create_standalone_legend(
            colors=colors,
            labels=seqdata.labels,
            ncol=min(5, len(seqdata.states)),
            figsize=(actual_figsize[0] * ncols, 1),
            fontsize=fontsize-2,
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
        plt.figure(figsize=(actual_figsize[0] * ncols, actual_figsize[1] * nrows + 1))
        plt.imshow(combined_img)
        plt.axis('off')
        if show or save_as:  # Show if displaying or saving is needed
            plt.show()
        plt.close()
    else:
        # Display plot without legend
        if save_as and not save_as.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            save_as = save_as + '.png'
        
        # Save or show the main plot directly
        plt.figure(figsize=(actual_figsize[0] * ncols, actual_figsize[1] * nrows))
        plt.imshow(main_buffer)
        plt.axis('off')
        
        if save_as:
            plt.savefig(save_as, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    # No longer return fig to avoid duplicate rendering by environment
    return None
    # return fig


def _plot_state_distribution_single(seqdata: SequenceData,
                                    weights="auto",
                                    figsize=(12, 7),
                                    plot_style="standard",
                                    title=None,
                                    xlabel="Time",
                                    ylabel="State Distribution (%)",
                                    stacked=True,
                                    save_as=None,
                                    dpi=200,
                                    show=False,
                                    include_legend=True,
                                    fontsize=12,
                                    show_title=True) -> None:
    """
    Creates a state distribution plot showing how the prevalence of states changes over time,
    with enhanced color vibrancy.

    :param seqdata: (SequenceData) A SequenceData object containing sequences
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param figsize: (tuple) Size of the figure (only used when plot_style="custom")
    :param plot_style: Plot aspect style ('standard', 'compact', 'wide', 'narrow', 'custom')
    :param title: (str) Optional title for the plot
    :param xlabel: (str) Label for the x-axis
    :param ylabel: (str) Label for the y-axis
    :param stacked: (bool) Whether to create a stacked area plot (True) or line plot (False)
    :param save_as: (str) Optional file path to save the plot
    :param dpi: (int) Resolution of the saved plot

    :return: None
    """
    # Determine figure size based on plot style
    style_sizes = {
        'standard': (12, 7),   # Balanced view
        'compact': (10, 8),    # More square, like R plots  
        'wide': (14, 5),       # Wide, emphasizes time
        'narrow': (9, 11),     # Moderately vertical
        'custom': figsize      # User-provided
    }
    
    if plot_style not in style_sizes:
        raise ValueError(f"Invalid plot_style '{plot_style}'. "
                        f"Supported styles: {list(style_sizes.keys())}")
    
    # Special validation for custom plot style
    if plot_style == 'custom' and figsize == (12, 7):
        raise ValueError(
            "When using plot_style='custom', you must explicitly provide a figsize parameter "
            "that differs from the default (12, 7). "
            "Suggested custom sizes:\n"
            "  - For wide plots: figsize=(16, 6)\n"
            "  - For tall plots: figsize=(8, 12)\n"
            "  - For square plots: figsize=(10, 10)\n"
            "  - For small plots: figsize=(8, 5)\n"
            "Example: plot_state_distribution(data, plot_style='custom', figsize=(14, 9))"
        )
    
    actual_figsize = style_sizes[plot_style]
    
    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    # Get sequence data as a DataFrame
    seq_df = seqdata.to_dataframe()
    
    # Get weights
    if weights is None:
        w = np.ones(len(seq_df))
    else:
        w = np.asarray(weights)

    # Broadcast weights to each time point
    W = np.repeat(w[:, None], seq_df.shape[1], axis=1)

    # Create a state mapping from numerical values back to state names
    inv_state_mapping = {v: k for k, v in seqdata.state_mapping.items()}

    # Calculate weighted state distributions at each time point
    distributions = []
    for t, col in enumerate(seq_df.columns):
        col_vals = seq_df[col].to_numpy()
        
        # Calculate weighted counts for each state
        sums = {s: float(W[col_vals == s, t].sum()) for s in range(1, len(seqdata.states)+1)}
        totw = float(W[:, t].sum())
        
        # Convert to weighted percentages
        dist = {inv_state_mapping.get(s, 'Missing'): 100.0 * (sums[s] / totw if totw > 0 else 0.0) 
                for s in range(1, len(seqdata.states) + 1)}

        # Add time point and distribution to the list
        distributions.append(dict({"time": col, **{str(k): v for k, v in dist.items()}}))

    # Ensure percentages sum to exactly 100% to avoid gaps
    for i in range(len(distributions)):
        # Get sum of all state percentages for this time point
        total_percentage = sum(distributions[i][str(state)] for state in seqdata.states)

        # If there's a gap, add the difference to the top-most state
        if total_percentage < 100:
            # Get the last (top-most) state in your stack
            top_state = str(seqdata.states[-1])
            # Add the difference to make total exactly 100%
            distributions[i][top_state] += (100 - total_percentage)

    # Convert to DataFrame for plotting
    dist_df = pd.DataFrame(distributions)

    # Create the plot
    plt.style.use('default')  # Start with default style for clean slate
    fig, ax = plt.subplots(figsize=actual_figsize)

    # Get colors for each state and enhance vibrancy
    base_colors = [seqdata.color_map[seqdata.state_mapping[state]] for state in seqdata.states]

    # Plot the data
    if stacked:
        # Create a stacked area plot with enhanced colors
        ax.stackplot(range(len(dist_df)),
                     [dist_df[str(state)] for state in seqdata.states],
                     labels=seqdata.labels,
                     colors=base_colors,
                     alpha=1.0)  # Full opacity for maximum vibrancy

        # Add grid lines behind the stack plot
        ax.grid(axis='y', linestyle='-', alpha=0.2)
        ax.set_axisbelow(True)
    else:
        # Create a line plot with enhanced colors
        for i, state in enumerate(seqdata.states):
            ax.plot(range(len(dist_df)), dist_df[str(state)],
                    label=state, color=base_colors[i],
                    linewidth=2.5, marker='o', markersize=5)

        # Add grid lines
        ax.grid(True, linestyle='-', alpha=0.2)

    # Set axis labels and title
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10)

    if title and show_title:
        ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)

    # Set x-axis labels based on time points
    set_up_time_labels_for_x_axis(seqdata, ax)
    
    # Set x-axis range to prevent over-extension like in the reference image
    ax.set_xlim(-0.5, len(seqdata.cleaned_time) - 0.5)

    # Enhance aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    
    # Move spines slightly away from the plot area for better aesthetics (same as index plot)
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    
    # Ensure ticks are visible and styled consistently
    ax.tick_params(axis='x', colors='gray', length=4, width=0.7, which='major')
    ax.tick_params(axis='y', colors='gray', length=4, width=0.7, which='major')

    # Set y-axis limits from 0 to 100%
    ax.set_ylim(0, 100)

    # Add legend
    if include_legend:
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                           frameon=False, fontsize=fontsize-2)

    # Adjust layout to make room for the legend
    plt.tight_layout()

    save_and_show_results(save_as, dpi=dpi, show=show)

    # return fig
    # No longer return fig to avoid duplicate rendering by environment
    return None


