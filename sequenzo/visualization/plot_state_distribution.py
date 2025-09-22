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
                            id_group_df=None,
                            categories=None,
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
                            show_group_titles: bool = True) -> None:
    """
    Creates state distribution plots for different groups, showing how state
    prevalence changes over time within each group.

    :param seqdata: (SequenceData) A SequenceData object containing sequences
    :param id_group_df: DataFrame with entity IDs and group information (if None, creates a single plot)
    :param categories: Column name in id_group_df that contains grouping information
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
    
    # If no grouping information, create a single plot
    if id_group_df is None or categories is None:
        return _plot_state_distribution_single(
            seqdata=seqdata, weights=weights, figsize=actual_figsize,
            plot_style=plot_style, title=title, xlabel=xlabel, ylabel=ylabel,
            save_as=save_as, dpi=dpi, stacked=stacked,
            show=show, include_legend=include_legend, fontsize=fontsize
        )

    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    # Ensure ID columns match (convert if needed)
    id_col_name = "Entity ID" if "Entity ID" in id_group_df.columns else id_group_df.columns[0]

    # Get unique groups and sort them based on user preference
    if group_order:
        # Use manually specified order, filter out non-existing groups
        groups = [g for g in group_order if g in id_group_df[categories].unique()]
        missing_groups = [g for g in id_group_df[categories].unique() if g not in group_order]
        if missing_groups:
            print(f"[Warning] Groups not in group_order will be excluded: {missing_groups}")
    elif sort_groups == 'numeric' or sort_groups == 'auto':
        groups = smart_sort_groups(id_group_df[categories].unique())
    elif sort_groups == 'alpha':
        groups = sorted(id_group_df[categories].unique())
    elif sort_groups == 'none':
        groups = list(id_group_df[categories].unique())
    else:
        raise ValueError(f"Invalid sort_groups value: {sort_groups}. Use 'auto', 'numeric', 'alpha', or 'none'.")
    
    num_groups = len(groups)

    # Calculate figure size and layout based on number of groups and specified layout
    nrows, ncols = determine_layout(num_groups, layout=layout, nrows=nrows, ncols=ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(actual_figsize[0] * ncols, actual_figsize[1] * nrows),
        gridspec_kw={'wspace': 0.2, 'hspace': 0.3}
    )
    axes = axes.flatten()

    # Create state mapping from numerical values back to state names
    inv_state_mapping = {v: k for k, v in seqdata.state_mapping.items()}

    # Process each group
    for i, group in enumerate(groups):
        # Get IDs for this group
        group_ids = id_group_df[id_group_df[categories] == group][id_col_name].values

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
        base_colors = [seqdata.color_map_by_label[state] for state in seqdata.states]

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

    # Add a common title if provided
    if title:
        fig.suptitle(title, fontsize=fontsize+2, y=1.02)

    # Adjust layout to remove tight_layout warning
    fig.subplots_adjust(wspace=0.2, hspace=0.3, bottom=0.1, top=0.9, right=0.9)

    # Save main figure to memory
    main_buffer = save_figure_to_buffer(fig, dpi=dpi)

    if include_legend:
        # Create standalone legend
        colors = dict(zip(seqdata.labels, [seqdata.color_map_by_label[state] for state in seqdata.states]))
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
                                    fontsize=12) -> None:
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

    if title:
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


