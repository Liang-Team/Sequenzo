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
    determine_layout
)


def plot_state_distribution(seqdata: SequenceData,
                            id_group_df=None,
                            categories=None,
                            figsize=(12, 7),
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
                            include_legend=True) -> None:
    """
    Creates state distribution plots for different groups, showing how state
    prevalence changes over time within each group.

    :param seqdata: (SequenceData) A SequenceData object containing sequences
    :param id_group_df: DataFrame with entity IDs and group information (if None, creates a single plot)
    :param categories: Column name in id_group_df that contains grouping information
    :param figsize: (tuple) Size of the figure
    :param title: (str) Optional title for the plot
    :param xlabel: (str) Label for the x-axis
    :param ylabel: (str) Label for the y-axis
    :param save_as: (str) Optional file path to save the plot
    :param dpi: (int) Resolution of the saved plot
    :param layout: (str) Layout style - 'column' (default, 3xn), 'grid' (nxn)
    :param stacked: (bool) Whether to create stacked area plots (True) or line plots (False)

    :return: None
    """
    # If no grouping information, create a single plot
    if id_group_df is None or categories is None:
        return _plot_state_distribution_single(
            seqdata=seqdata, figsize=figsize,
            title=title, xlabel=xlabel, ylabel=ylabel,
            save_as=save_as, dpi=dpi, stacked=stacked,
            show=show, include_legend=include_legend
        )

    # Ensure ID columns match (convert if needed)
    id_col_name = "Entity ID" if "Entity ID" in id_group_df.columns else id_group_df.columns[0]

    # Get unique groups and sort them
    groups = sorted(id_group_df[categories].unique())
    num_groups = len(groups)

    # Calculate figure size and layout based on number of groups and specified layout
    nrows, ncols = determine_layout(num_groups, layout=layout, nrows=nrows, ncols=ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
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

        # Calculate state distributions at each time point
        distributions = []
        for col in group_seq_df.columns:
            # Count occurrences of each state at this time point
            state_counts = group_seq_df[col].value_counts().sort_index()

            # Convert to percentages
            total = len(group_seq_df)
            state_percentages = (state_counts / total) * 100

            # Create a dictionary with states as keys and percentages as values
            dist = {inv_state_mapping.get(i, 'Missing'): state_percentages.get(i, 0)
                    for i in range(1, len(seqdata.states) + 1)}

            # Add time point and distribution to the list
            # distributions.append(dict(time=col, **dist))
            distributions.append(dict({"time": col, **{str(k): v for k, v in dist.items()}}))

        # Ensure percentages sum to exactly 100% to avoid gaps
        for j in range(len(distributions)):
            total_percentage = sum(distributions[j][str(state)] for state in seqdata.states)
            if total_percentage < 100:
                top_state = seqdata.states[-1]
                distributions[j][str(top_state)] += (100 - total_percentage)

        # Convert to DataFrame for plotting
        dist_df = pd.DataFrame(distributions)

        # Plot on the corresponding axis
        ax = axes[i]

        # Get colors for each state
        # seqdata.states 是整数编码（如 1, 2, ...）
        # seqdata.state_mapping[state] 把整数映射为 label（如 'Married', 'Single'）
        # seqdata.color_map[...] 用 label 取颜色
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

        # Set group title
        group_title = f"{categories} {group} (n = {len(group_seq_df)})"
        ax.set_title(group_title, fontsize=12, loc='right')

        # Set y-axis limits from 0 to 100%
        ax.set_ylim(0, 100)

        # Clean up axis aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)
        ax.tick_params(axis='x', colors='gray', length=4, width=0.7)
        ax.tick_params(axis='y', colors='gray', length=4, width=0.7)

        # Set x-axis labels
        set_up_time_labels_for_x_axis(seqdata, ax)

        # Add axis labels
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10, color='black')

        if i >= num_groups - ncols:
            ax.set_xlabel(xlabel, fontsize=12, labelpad=10, color='black')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Add a common title if provided
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    # Adjust layout to remove tight_layout warning
    fig.subplots_adjust(wspace=0.2, hspace=0.3, bottom=0.1, top=0.9, right=0.9)

    # Save main figure to memory
    main_buffer = save_figure_to_buffer(fig, dpi=dpi)

    # Create standalone legend
    colors = dict(zip(seqdata.labels, [seqdata.color_map[seqdata.state_mapping[state]] for state in seqdata.states]))
    legend_buffer = create_standalone_legend(
        colors=colors,
        labels=seqdata.labels,
        ncol=min(5, len(seqdata.states)),
        figsize=(figsize[0] * ncols, 1),
        fontsize=10,
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
    plt.figure(figsize=(figsize[0] * ncols, figsize[1] * nrows + 1))
    plt.imshow(combined_img)
    plt.axis('off')
    if show or save_as:  # 显示或保存都需要触发
        plt.show()
    plt.close()

    # 不再返回 fig，避免它被环境自动渲染重复
    return None
    # return fig


def _plot_state_distribution_single(seqdata: SequenceData,
                                    figsize=(12, 7),
                                    title=None,
                                    xlabel="Time",
                                    ylabel="State Distribution (%)",
                                    stacked=True,
                                    save_as=None,
                                    dpi=200,
                                    show=False,
                                    include_legend=True) -> None:
    """
    Creates a state distribution plot showing how the prevalence of states changes over time,
    with enhanced color vibrancy.

    :param seqdata: (SequenceData) A SequenceData object containing sequences
    :param figsize: (tuple) Size of the figure
    :param title: (str) Optional title for the plot
    :param xlabel: (str) Label for the x-axis
    :param ylabel: (str) Label for the y-axis
    :param stacked: (bool) Whether to create a stacked area plot (True) or line plot (False)
    :param save_as: (str) Optional file path to save the plot
    :param dpi: (int) Resolution of the saved plot

    :return: None
    """
    # Get sequence data as a DataFrame
    seq_df = seqdata.to_dataframe()

    # Create a state mapping from numerical values back to state names
    inv_state_mapping = {v: k for k, v in seqdata.state_mapping.items()}

    # Calculate state distributions at each time point
    distributions = []
    for col in seq_df.columns:
        # Count occurrences of each state at this time point
        state_counts = seq_df[col].value_counts().sort_index()

        # Convert to percentages
        total = len(seq_df)
        state_percentages = (state_counts / total) * 100

        # Create a dictionary with states as keys and percentages as values
        # Ensure all states are included (with 0% if not present)
        dist = {inv_state_mapping.get(i, 'Missing'): state_percentages.get(i, 0)
                for i in range(1, len(seqdata.states) + 1)}

        # Add time point and distribution to the list
        # distributions.append(dict(time=col, **dist))
        distributions.append(dict({"time": col, **{str(k): v for k, v in dist.items()}}))

    # Ensure percentages sum to exactly 100% to avoid gaps
    for i in range(len(distributions)):
        # Get sum of all state percentages for this time point
        total_percentage = sum(distributions[i][str(state)] for state in seqdata.states)

        # If there's a gap, add the difference to the top-most state
        if total_percentage < 100:
            # Get the last (top-most) state in your stack
            top_state = seqdata.states[-1]
            # Add the difference to make total exactly 100%
            distributions[i][top_state] += (100 - total_percentage)

    # Convert to DataFrame for plotting
    dist_df = pd.DataFrame(distributions)

    # Create the plot
    plt.style.use('default')  # Start with default style for clean slate
    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set x-axis labels based on time points
    set_up_time_labels_for_x_axis(seqdata, ax)

    # Enhance aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    # Set y-axis limits from 0 to 100%
    ax.set_ylim(0, 100)

    # Add legend
    if include_legend:
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                           frameon=False, fontsize=10)

    # Adjust layout to make room for the legend
    plt.tight_layout()

    save_and_show_results(save_as, dpi=dpi, show=show)

    # return fig
    # 不再返回 fig，避免它被环境自动渲染重复
    return None


