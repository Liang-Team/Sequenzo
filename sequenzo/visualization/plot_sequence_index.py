"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_sequence_index.py
@Time    : 29/12/2024 09:08
@Desc    : 
    Generate sequence index plots.
"""
import numpy as np
import matplotlib.pyplot as plt

# Use relative import to avoid circular import when top-level package imports visualization
from ..define_sequence_data import SequenceData
from sequenzo.visualization.utils import (
    set_up_time_labels_for_x_axis,
    save_figure_to_buffer,
    create_standalone_legend,
    combine_plot_with_legend,
    save_and_show_results,
    determine_layout,
    show_plot_title
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


def _cmdscale(D):
    """
    Classic Multidimensional Scaling (MDS), equivalent to R's cmdscale()
    
    :param D: A NxN symmetric distance matrix
    :return: Y, a Nxd coordinate matrix, where d is the largest positive eigenvalues' count
    """
    n = len(D)
    
    # Step 1: Compute the centering matrix
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Step 2: Compute the double centered distance matrix
    B = -0.5 * H @ (D ** 2) @ H
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Step 4: Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Step 5: Select only positive eigenvalues
    w, = np.where(eigvals > 0)
    if len(w) > 0:
        L = np.diag(np.sqrt(eigvals[w]))
        V = eigvecs[:, w]
        return V @ L  # Return the MDS coordinates
    else:
        # Fallback if no positive eigenvalues
        return np.zeros((n, 1))


def _find_most_frequent_sequence(sequences):
    """
    Find the most frequent sequence in the dataset.
    
    :param sequences: numpy array of sequences
    :return: index of the most frequent sequence
    """
    from collections import Counter
    
    # Convert sequences to tuples for hashing
    seq_tuples = [tuple(seq) for seq in sequences]
    
    # Count frequencies
    counter = Counter(seq_tuples)
    
    # Find the most frequent sequence
    most_frequent = counter.most_common(1)[0][0]
    
    # Find the index of this sequence in the original array
    for i, seq in enumerate(seq_tuples):
        if seq == most_frequent:
            return i
    
    return 0  # Fallback


def sort_sequences_by_method(seqdata, method="unsorted", mask=None, distance_matrix=None, weights=None):
    """
    Sort sequences in SequenceData based on specified method.
    
    :param seqdata: SequenceData object
    :param method: str, sorting method - "unsorted", "lexicographic", "mds", "distance_to_most_frequent"
    :param mask: np.array(bool), if provided, sort only this subset
    :param distance_matrix: np.array, required for "mds" and "distance_to_most_frequent" methods
    :param weights: np.array, optional weights for sequences
    :return: np.array sorting indices (relative to original order)
    """
    values = seqdata.values.copy()
    
    n_sequences = len(values) if mask is None else int(np.sum(mask))
    
    if mask is not None:
        values = values[mask]
        if distance_matrix is not None:
            # Only slice if distance_matrix is for the full sample
            if distance_matrix.shape[0] != n_sequences:
                masked_indices = np.where(mask)[0]
                distance_matrix = distance_matrix[np.ix_(masked_indices, masked_indices)]
    
    if method == "unsorted" or method == "none":
        # Keep original order (R default)
        return np.arange(n_sequences)
    
    elif method == "lexicographic":
        # Lexicographic sorting (NaN-safe)
        vals = values.astype(float, copy=True)
        # Push NaNs to the end for sorting
        vals = np.nan_to_num(vals, nan=np.inf)
        return np.lexsort(vals.T[::-1])
    
    elif method == "mds":
        # MDS first dimension sorting
        if distance_matrix is None:
            raise ValueError("Distance matrix is required for MDS sorting")
        
        # TODO: Support weighted MDS (TraMineR's wcmdscale analogue) when weights are provided.
        # Compute MDS coordinates
        mds_coords = _cmdscale(distance_matrix)
        
        # Sort by first MDS dimension
        return np.argsort(mds_coords[:, 0])
    
    elif method == "distance_to_most_frequent":
        # Sort by distance to most frequent sequence
        if distance_matrix is None:
            raise ValueError("Distance matrix is required for distance_to_most_frequent sorting")
        
        # Find most frequent sequence
        most_freq_idx = _find_most_frequent_sequence(values)
        
        # Get distances to most frequent sequence
        distances = distance_matrix[most_freq_idx, :]
        
        # Sort by distance (ascending)
        return np.argsort(distances)
    
    else:
        raise ValueError(f"Unsupported sorting method: {method}. "
                        f"Supported methods are: 'unsorted', 'lexicographic', 'mds', 'distance_to_most_frequent'")


def plot_sequence_index(seqdata: SequenceData,
                        id_group_df=None,
                        categories=None,
                        sort_by="lexicographic",
                        sort_by_weight=False,
                        weights="auto",
                        figsize=(10, 6),
                        title=None,
                        xlabel="Time",
                        ylabel="Sequences",
                        save_as=None,
                        dpi=200,
                        layout='column',
                        nrows: int = None,
                        ncols: int = None,
                        group_order=None,
                        sort_groups='auto',
                        fontsize=12,
                        show_group_titles: bool = True
                        ):
    """Creates sequence index plots, optionally grouped by categories.
    
    This function creates index plots that visualize sequences as horizontal lines,
    with different sorting options matching R's TraMineR functionality.

    :param seqdata: SequenceData object containing sequence information
    :param id_group_df: DataFrame with entity IDs and group information (if None, creates a single plot)
    :param categories: Column name in id_group_df that contains grouping information
    :param sort_by: Sorting method for sequences within groups:
        - 'unsorted' or 'none': Keep original order (R TraMineR default)
        - 'lexicographic': Sort sequences lexicographically
        - 'mds': Sort by first MDS dimension (requires distance computation)
        - 'distance_to_most_frequent': Sort by distance to most frequent sequence
    :param sort_by_weight: If True, sort sequences by weight (descending), overrides sort_by
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param figsize: Size of each subplot figure
    :param title: Title for the plot (if None, default titles will be used)
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param save_as: File path to save the plot (if None, plot will be shown)
    :param dpi: DPI for saved image
    :param layout: Layout style - 'column' (default, 3xn), 'grid' (nxn)
    :param group_order: List, manually specify group order (overrides sort_groups)
    :param sort_groups: String, sorting method: 'auto'(smart numeric), 'numeric'(numeric prefix), 'alpha'(alphabetical), 'none'(original order)
    :param fontsize: Base font size for text elements (titles use fontsize+2, ticks use fontsize-2)
    :param show_group_titles: Whether to show group titles
    
    Note: For 'mds' and 'distance_to_most_frequent' sorting, distance matrices are computed
    automatically using Optimal Matching (OM) with constant substitution costs.
    """
    # If no grouping information, create a single plot
    if id_group_df is None or categories is None:
        return _sequence_index_plot_single(seqdata, sort_by, sort_by_weight, weights, figsize, title, xlabel, ylabel, save_as, dpi, fontsize)

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
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        gridspec_kw={'wspace': 0.2, 'hspace': 0.3}
    )
    axes = axes.flatten()

    # Create a plot for each group
    for i, group in enumerate(groups):
        # Get IDs for this group
        group_ids = id_group_df[id_group_df[categories] == group][id_col_name].values

        # Match IDs with sequence data
        mask = np.isin(seqdata.ids, group_ids)
        if not np.any(mask):
            print(f"Warning: No matching sequences found for group '{group}'")
            continue

        # Extract sequences for this group
        group_sequences = seqdata.values[mask]
        
        # Get weights for this group
        if weights is not None:
            group_weights = weights[mask]
        else:
            group_weights = None

        # Handle NaN values for better visualization
        if np.isnan(group_sequences).any():
            # Map NaN to a dedicated state code with proper masking
            group_sequences = group_sequences.astype(float)
            group_sequences[np.isnan(group_sequences)] = np.nan

        if sort_by_weight and group_weights is not None:
            # Sort by weight (descending)
            sorted_indices = np.argsort(-group_weights)
        else:
            # Use the new sorting method (requires distance matrix for some methods)
            distance_matrix = None
            if sort_by in ["mds", "distance_to_most_frequent"]:
                # Import distance calculation function
                try:
                    from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
                    
                    # Create subset of seqdata for this group
                    group_seqdata = SequenceData(
                        values=group_sequences,
                        ids=seqdata.ids[mask] if hasattr(seqdata, 'ids') else None,
                        time_points=seqdata.time_points,
                        states=seqdata.states,
                        labels=seqdata.labels,
                        color_map=seqdata.color_map
                    )
                    
                    # Compute distance matrix for this group
                    distance_matrix = get_distance_matrix(
                        seqdata=group_seqdata, 
                        method="OM", 
                        sm="CONSTANT", 
                        indel="auto"
                    )
                    
                    # Convert to numpy array if it's a DataFrame
                    if hasattr(distance_matrix, 'values'):
                        distance_matrix = distance_matrix.values
                        
                except ImportError:
                    print(f"Warning: Cannot compute distance matrix for '{sort_by}' sorting. Using unsorted order.")
                    sort_by = "unsorted"
                    
            sorted_indices = sort_sequences_by_method(
                seqdata=seqdata, 
                method=sort_by, 
                mask=mask,
                distance_matrix=distance_matrix,
                weights=group_weights
            )

        sorted_data = group_sequences[sorted_indices]

        # Plot on the corresponding axis
        ax = axes[i]
        # Use masked array for better NaN handling
        data = sorted_data.astype(float)
        data[data < 1] = np.nan
        
        # Check for all-missing or all-invalid data
        if np.all(~np.isfinite(data)):
            print(f"Warning: all values missing/invalid for group '{group}'")
            ax.axis('off')
            continue
            
        im = ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap=seqdata.get_colormap(),
                       interpolation='nearest', vmin=1, vmax=len(seqdata.states))

        # Remove grid lines
        ax.grid(False)

        # Set up time labels
        set_up_time_labels_for_x_axis(seqdata, ax)

        # Enhance y-axis aesthetics - evenly spaced ticks including the last sequence
        num_sequences = sorted_data.shape[0]
        num_ticks = min(11, num_sequences)
        ytick_positions = np.linspace(0, num_sequences - 1, num=num_ticks, dtype=int)
        ytick_positions = np.unique(ytick_positions)
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels((ytick_positions + 1).astype(int), fontsize=fontsize-2, color='black')

        # Customize axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)
        ax.tick_params(axis='x', colors='gray', length=4, width=0.7)
        ax.tick_params(axis='y', colors='gray', length=4, width=0.7)

        # Add group title with weight information
        # Check if we have effective weights (not all 1.0) and they were provided by user
        original_weights = getattr(seqdata, "weights", None)
        if original_weights is not None and not np.allclose(original_weights, 1.0) and group_weights is not None:
            sum_w = float(group_weights.sum())
            group_title = f"{group} (n = {num_sequences}, total weight = {sum_w:.1f})"
        else:
            group_title = f"{group} (n = {num_sequences})"
        if show_group_titles:
            show_plot_title(ax, group_title, show=True, fontsize=fontsize, loc='right')

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

    # Create standalone legend
    colors = seqdata.color_map_by_label
    legend_buffer = create_standalone_legend(
        colors=colors,
        labels=seqdata.labels,
        ncol=min(5, len(seqdata.states)),
        figsize=(figsize[0] * ncols, 1),
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
    plt.figure(figsize=(figsize[0] * ncols, figsize[1] * nrows + 1))
    plt.imshow(combined_img)
    plt.axis('off')
    plt.show()
    plt.close()


def _sequence_index_plot_single(seqdata: SequenceData,
                                sort_by="unsorted",
                                sort_by_weight=False,
                                weights="auto",
                                figsize=(10, 6),
                                title=None,
                                xlabel="Time",
                                ylabel="Sequences",
                                save_as=None,
                                dpi=200,
                                fontsize=12):
    """Efficiently creates a sequence index plot using `imshow` for faster rendering.

    :param seqdata: SequenceData object containing sequence information
    :param sort_by: Sorting method ('unsorted', 'lexicographic', 'mds', 'distance_to_most_frequent')
    :param sort_by_weight: If True, sort sequences by weight (descending), overrides sort_by
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param figsize: (tuple): Size of the figure.
    :param title: (str): Title for the plot.
    :param xlabel: (str): Label for the x-axis.
    :param ylabel: (str): Label for the y-axis.
    :param save_as: File path to save the plot
    :param dpi: DPI for saved image

    :return None.
    """
    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    # Get sequence values as NumPy array
    sequence_values = seqdata.values.copy()

    # Handle NaN values for better visualization
    if np.isnan(sequence_values).any():
        # Keep NaN as float for proper masking
        sequence_values = sequence_values.astype(float)

    # Sort sequences based on specified method
    if sort_by_weight and weights is not None:
        # Sort by weight (descending)
        sorted_indices = np.argsort(-weights)
    else:
        # Use the new sorting method
        distance_matrix = None
        if sort_by in ["mds", "distance_to_most_frequent"]:
            # Import distance calculation function
            try:
                from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
                
                # Compute distance matrix
                distance_matrix = get_distance_matrix(
                    seqdata=seqdata, 
                    method="OM", 
                    sm="CONSTANT", 
                    indel="auto"
                )
                
                # Convert to numpy array if it's a DataFrame
                if hasattr(distance_matrix, 'values'):
                    distance_matrix = distance_matrix.values
                    
            except ImportError:
                print(f"Warning: Cannot compute distance matrix for '{sort_by}' sorting. Using unsorted order.")
                sort_by = "unsorted"
                
        sorted_indices = sort_sequences_by_method(
            seqdata=seqdata, 
            method=sort_by, 
            distance_matrix=distance_matrix,
            weights=weights
        )
    
    sorted_data = sequence_values[sorted_indices]

    # Create the plot using imshow with proper NaN handling
    fig, ax = plt.subplots(figsize=figsize)
    # Use masked array for better NaN handling
    data = sorted_data.astype(float)
    data[data < 1] = np.nan
    
    # Check for all-missing or all-invalid data
    if np.all(~np.isfinite(data)):
        print(f"Warning: all values missing/invalid in sequence data")
        ax.axis('off')
        return
        
    ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap=seqdata.get_colormap(), 
              interpolation='nearest', vmin=1, vmax=len(seqdata.states))

    # Disable background grid and all axis guide lines
    ax.grid(False)

    # Optional: remove tick marks and tick labels to avoid visual grid effects
    # ax.set_xticks([])
    # ax.set_yticks([])

    # x label
    set_up_time_labels_for_x_axis(seqdata, ax)

    # Enhance y-axis aesthetics - evenly spaced ticks including the last sequence
    num_sequences = sorted_data.shape[0]
    num_ticks = min(11, num_sequences)
    ytick_positions = np.linspace(0, num_sequences - 1, num=num_ticks, dtype=int)
    ytick_positions = np.unique(ytick_positions)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels((ytick_positions + 1).astype(int), fontsize=fontsize-2, color='black')


    # Customize axis line styles and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.tick_params(axis='x', colors='gray', length=4, width=0.7)
    ax.tick_params(axis='y', colors='gray', length=4, width=0.7)

    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color='black')
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10, color='black')
    
    # Set title with weight information if available
    if title is not None:
        display_title = title
        
        # Check if we have effective weights (not all 1.0) and they were provided by user
        original_weights = getattr(seqdata, "weights", None)
        if original_weights is not None and not np.allclose(original_weights, 1.0) and weights is not None:
            sum_w = float(weights.sum())
            display_title += f" (n = {num_sequences}, total weight = {sum_w:.1f})"
        else:
            display_title += f" (n = {num_sequences})"
        
        ax.set_title(display_title, fontsize=fontsize+2, color='black')

    # Use legend from SequenceData
    ax.legend(*seqdata.get_legend(), bbox_to_anchor=(1.05, 1), loc='upper left')

    save_and_show_results(save_as, dpi=dpi)