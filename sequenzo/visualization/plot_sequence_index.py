"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_sequence_index.py
@Time    : 29/12/2024 09:08
@Desc    : 
    Generate sequence index plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def _select_sequences_subset(seqdata, sequence_selection, n_sequences, sort_by, sort_by_weight, weights, mask=None):
    """
    Select a subset of sequences based on the selection method.
    
    :param seqdata: SequenceData object
    :param sequence_selection: Selection method ("all", "first_n", "last_n", or list of IDs)
    :param n_sequences: Number of sequences for "first_n" or "last_n"
    :param sort_by: Sorting method to use before selection
    :param sort_by_weight: Whether to sort by weight
    :param weights: Sequence weights
    :param mask: Optional mask for pre-filtering sequences
    :return: Boolean mask for selected sequences
    """
    # Start with all sequences or pre-filtered mask
    if mask is None:
        mask = np.ones(len(seqdata.values), dtype=bool)
    
    # If "all", return the current mask
    if sequence_selection == "all":
        return mask
    
    # Get indices of sequences that pass the mask
    valid_indices = np.where(mask)[0]
    
    # Handle ID list selection
    if isinstance(sequence_selection, list):
        # Convert list to set for faster lookup
        selected_ids = set(sequence_selection)
        
        # Find indices of sequences with matching IDs
        selected_mask = np.zeros(len(seqdata.values), dtype=bool)
        if hasattr(seqdata, 'ids') and seqdata.ids is not None:
            for i in valid_indices:
                if seqdata.ids[i] in selected_ids:
                    selected_mask[i] = True
        else:
            print("Warning: sequence_selection provided as ID list but seqdata has no IDs. Using all sequences.")
            return mask
        
        return selected_mask
    
    # For "first_n" or "last_n", we need to sort first
    if sequence_selection in ["first_n", "last_n"]:
        # Get the subset of data based on current mask
        subset_seqdata = seqdata
        subset_weights = weights
        
        if not np.all(mask):
            # Create subset if mask is not all True
            subset_values = seqdata.values[mask]
            subset_ids = seqdata.ids[mask] if hasattr(seqdata, 'ids') and seqdata.ids is not None else None
            
            # Use original seqdata for structure, just work with filtered values
            subset_seqdata = seqdata  # Keep original structure
            
            if weights is not None:
                subset_weights = weights[mask]
        
        # Apply sorting to get the order
        distance_matrix = None
        if sort_by in ["mds", "distance_to_most_frequent"]:
            try:
                from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
                distance_matrix = get_distance_matrix(
                    seqdata=subset_seqdata,
                    method="OM",
                    sm="CONSTANT",
                    indel="auto"
                )
                if hasattr(distance_matrix, 'values'):
                    distance_matrix = distance_matrix.values
            except ImportError:
                print(f"Warning: Cannot compute distance matrix for '{sort_by}' sorting. Using unsorted order.")
                sort_by = "unsorted"
        
        # Apply sorting to the masked subset
        if sort_by_weight and subset_weights is not None:
            # Sort by weight on the subset
            sorted_indices = np.argsort(-subset_weights)
        else:
            # Sort on the subset values
            if sort_by == "unsorted" or sort_by == "none":
                sorted_indices = np.arange(len(valid_indices))
            elif sort_by == "lexicographic":
                subset_values = seqdata.values[mask]
                vals = subset_values.astype(float, copy=True)
                vals = np.nan_to_num(vals, nan=np.inf)
                sorted_indices = np.lexsort(vals.T[::-1])
            elif sort_by in ["mds", "distance_to_most_frequent"]:
                # For complex sorting that requires distance matrix, 
                # we'll fall back to simple lexicographic for now
                subset_values = seqdata.values[mask]
                vals = subset_values.astype(float, copy=True)
                vals = np.nan_to_num(vals, nan=np.inf)
                sorted_indices = np.lexsort(vals.T[::-1])
                print(f"Warning: {sort_by} sorting simplified to lexicographic for sequence selection")
            else:
                sorted_indices = np.arange(len(valid_indices))
        
        # Select first_n or last_n
        n_available = len(sorted_indices)
        n_to_select = min(n_sequences, n_available)
        
        if sequence_selection == "first_n":
            selected_subset_indices = sorted_indices[:n_to_select]
        elif sequence_selection == "last_n":
            selected_subset_indices = sorted_indices[-n_to_select:]
        
        # Map back to original indices
        original_indices = valid_indices[selected_subset_indices]
        
        # Create final mask
        final_mask = np.zeros(len(seqdata.values), dtype=bool)
        final_mask[original_indices] = True
        
        return final_mask
    
    else:
        raise ValueError(f"Unsupported sequence_selection: {sequence_selection}. "
                        f"Supported options: 'all', 'first_n', 'last_n', or list of IDs")


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
                        # Grouping parameters
                        group_by_column=None,
                        group_dataframe=None,
                        group_column_name=None,
                        group_labels=None,
                        # Other parameters
                        sort_by="lexicographic",
                        sort_by_weight=False,
                        weights="auto",
                        figsize=(10, 6),
                        plot_style="standard",
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
                        show_group_titles: bool = True,
                        include_legend: bool = True,
                        sequence_selection="all",
                        n_sequences=10,
                        show_sequence_ids=False,
                        sort_by_ids=None,
                        return_sorted_ids=False,
                        show_title=True,
                        proportional_scaling=False,
                        hide_y_axis=False
                        ):
    """Creates sequence index plots, optionally grouped by categories.
    
    This function creates index plots that visualize sequences as horizontal lines,
    with different sorting options matching R's TraMineR functionality.

    **Two API modes for grouping:**
    
    1. **Simplified API** (when grouping info is already in the data):
       ```python
       plot_sequence_index(seqdata, group_by_column="Cluster", group_labels=cluster_labels)
       ```
    
    2. **Complete API** (when grouping info is in a separate dataframe):
       ```python
       plot_sequence_index(seqdata, group_dataframe=membership_df, 
                          group_column_name="Cluster", group_labels=cluster_labels)
       ```

    :param seqdata: SequenceData object containing sequence information
    
    **New API parameters (recommended):**
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
    
    :param sort_by: Sorting method for sequences within groups:
        - 'unsorted' or 'none': Keep original order (R TraMineR default)
        - 'lexicographic': Sort sequences lexicographically
        - 'mds': Sort by first MDS dimension (requires distance computation)
        - 'distance_to_most_frequent': Sort by distance to most frequent sequence
    :param sort_by_weight: If True, sort sequences by weight (descending), overrides sort_by
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param figsize: Size of each subplot figure (only used when plot_style="custom")
    :param plot_style: Plot aspect style:
        - 'standard': Standard proportions (10, 6) - balanced view
        - 'compact': Compact/vertical proportions (8, 8) - more vertical like R plots
        - 'wide': Wide proportions (12, 4) - emphasizes time progression  
        - 'narrow': Narrow/tall proportions (8, 10) - moderately vertical
        - 'custom': Use the provided figsize parameter
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
    :param include_legend: Whether to include legend in the plot (True by default)
    :param sequence_selection: Method for selecting sequences to visualize:
        - "all": Show all sequences (default)
        - "first_n": Show first n sequences from each group
        - "last_n": Show last n sequences from each group  
        - list: List of specific sequence IDs to show
    :param n_sequences: Number of sequences to show when using "first_n" or "last_n" (default: 10)
    :param show_sequence_ids: If True, show actual sequence IDs on y-axis instead of sequence numbers. 
        Most useful when sequence_selection is a list of IDs (default: False)
    :param sort_by_ids: (list or np.ndarray, optional) Custom ID order for sorting sequences.
                       When provided, sequences will be sorted to match this ID order, overriding 
                       the sort_by parameter. This is useful for aligning multiple plots so that 
                       the same IDs appear in the same row across different visualizations.
                       Example: sort_by_ids=[1, 3, 2, 5, 4] will sort sequences by this exact order.
    :param return_sorted_ids: (bool, default: False) If True, returns the sorted ID order after plotting.
                             This is useful for multidomain analysis where you want to use the sorted 
                             IDs from the first plot to align subsequent plots.
                             Returns a dictionary with group names as keys and sorted ID arrays as values
                             (for grouped plots), or a single array of sorted IDs (for single plots).
    :param show_title: (bool, default: True) If False, suppresses the main title display even if title parameter is provided.
                      This allows you to control title visibility separately from providing a title string.
    :param proportional_scaling: (bool, default: False) If True, scales subplot heights proportionally based on 
                                the number of sequences in each group. Useful when groups have very different sizes.
                                Only applies to grouped plots with layout='column'.
    :param hide_y_axis: (bool, default: False) If True, hides y-axis ticks, labels, and spine for all subplots.
                       Useful when using proportional_scaling to create cleaner visualizations.
    
    Note: For 'mds' and 'distance_to_most_frequent' sorting, distance matrices are computed
    automatically using Optimal Matching (OM) with constant substitution costs.
    """
    # Determine figure size based on plot style
    style_sizes = {
        'standard': (10, 6),   # Balanced view
        'compact': (8, 8),     # More square, like R plots  
        'wide': (12, 4),       # Wide, emphasizes time
        'narrow': (8, 10),     # Moderately vertical
        'custom': figsize      # User-provided
    }
    
    if plot_style not in style_sizes:
        raise ValueError(f"Invalid plot_style '{plot_style}'. "
                        f"Supported styles: {list(style_sizes.keys())}")
    
    # Special validation for custom plot style
    if plot_style == 'custom' and figsize == (10, 6):
        raise ValueError(
            "When using plot_style='custom', you must explicitly provide a figsize parameter "
            "that differs from the default (10, 6). "
            "Suggested custom sizes:\n"
            "  - For wide plots: figsize=(15, 5)\n"
            "  - For tall plots: figsize=(7, 12)\n"
            "  - For square plots: figsize=(9, 9)\n"
            "  - For small plots: figsize=(6, 4)\n"
            "Example: plot_sequence_index(data, plot_style='custom', figsize=(12, 8))"
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
        return _sequence_index_plot_single(seqdata, sort_by, sort_by_weight, weights, actual_figsize, plot_style, title, xlabel, ylabel, save_as, dpi, fontsize, include_legend, sequence_selection, n_sequences, show_sequence_ids, sort_by_ids, return_sorted_ids, show_title)

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

    # Calculate height ratios for proportional scaling if enabled
    if proportional_scaling and layout == 'column':
        # First pass: collect group sizes
        group_sizes = []
        for group in groups:
            group_ids = group_dataframe[group_dataframe[group_column_name] == group][id_col_name].values
            mask = np.isin(seqdata.ids, group_ids)
            if np.any(mask):
                mask = _select_sequences_subset(seqdata, sequence_selection, n_sequences, sort_by, sort_by_weight, weights, mask)
                group_sizes.append(int(np.sum(mask)))
            else:
                group_sizes.append(1)
        
        # Calculate height ratios (min 0.3 to avoid too small subplots)
        if len(group_sizes) > 0:
            max_size = max(group_sizes)
            height_ratios = [max(0.3, size / max_size) for size in group_sizes]
            max_ratio = max(height_ratios)
            height_ratios = [h / max_ratio for h in height_ratios]
        else:
            height_ratios = [1.0] * num_groups
        
        # Use gridspec for proportional heights
        # Increase hspace to prevent x-axis labels from overlapping with subplot above
        # Use larger hspace for column layout to accommodate group titles and x-axis labels
        hspace_value = 0.4 if layout == 'column' else 0.25
        fig = plt.figure(figsize=(actual_figsize[0], actual_figsize[1] * sum(height_ratios) / len(height_ratios) * num_groups))
        gs = gridspec.GridSpec(nrows=num_groups, ncols=1, figure=fig, 
                               height_ratios=height_ratios, hspace=hspace_value, wspace=0.15)
        axes = [fig.add_subplot(gs[i]) for i in range(num_groups)]
    else:
        # Standard subplots with equal heights
        # Increase hspace for column layout to prevent x-axis labels from overlapping
        hspace_value = 0.4 if layout == 'column' else 0.25
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(actual_figsize[0] * ncols, actual_figsize[1] * nrows),
            gridspec_kw={'wspace': 0.15, 'hspace': hspace_value}
        )
        axes = axes.flatten()

    # Dictionary to store sorted IDs for each group (if return_sorted_ids is True)
    # Use OrderedDict or list to maintain group order
    sorted_ids_by_group = {}
    group_order_list = []  # Track the order of groups as they are processed

    # Create a plot for each group
    for i, group in enumerate(groups):
        # Get IDs for this group
        group_ids = group_dataframe[group_dataframe[group_column_name] == group][id_col_name].values

        # Match IDs with sequence data
        mask = np.isin(seqdata.ids, group_ids)
        if not np.any(mask):
            print(f"Warning: No matching sequences found for group '{group}'")
            continue
        
        # Apply sequence selection to this group
        mask = _select_sequences_subset(seqdata, sequence_selection, n_sequences, sort_by, sort_by_weight, weights, mask)

        # Extract sequences for this group
        group_sequences = seqdata.values[mask]
        
        # Track group IDs for y-axis labels
        group_ids_for_labels = None
        if hasattr(seqdata, 'ids') and seqdata.ids is not None and show_sequence_ids:
            group_ids_for_labels = seqdata.ids[mask]
        
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

        # Get the IDs for this group (after selection)
        group_ids_after_selection = seqdata.ids[mask]

        # Determine sorting method: sort_by_ids takes priority if provided
        if sort_by_ids is not None:
            # Sort by custom ID order
            # Convert sort_by_ids to numpy array for easier handling
            sort_by_ids_array = np.asarray(sort_by_ids)
            
            # Create a mapping from ID to position in sort_by_ids
            # IDs not in sort_by_ids will be placed at the end
            id_to_position = {id_val: pos for pos, id_val in enumerate(sort_by_ids_array)}
            
            # Get positions for each ID in the current group
            # IDs not in sort_by_ids get a very large position value (placed at end)
            max_position = len(sort_by_ids_array)
            positions = np.array([id_to_position.get(id_val, max_position + i) 
                                 for i, id_val in enumerate(group_ids_after_selection)])
            
            # Sort by position (ascending order)
            sorted_indices = np.argsort(positions)
            
            # Warn if some IDs in the group are not in sort_by_ids
            missing_ids = set(group_ids_after_selection) - set(sort_by_ids_array)
            if missing_ids:
                print(f"[Warning] Group '{group}': {len(missing_ids)} IDs not found in sort_by_ids, "
                      f"they will be placed at the end: {list(missing_ids)[:5]}{'...' if len(missing_ids) > 5 else ''}")
        
        elif sort_by_weight and group_weights is not None:
            # Sort by weight (descending)
            sorted_indices = np.argsort(-group_weights)
        else:
            # For group plots, we'll use simpler sorting to avoid complex object creation
            if sort_by == "lexicographic":
                vals = group_sequences.astype(float, copy=True)
                vals = np.nan_to_num(vals, nan=np.inf)
                sorted_indices = np.lexsort(vals.T[::-1])
            elif sort_by in ["mds", "distance_to_most_frequent"]:
                # Fallback to lexicographic for complex sorting methods
                print(f"Warning: {sort_by} sorting simplified to lexicographic for grouped plots with sequence selection")
                vals = group_sequences.astype(float, copy=True)
                vals = np.nan_to_num(vals, nan=np.inf)
                sorted_indices = np.lexsort(vals.T[::-1])
            else:
                # unsorted or other methods
                sorted_indices = np.arange(len(group_sequences))

        sorted_data = group_sequences[sorted_indices]
        
        # Track sorted IDs for y-axis labels if needed
        sorted_group_ids = None
        if group_ids_for_labels is not None and show_sequence_ids:
            sorted_group_ids = group_ids_for_labels[sorted_indices]
        
        # Store sorted IDs for this group if return_sorted_ids is True
        if return_sorted_ids:
            sorted_ids_by_group[group] = group_ids_after_selection[sorted_indices]
            # Track group order (only add once per group)
            if group not in group_order_list:
                group_order_list.append(group)

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
        
        # Determine tick positions and labels
        if show_sequence_ids and sorted_group_ids is not None:
            # Show sequence IDs instead of sequence numbers
            # For large number of sequences, show fewer ticks to avoid overcrowding
            if num_sequences <= 10:
                ytick_positions = np.arange(num_sequences)
                ytick_labels = [str(sid) for sid in sorted_group_ids]
            else:
                # Show subset of IDs for readability
                if plot_style == "narrow":
                    num_ticks = min(8, num_sequences)
                else:
                    num_ticks = min(11, num_sequences)
                ytick_positions = np.linspace(0, num_sequences - 1, num=num_ticks, dtype=int)
                ytick_positions = np.unique(ytick_positions)
                ytick_labels = [str(sorted_group_ids[pos]) for pos in ytick_positions]
        else:
            # Default behavior: show sequence numbers
            if plot_style == "narrow":
                num_ticks = min(8, num_sequences)  # Fewer ticks for narrow plots
            else:
                num_ticks = min(11, num_sequences)
            ytick_positions = np.linspace(0, num_sequences - 1, num=num_ticks, dtype=int)
            ytick_positions = np.unique(ytick_positions)
            ytick_labels = (ytick_positions + 1).astype(int)
        
        # Hide y-axis if requested
        if hide_y_axis:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
        else:
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels(ytick_labels, fontsize=fontsize-2, color='black')

        # Customize axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if not hide_y_axis:
            ax.spines['left'].set_color('gray')
            ax.spines['left'].set_linewidth(0.7)
            ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_color('gray')
        ax.spines['bottom'].set_linewidth(0.7)
        
        # Move spines slightly away from the plot area for better aesthetics
        ax.spines['bottom'].set_position(('outward', 5))
        
        # Ensure ticks are always visible regardless of plot style
        ax.tick_params(axis='x', colors='gray', length=4, width=0.7, which='major')
        if not hide_y_axis:
            ax.tick_params(axis='y', colors='gray', length=4, width=0.7, which='major')
        
        # Force tick visibility for narrow plot styles
        ax.xaxis.set_ticks_position('bottom')
        if not hide_y_axis:
            ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='both', which='major', direction='out')

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
        if i % ncols == 0 and not hide_y_axis:
            ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10, color='black')

        # For column layout, only show x-axis label on the bottom subplot
        # For grid layout, show x-axis label on bottom row subplots
        if layout == 'column':
            # Only show xlabel on the last (bottom) subplot
            if i == num_groups - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color='black')
        else:
            # For grid layout, show xlabel on bottom row
            if i >= num_groups - ncols:
                ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color='black')

    # Hide unused subplots (not needed for proportional scaling with column layout)
    if not (proportional_scaling and layout == 'column'):
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    # Add a common title if provided and show_title is True
    if title and show_title:
        fig.suptitle(title, fontsize=fontsize+2, y=1.02)

    # Adjust layout to remove tight_layout warning and eliminate extra right space
    # Increase hspace for column layout to prevent x-axis labels from overlapping with subplot above
    if proportional_scaling and layout == 'column':
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.9, wspace=0.15, hspace=0.4)
    else:
        hspace_value = 0.4 if layout == 'column' else 0.25
        fig.subplots_adjust(wspace=0.15, hspace=hspace_value, bottom=0.1, top=0.9, right=0.98, left=0.08)

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
        plt.show()
        plt.close()
    
    # Return sorted IDs if requested
    # Return as dictionary with group order preserved (groups processed in plot order)
    if return_sorted_ids:
        # Create an ordered dictionary or return the dictionary with group_order_list for reference
        # For simplicity, return the dictionary (Python 3.7+ maintains insertion order)
        # But we'll also return group_order_list as metadata if needed
        return sorted_ids_by_group


def _sequence_index_plot_single(seqdata: SequenceData,
                                sort_by="unsorted",
                                sort_by_weight=False,
                                weights="auto",
                                figsize=(10, 6),
                                plot_style="standard",
                                title=None,
                                xlabel="Time",
                                ylabel="Sequences",
                                save_as=None,
                                dpi=200,
                                fontsize=12,
                                include_legend=True,
                                sequence_selection="all",
                                n_sequences=10,
                                show_sequence_ids=False,
                                sort_by_ids=None,
                                return_sorted_ids=False,
                                show_title=True):
    """Efficiently creates a sequence index plot using `imshow` for faster rendering.

    :param seqdata: SequenceData object containing sequence information
    :param sort_by: Sorting method ('unsorted', 'lexicographic', 'mds', 'distance_to_most_frequent')
    :param sort_by_weight: If True, sort sequences by weight (descending), overrides sort_by
    :param weights: (np.ndarray or "auto") Weights for sequences. If "auto", uses seqdata.weights if available
    :param figsize: (tuple): Size of the figure (only used when plot_style="custom").
    :param plot_style: Plot aspect style ('standard', 'compact', 'wide', 'narrow', 'custom')
    :param title: (str): Title for the plot.
    :param xlabel: (str): Label for the x-axis.
    :param ylabel: (str): Label for the y-axis.
    :param save_as: File path to save the plot
    :param dpi: DPI for saved image
    :param include_legend: Whether to include legend in the plot (True by default)
    :param sequence_selection: Method for selecting sequences ("all", "first_n", "last_n", or list of IDs)
    :param n_sequences: Number of sequences for "first_n" or "last_n"
    :param show_sequence_ids: If True, show actual sequence IDs on y-axis instead of sequence numbers

    :return None.
    """
    # Determine figure size based on plot style
    style_sizes = {
        'standard': (10, 6),   # Balanced view
        'compact': (8, 8),     # More square, like R plots  
        'wide': (12, 4),       # Wide, emphasizes time
        'narrow': (8, 10),     # Moderately vertical
        'custom': figsize      # User-provided
    }
    
    if plot_style not in style_sizes:
        raise ValueError(f"Invalid plot_style '{plot_style}'. "
                        f"Supported styles: {list(style_sizes.keys())}")
    
    # Special validation for custom plot style
    if plot_style == 'custom' and figsize == (10, 6):
        raise ValueError(
            "When using plot_style='custom', you must explicitly provide a figsize parameter "
            "that differs from the default (10, 6). "
            "Suggested custom sizes:\n"
            "  - For wide plots: figsize=(15, 5)\n"
            "  - For tall plots: figsize=(7, 12)\n"
            "  - For square plots: figsize=(9, 9)\n"
            "  - For small plots: figsize=(6, 4)\n"
            "Example: plot_sequence_index(data, plot_style='custom', figsize=(12, 8))"
        )
    
    actual_figsize = style_sizes[plot_style]
    
    # Process weights
    if isinstance(weights, str) and weights == "auto":
        weights = getattr(seqdata, "weights", None)
    
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(weights) != len(seqdata.values):
            raise ValueError("Length of weights must equal number of sequences.")
    
    # Apply sequence selection and get the filtered data directly
    selection_mask = _select_sequences_subset(seqdata, sequence_selection, n_sequences, sort_by, sort_by_weight, weights)
    
    # Get sequence values as NumPy array (apply selection if needed)
    selected_ids = None  # Track selected IDs for y-axis labels
    if not np.all(selection_mask):
        sequence_values = seqdata.values[selection_mask].copy()
        # Track selected IDs for y-axis display
        if hasattr(seqdata, 'ids') and seqdata.ids is not None:
            selected_ids = seqdata.ids[selection_mask]
        # Update weights if provided
        if weights is not None:
            weights = weights[selection_mask]
    else:
        sequence_values = seqdata.values.copy()
        # All IDs are selected
        if hasattr(seqdata, 'ids') and seqdata.ids is not None:
            selected_ids = seqdata.ids

    # Handle NaN values for better visualization
    if np.isnan(sequence_values).any():
        # Keep NaN as float for proper masking
        sequence_values = sequence_values.astype(float)

    # Sort sequences based on specified method
    # sort_by_ids takes priority if provided
    if sort_by_ids is not None:
        # Sort by custom ID order
        # Convert sort_by_ids to numpy array for easier handling
        sort_by_ids_array = np.asarray(sort_by_ids)
        
        # Create a mapping from ID to position in sort_by_ids
        # IDs not in sort_by_ids will be placed at the end
        id_to_position = {id_val: pos for pos, id_val in enumerate(sort_by_ids_array)}
        
        # Get positions for each ID in the selected sequences
        # IDs not in sort_by_ids get a very large position value (placed at end)
        max_position = len(sort_by_ids_array)
        positions = np.array([id_to_position.get(id_val, max_position + i) 
                             for i, id_val in enumerate(selected_ids)])
        
        # Sort by position (ascending order)
        sorted_indices = np.argsort(positions)
        
        # Warn if some IDs are not in sort_by_ids
        missing_ids = set(selected_ids) - set(sort_by_ids_array)
        if missing_ids:
            print(f"[Warning] {len(missing_ids)} IDs not found in sort_by_ids, "
                  f"they will be placed at the end: {list(missing_ids)[:5]}{'...' if len(missing_ids) > 5 else ''}")
    
    elif sort_by_weight and weights is not None:
        # Sort by weight (descending)
        sorted_indices = np.argsort(-weights)
    else:
        # Use simpler sorting for the filtered data
        if sort_by == "lexicographic":
            vals = sequence_values.astype(float, copy=True)
            vals = np.nan_to_num(vals, nan=np.inf)
            sorted_indices = np.lexsort(vals.T[::-1])
        elif sort_by in ["mds", "distance_to_most_frequent"]:
            # Fallback to lexicographic for complex sorting methods
            print(f"Warning: {sort_by} sorting simplified to lexicographic for sequence selection")
            vals = sequence_values.astype(float, copy=True)
            vals = np.nan_to_num(vals, nan=np.inf)
            sorted_indices = np.lexsort(vals.T[::-1])
        else:
            # unsorted or other methods
            sorted_indices = np.arange(len(sequence_values))
    
    sorted_data = sequence_values[sorted_indices]
    
    # Track sorted IDs for y-axis labels if needed
    sorted_ids = None
    if selected_ids is not None and show_sequence_ids:
        sorted_ids = selected_ids[sorted_indices]

    # Create the plot using imshow with proper NaN handling
    fig, ax = plt.subplots(figsize=actual_figsize)
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
    
    # Determine tick positions and labels
    if show_sequence_ids and sorted_ids is not None:
        # Show sequence IDs instead of sequence numbers
        # For large number of sequences, show fewer ticks to avoid overcrowding
        if num_sequences <= 10:
            ytick_positions = np.arange(num_sequences)
            ytick_labels = [str(sid) for sid in sorted_ids]
        else:
            # Show subset of IDs for readability
            if plot_style == "narrow":
                num_ticks = min(8, num_sequences)
            else:
                num_ticks = min(11, num_sequences)
            ytick_positions = np.linspace(0, num_sequences - 1, num=num_ticks, dtype=int)
            ytick_positions = np.unique(ytick_positions)
            ytick_labels = [str(sorted_ids[pos]) for pos in ytick_positions]
    else:
        # Default behavior: show sequence numbers
        if plot_style == "narrow":
            num_ticks = min(8, num_sequences)  # Fewer ticks for narrow plots
        else:
            num_ticks = min(11, num_sequences)
        ytick_positions = np.linspace(0, num_sequences - 1, num=num_ticks, dtype=int)
        ytick_positions = np.unique(ytick_positions)
        ytick_labels = (ytick_positions + 1).astype(int)
    
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=fontsize-2, color='black')


    # Customize axis line styles and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    
    # Move spines slightly away from the plot area for better aesthetics
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    
    # Ensure ticks are always visible regardless of plot style
    ax.tick_params(axis='x', colors='gray', length=4, width=0.7, which='major')
    ax.tick_params(axis='y', colors='gray', length=4, width=0.7, which='major')
    
    # Force tick visibility for narrow plot styles
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', direction='out')

    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10, color='black')
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10, color='black')
    
    # Set title with weight information if available and show_title is True
    if title is not None and show_title:
        display_title = title
        
        # Check if we have effective weights (not all 1.0) and they were provided by user
        original_weights = getattr(seqdata, "weights", None)
        if original_weights is not None and not np.allclose(original_weights, 1.0) and weights is not None:
            sum_w = float(weights.sum())
            display_title += f" (n = {num_sequences}, total weight = {sum_w:.1f})"
        else:
            display_title += f" (n = {num_sequences})"
        
        ax.set_title(display_title, fontsize=fontsize+2, color='black')

    # Use legend from SequenceData if requested
    if include_legend:
        ax.legend(*seqdata.get_legend(), bbox_to_anchor=(1.05, 1), loc='upper left')

    save_and_show_results(save_as, dpi=dpi)
    
    # Return sorted IDs if requested
    if return_sorted_ids:
        return selected_ids[sorted_indices] if selected_ids is not None else None
