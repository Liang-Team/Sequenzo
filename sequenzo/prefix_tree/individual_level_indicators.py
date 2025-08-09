"""
@Author  : Yuqi Liang 梁彧祺
@File    : individual_level_indicators.py
@Time    : 02/05/2025 11:07
@Desc    : 
    This module provides methods for calculating individual-level indicators in sequence data analysis.
    It includes tools to assess sequence divergence, identify divergence timing, measure prefix rarity,
    and evaluate path uniqueness for individuals or groups. 
    
    These indicators help quantify how typical or unique an individual's sequence is within a population, 
    and can be used for both overall and subgroup analyses.
"""
from collections import defaultdict, Counter
import numpy as np
import math
import pandas as pd


class IndividualDivergence:
    def __init__(self, sequences):
        self.sequences = sequences
        self.T = len(sequences[0])
        self.prefix_freq_by_year = self._build_prefix_frequencies()

    def _build_prefix_frequencies(self):
        freq_by_year = [defaultdict(int) for _ in range(self.T)]
        for seq in self.sequences:
            prefix = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq_by_year[t][tuple(prefix)] += 1
        return freq_by_year

    def compute_diverged(self, z_threshold=1.5, min_t=3, window=1):
        """
        Compute binary diverged status based on rarity score z-scores.

        :param z_threshold: Z-score threshold above which an individual is considered diverged.
        :param min_t: Minimum year (1-indexed) after which divergence is considered valid.
        :param window: Number of consecutive high-z years required (default: 1).
        :return: List of 0/1 flags indicating whether each individual diverged.
        """
        N = len(self.sequences)
        rarity_matrix = []

        for seq in self.sequences:
            prefix = []
            score = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        rarity_df = pd.DataFrame(rarity_matrix)
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        flags = []
        for i in range(N):
            z = rarity_z.iloc[i]
            diverged = 0
            for t in range(min_t - 1, self.T - window + 1):
                if all(z[t + k] > z_threshold for k in range(window)):
                    diverged = 1
                    break
            flags.append(diverged)
        return flags

    def compute_first_divergence_year(self, z_threshold=1.5, min_t=3, window=1):
        """
        Compute the first divergence year for each individual based on rarity score z-scores.
        
        Returns the earliest year when an individual's trajectory diverges from the mainstream,
        defined as having z-scores above threshold for consecutive years.

        Note on Zero Variance Years:
        When standard deviation of rarity scores approaches zero within a given year, 
        z-scores become undefined (NaN), indicating absence of divergence from mainstream. 
        This is conceptually appropriate as it reflects periods where all individuals 
        follow similar trajectories, consistent with strong institutional constraints 
        or normative expectations at specific life course stages.

        Parameters:
        -----------
        z_threshold : float, default=1.5
            Z-score threshold for defining divergence from mainstream
        min_t : int, default=3
            Minimum year (1-indexed) considered valid for divergence detection
        window : int, default=1
            Number of consecutive high-z years required to confirm divergence
            
        Returns:
        --------
        List[Optional[int]]
            List of first divergence years (1-indexed) for each individual.
            None indicates no divergence detected for that individual.
        """
        N = len(self.sequences)
        rarity_matrix = []

        for seq in self.sequences:
            prefix = []
            score = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        rarity_df = pd.DataFrame(rarity_matrix)
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        years = []
        for i in range(N):
            z = rarity_z.iloc[i]
            year = None
            for t in range(min_t - 1, self.T - window + 1):
                if all(z[t + k] > z_threshold for k in range(window)):
                    year = t + 1
                    break
            years.append(year)
        return years

    def compute_prefix_rarity_score(self):
        rarity_scores = []
        N = len(self.sequences)

        for seq in self.sequences:
            prefix = []
            score = 0.0
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score += -math.log(freq + 1e-10)  # small constant to avoid log(0)
            rarity_scores.append(score)
        return rarity_scores

    def diagnose_divergence_calculation(self, z_threshold=1.5, min_t=3, window=1):
        """
        Diagnostic function to analyze divergence year calculation and identify 
        years with insufficient variance (std ≈ 0) that cannot trigger divergence.
        
        This is methodologically appropriate: when all individuals follow similar 
        trajectories in a given year, no divergence should be detected.
        
        Returns:
        --------
        dict: Diagnostic information including:
            - years_with_zero_variance: List of years where std ≈ 0
            - rarity_std_by_year: Standard deviation of rarity scores per year
            - n_individuals_with_divergence: Count of individuals with any divergence
            - divergence_year_distribution: Value counts of divergence years
        """
        N = len(self.sequences)
        rarity_matrix = []

        # Calculate rarity scores (same as in compute_divergence_year)
        for seq in self.sequences:
            prefix = []
            score = []
            for t in range(self.T):
                prefix.append(seq[t])
                freq = self.prefix_freq_by_year[t][tuple(prefix)] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        rarity_df = pd.DataFrame(rarity_matrix)
        
        # Calculate standard deviations by year
        rarity_std_by_year = rarity_df.std(axis=0)
        years_with_zero_variance = []
        
        # Identify years with near-zero variance (threshold can be adjusted)
        for t, std_val in enumerate(rarity_std_by_year):
            if pd.isna(std_val) or std_val < 1e-10:
                years_with_zero_variance.append(t + 1)  # 1-indexed
        
        # Calculate z-scores
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        
        # Count individuals with divergence
        divergence_years = self.compute_first_divergence_year(z_threshold, min_t, window)
        n_individuals_with_divergence = sum(1 for year in divergence_years if year is not None)
        
        # Distribution of divergence years
        divergence_year_counts = pd.Series(divergence_years).value_counts(dropna=False).sort_index()
        
        return {
            'years_with_zero_variance': years_with_zero_variance,
            'rarity_std_by_year': rarity_std_by_year.tolist(),
            'n_individuals_with_divergence': n_individuals_with_divergence,
            'divergence_year_distribution': divergence_year_counts.to_dict(),
            'total_individuals': N,
            'parameters_used': {
                'z_threshold': z_threshold,
                'min_t': min_t, 
                'window': window
            }
        }

    def compute_path_uniqueness(self):
        uniqueness_scores = []
        for seq in self.sequences:
            prefix = []
            count = 0
            for t in range(self.T):
                prefix.append(seq[t])
                if self.prefix_freq_by_year[t][tuple(prefix)] == 1:
                    count += 1
            uniqueness_scores.append(count)
        return uniqueness_scores


def plot_prefix_rarity_distribution(
    data,
    group_names=None,
    show_threshold=True,
    z_threshold=1.5,
    threshold_label=None,
    colors=None,
    figsize=(10, 6),
    save_as=None,
    dpi=300,
    show=True
):
    """
    Plot prefix rarity score distribution(s) with optional z-score threshold line.
    
    Parameters:
    -----------
    data : dict or list or array-like
        If dict: {"group1": [scores], "group2": [scores], ...} for multi-group comparison
        If list/array: single group scores
    group_names : list, optional
        Custom names for groups. If None and data is dict, uses keys.
        If None and data is list/array, uses "Group"
    show_threshold : bool, default=True
        Whether to show the z-score threshold line
    z_threshold : float, default=1.5
        Z-score threshold value for the vertical line
    threshold_label : str, optional
        Custom label for threshold line. If None, uses "z = {z_threshold}"
    colors : list or dict, optional
        Colors for each group. If None, uses default palette
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    save_as : str, optional
        Path to save the figure (without extension)
    dpi : int, default=300
        DPI for saving
    show : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    dict: Statistics including threshold value in original scale (if show_threshold=True)
    
    Example:
    --------
    # Single group
    >>> plot_prefix_rarity_distribution(india_scores)
    
    # Multi-group comparison
    >>> data = {"India": india_scores, "US": us_scores}
    >>> plot_prefix_rarity_distribution(
    ...     data, 
    ...     show_threshold=True,
    ...     z_threshold=1.5,
    ...     save_as="rarity_comparison"
    ... )
    
    # Custom colors and no threshold
    >>> plot_prefix_rarity_distribution(
    ...     data,
    ...     colors={"India": "#E8B88A", "US": "#A3BFD9"},
    ...     show_threshold=False
    ... )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Process input data
    if isinstance(data, dict):
        # Multi-group case
        groups = data
        if group_names is None:
            group_names = list(groups.keys())
    else:
        # Single group case
        if group_names is None:
            group_names = ["Group"]
        groups = {group_names[0]: data}
    
    # Set up colors
    if colors is None:
        default_colors = ["#E8B88A", "#A3BFD9", "#C6A5CF", "#A6C1A9", "#F4A460", "#87CEEB"]
        if isinstance(colors, dict):
            color_map = colors
        else:
            color_map = dict(zip(group_names, default_colors[:len(group_names)]))
    elif isinstance(colors, dict):
        color_map = colors
    else:
        color_map = dict(zip(group_names, colors))
    
    # Calculate threshold if needed
    stats = {}
    if show_threshold:
        # Combine all data to calculate overall mean and std
        all_scores = []
        for scores in groups.values():
            all_scores.extend(scores)
        all_scores = np.array(all_scores)
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        x_thresh = mean_score + z_threshold * std_score
        
        stats = {
            'mean': mean_score,
            'std': std_score,
            'threshold_value': x_thresh,
            'z_threshold': z_threshold
        }
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot distributions
    for group_name in group_names:
        if group_name in groups:
            scores = groups[group_name]
            color = color_map.get(group_name, "#1f77b4")
            sns.kdeplot(scores, label=group_name, fill=True, color=color, linewidth=2)
    
    # Add threshold line if requested
    if show_threshold:
        plt.axvline(x_thresh, color="grey", linestyle="--", linewidth=1.5)
        
        # Dynamic text positioning
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        text_y = y_max * 0.85
        
        # Custom or default threshold label
        if threshold_label is None:
            threshold_label = f"z = {z_threshold}"
        
        plt.text(x_thresh + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02, 
                text_y, threshold_label, color="grey", fontsize=11)
    
    # Formatting
    plt.xlabel("Prefix Rarity Score", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    if len(group_names) > 1:
        plt.legend(title="Group")
    sns.despine()
    plt.tight_layout()
    
    # Save and show
    if save_as:
        plt.savefig(f"{save_as}.png", dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return stats


def plot_individual_indicators_correlation(
    df,
    indicator_columns=None,
    correlation_method='pearson',
    group_column=None,
    figsize=(10, 8),
    cmap='RdBu_r',
    center=0,
    annot=True,
    fmt='.2f',
    save_as=None,
    dpi=300,
    show=True
):
    """
    Plot correlation heatmap of individual-level indicators with beautiful styling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing individual-level indicators
    indicator_columns : list, optional
        List of column names to include in correlation analysis.
        If None, automatically detects indicator columns (diverged, divergence_year, 
        prefix_rarity_score, path_uniqueness, etc.)
    correlation_method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', 'kendall'
    group_column : str, optional
        Column name for grouping (e.g., 'country'). If provided, shows separate 
        heatmaps for each group
    figsize : tuple, default=(10, 8)
        Figure size (width, height)
    cmap : str, default='RdBu_r'
        Colormap for heatmap. Options: 'RdBu_r', 'coolwarm', 'viridis', 'plasma'
    center : float, default=0
        Value to center the colormap at
    annot : bool, default=True
        Whether to annotate cells with correlation values
    fmt : str, default='.2f'
        Format for annotations
    save_as : str, optional
        Path to save the figure (without extension)
    dpi : int, default=300
        DPI for saving
    show : bool, default=True
        Whether to display the plot
        
    Returns:
    --------
    dict: Correlation matrix/matrices and statistics
    
    Example:
    --------
    # Basic usage
    >>> plot_individual_indicators_correlation(df)
    
    # Custom indicators with grouping
    >>> plot_individual_indicators_correlation(
    ...     df, 
    ...     indicator_columns=['diverged', 'prefix_rarity_score', 'path_uniqueness'],
    ...     group_column='country',
    ...     correlation_method='spearman'
    ... )
    
    # Custom styling
    >>> plot_individual_indicators_correlation(
    ...     df,
    ...     cmap='plasma',
    ...     figsize=(12, 10),
    ...     save_as="indicators_correlation_heatmap"
    ... )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Auto-detect indicator columns if not provided
    if indicator_columns is None:
        # Common individual-level indicator patterns
        potential_indicators = [
            'diverged', 'first_divergence_year', 'divergence_year',
            'prefix_rarity_score', 'path_uniqueness',
            'rarity_score', 'uniqueness_score'
        ]
        indicator_columns = [col for col in df.columns if col in potential_indicators]
        
        # Also include numeric columns that might be indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in indicator_columns and any(
                keyword in col.lower() for keyword in 
                ['score', 'index', 'count', 'factor', 'rate', 'ratio']
            ):
                indicator_columns.append(col)
    
    # Filter and clean data
    df_indicators = df[indicator_columns].copy()
    
    # Handle missing values and convert data types
    for col in df_indicators.columns:
        if df_indicators[col].dtype == 'object':
            # Try to convert to numeric
            df_indicators[col] = pd.to_numeric(df_indicators[col], errors='coerce')
    
    # Remove columns with too many missing values (>50%)
    valid_cols = []
    for col in df_indicators.columns:
        if df_indicators[col].notna().sum() / len(df_indicators) > 0.5:
            valid_cols.append(col)
    
    df_indicators = df_indicators[valid_cols]
    
    # Drop rows with any missing values for correlation calculation
    df_clean = df_indicators.dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data remaining after cleaning. Check for missing values.")
    
    # Calculate correlations
    results = {}
    
    if group_column is None or group_column not in df.columns:
        # Single correlation matrix
        corr_matrix = df_clean.corr(method=correlation_method)
        results['overall'] = corr_matrix
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle (optional - makes it cleaner)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            center=center,
            square=True,
            cbar_kws={"shrink": .8, "label": f"{correlation_method.title()} Correlation"},
            linewidths=0.5
        )
        
        plt.title(f"Individual-Level Indicators Correlation Heatmap\n({correlation_method.title()} Correlation)", 
                 fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
    else:
        # Group-based correlation matrices
        groups = df[group_column].unique()
        n_groups = len(groups)
        
        # Calculate subplot layout
        if n_groups <= 2:
            nrows, ncols = 1, n_groups
            figsize = (figsize[0] * n_groups, figsize[1])
        else:
            ncols = min(3, n_groups)
            nrows = (n_groups + ncols - 1) // ncols
            figsize = (figsize[0] * ncols, figsize[1] * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, group in enumerate(groups):
            group_data = df[df[group_column] == group][indicator_columns].dropna()
            
            if len(group_data) < 2:
                print(f"Warning: Group '{group}' has insufficient data for correlation")
                continue
                
            corr_matrix = group_data.corr(method=correlation_method)
            results[group] = corr_matrix
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Plot heatmap
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                center=center,
                square=True,
                cbar=i == 0,  # Only show colorbar for first subplot
                cbar_kws={"shrink": .8, "label": f"{correlation_method.title()} Correlation"} if i == 0 else {},
                linewidths=0.5,
                ax=axes[i]
            )
            
            axes[i].set_title(f"{group}\n({len(group_data)} individuals)", fontsize=12)
            axes[i].set_xticks(axes[i].get_xticks())
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            axes[i].set_yticks(axes[i].get_yticks())
            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(f"Individual-Level Indicators Correlation by {group_column.title()}\n({correlation_method.title()} Correlation)", 
                    fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save and show
    if save_as:
        plt.savefig(f"{save_as}.png", dpi=dpi, bbox_inches='tight')
    
    if show:
        plt.show()
    
    # Add summary statistics
    results['summary'] = {
        'method': correlation_method,
        'n_indicators': len(valid_cols),
        'indicators_included': valid_cols,
        'sample_size': len(df_clean) if group_column is None else {group: len(df[df[group_column] == group].dropna()) for group in df[group_column].unique()}
    }
    
    return results


def compute_path_uniqueness_by_group(sequences, group_labels):
        """
        Compute path uniqueness within each subgroup defined by group_labels.
        :param sequences: List of sequences.
        :param group_labels: List of group keys (same length as sequences), e.g., country, gender.
        :return: List of path uniqueness scores (same order as input).
        """
        from collections import defaultdict
        import math

        T = len(sequences[0])
        df = pd.DataFrame({
            "sequence": sequences,
            "group": group_labels
        })

        # Step 1: Precompute prefix frequency tables per group
        group_prefix_freq = {}
        for group, group_df in df.groupby("group"):
            prefix_freq = [defaultdict(int) for _ in range(T)]
            for seq in group_df["sequence"]:
                prefix = []
                for t in range(T):
                    prefix.append(seq[t])
                    prefix_freq[t][tuple(prefix)] += 1
            group_prefix_freq[group] = prefix_freq

        # Step 2: Compute path uniqueness per individual
        uniqueness_scores = []
        for seq, group in zip(sequences, group_labels):
            prefix_freq = group_prefix_freq[group]
            prefix = []
            count = 0
            for t in range(T):
                prefix.append(seq[t])
                if prefix_freq[t][tuple(prefix)] == 1:
                    count += 1
            uniqueness_scores.append(count)

        return uniqueness_scores
