"""
@Author  : Yuqi Liang 梁彧祺
@File    : individual_level_indicators.py
@Time    : 08/08/2025 15:30
@Desc    : 
    This module provides methods for calculating individual-level convergence indicators 
    in sequence data analysis. It includes tools to assess convergence, identify timing, 
    measure suffix rarity, and evaluate path uniqueness.

    The convergence indicators capture whether, when, and to what extent a person's trajectory 
    aligns with dominant population patterns over time.

    Key indicators:
    - Suffix Rarity Score: cumulative rarity of path suffixes (positive, higher = rarer)
    - Binary converged indicator: low rarity z-scores indicate convergence to typical patterns
    - First convergence year: timing when trajectory becomes more typical (low rarity)
    - Path uniqueness for extreme structural isolation
"""
from collections import defaultdict
import numpy as np
import pandas as pd


class IndividualConvergence:
    def __init__(self, sequences):
        # Handle case where sequences might already be an IndividualConvergence object
        if isinstance(sequences, IndividualConvergence):
            # Extract sequences from existing object
            self.sequences = sequences.sequences
        elif hasattr(sequences, 'sequences'):
            # Handle case where input might be another object with sequences attribute
            self.sequences = sequences.sequences
        else:
            # Normal case: sequences is a list of sequences
            self.sequences = sequences
        
        # Validate input
        if not self.sequences or len(self.sequences) == 0:
            raise ValueError("sequences cannot be empty")
        if not hasattr(self.sequences[0], '__len__') and not hasattr(self.sequences[0], '__iter__'):
            raise ValueError("sequences must be a list of sequences (e.g., [[1,2,3], [2,3,1], ...])")
        
        # 验证所有序列长度相同，防止不规整序列的静默错误
        L0 = len(self.sequences[0])
        if any(len(s) != L0 for s in self.sequences):
            raise ValueError("All sequences must have the same length")
        self.T = L0
        
        self.suffix_freq_by_year = self._build_suffix_frequencies()

    def _build_suffix_frequencies(self):
        """
        Build suffix frequencies for each year t.
        suffix[t] contains frequency of suffixes from year t to end for all individuals.
        """
        freq_by_year = [defaultdict(int) for _ in range(self.T)]
        for seq in self.sequences:
            for t in range(self.T):
                suffix = tuple(seq[t:])  # suffix from year t to end
                freq_by_year[t][suffix] += 1
        return freq_by_year

    # Divergence-related computations are intentionally omitted in this convergence-focused module.

    def compute_converged(self, z_threshold=1.5, min_t=1, max_t=None, window=1, inclusive=False, group_labels=None):
        """
        Compute binary converged status based on suffix rarity score z-scores.
        
        Convergence is defined as low rarity (more typical) sustained over time.
        An individual converges when their rarity z-scores fall below -z_threshold
        for consecutive years, indicating movement toward more common patterns.

        :param z_threshold: Z-score threshold below which (as -z_threshold) an individual is considered converged.
        :param min_t: Minimum year (1-indexed) after which convergence is considered valid (default: 1).
        :param max_t: Maximum year (1-indexed) before which convergence is considered valid. 
                      If None, uses T-window+1.
        :param window: Number of consecutive low-rarity-z years required (default: 1).
        :param inclusive: If True, uses <= comparison; if False, uses < comparison (default: False).
        :param group_labels: Optional list of group labels (same length as sequences) for within-group convergence calculation.
        :return: List of 0/1 flags indicating whether each individual converged.
        """
        if max_t is None:
            max_t = self.T - window + 1

        N = len(self.sequences)
        
        if group_labels is not None:
            # 组内收敛：使用组内频率和样本大小
            return self._compute_converged_by_group(z_threshold, min_t, max_t, window, inclusive, group_labels)
        
        # 使用全局频率计算稀有度
        rarity_matrix = []
        for seq in self.sequences:
            score = []
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        rarity_df = pd.DataFrame(rarity_matrix)
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        # 处理零方差年份：NaN 会使比较失败，显式设为无穷大以确保不满足收敛条件
        rarity_z = rarity_z.replace([np.inf, -np.inf], np.nan).fillna(np.inf)

        flags = []
        for i in range(N):
            z = rarity_z.iloc[i]
            converged = 0
            for t in range(min_t - 1, max_t):  # min_t-1 for 0-indexed, max_t already accounts for window
                # 收敛 = 低稀有（更典型）
                if inclusive:
                    condition = all(z[t + k] <= -z_threshold for k in range(window))
                else:
                    condition = all(z[t + k] < -z_threshold for k in range(window))
                
                if condition:
                    converged = 1
                    break
            flags.append(converged)
        return flags
    
    def _compute_converged_by_group(self, z_threshold, min_t, max_t, window, inclusive, group_labels):
        """
        计算组内收敛：使用组内频率和样本大小计算稀有度
        """
        from collections import defaultdict
        
        # 按组构建 suffix 频率表
        group_suffix_freq = {}
        group_sizes = {}
        
        # 先统计各组信息
        group_sequences = defaultdict(list)
        for i, (seq, group) in enumerate(zip(self.sequences, group_labels)):
            group_sequences[group].append((i, seq))
        
        # 为每个组构建 suffix 频率表
        for group, seq_list in group_sequences.items():
            group_sizes[group] = len(seq_list)
            freq_by_year = [defaultdict(int) for _ in range(self.T)]
            
            for _, seq in seq_list:
                for t in range(self.T):
                    suffix = tuple(seq[t:])
                    freq_by_year[t][suffix] += 1
            
            group_suffix_freq[group] = freq_by_year
        
        # 为每个个体计算组内稀有度
        all_flags = [0] * len(self.sequences)
        
        for group, seq_list in group_sequences.items():
            group_n = group_sizes[group]
            group_freq = group_suffix_freq[group]
            
            # 计算该组的稀有度矩阵
            rarity_matrix = []
            group_indices = []
            
            for orig_idx, seq in seq_list:
                group_indices.append(orig_idx)
                score = []
                for t in range(self.T):
                    suffix = tuple(seq[t:])
                    freq = group_freq[t][suffix] / group_n
                    score.append(-np.log(freq + 1e-10))
                rarity_matrix.append(score)
            
            # 计算 z 分数
            rarity_df = pd.DataFrame(rarity_matrix)
            rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
            rarity_z = rarity_z.replace([np.inf, -np.inf], np.nan).fillna(np.inf)
            
            # 判断收敛
            for i, orig_idx in enumerate(group_indices):
                z = rarity_z.iloc[i]
                converged = 0
                for t in range(min_t - 1, max_t):
                    if inclusive:
                        condition = all(z[t + k] <= -z_threshold for k in range(window))
                    else:
                        condition = all(z[t + k] < -z_threshold for k in range(window))
                    
                    if condition:
                        converged = 1
                        break
                
                all_flags[orig_idx] = converged
        
        return all_flags

    # First-divergence timing is intentionally omitted in this convergence-focused module.

    def compute_first_convergence_year(self, z_threshold=1.5, min_t=1, max_t=None, window=1, inclusive=False, group_labels=None):
        """
        Compute the first convergence year for each individual based on suffix rarity score z-scores.
        
        Returns the earliest year when an individual's trajectory converges to the mainstream,
        defined as having low rarity z-scores (below -threshold) for consecutive years,
        indicating movement toward more typical patterns.

        Parameters:
        -----------
        z_threshold : float, default=1.5
            Z-score threshold for defining convergence (convergence when z < -z_threshold)
        min_t : int, default=1
            Minimum year (1-indexed) after which convergence is considered valid.
        max_t : int, optional
            Maximum year (1-indexed) considered valid for convergence detection.
            If None, uses T-window+1.
        window : int, default=1
            Number of consecutive low-rarity-z years required to confirm convergence
        inclusive : bool, default=False
            If True, uses <= comparison; if False, uses < comparison
        group_labels : list, optional
            List of group labels (same length as sequences) for within-group convergence calculation
            
        Returns:
        --------
        List[Optional[int]]
            List of first convergence years (1-indexed) for each individual.
            None indicates no convergence detected for that individual.
        """
        if max_t is None:
            max_t = self.T - window + 1

        N = len(self.sequences)
        
        if group_labels is not None:
            # 组内收敛：使用组内频率和样本大小
            return self._compute_first_convergence_year_by_group(z_threshold, min_t, max_t, window, inclusive, group_labels)
        
        # 使用全局频率计算稀有度
        rarity_matrix = []
        for seq in self.sequences:
            score = []
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        rarity_df = pd.DataFrame(rarity_matrix)
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        # 处理零方差年份：NaN 会使比较失败，显式设为无穷大以确保不满足收敛条件
        rarity_z = rarity_z.replace([np.inf, -np.inf], np.nan).fillna(np.inf)

        years = []
        for i in range(N):
            z = rarity_z.iloc[i]
            year = None
            for t in range(min_t - 1, max_t):  # min_t-1 for 0-indexed, max_t already accounts for window
                # 收敛 = 低稀有（更典型）
                if inclusive:
                    condition = all(z[t + k] <= -z_threshold for k in range(window))
                else:
                    condition = all(z[t + k] < -z_threshold for k in range(window))
                
                if condition:
                    year = t + 1  # Convert to 1-indexed
                    break
            years.append(year)
        return years
    
    def _compute_first_convergence_year_by_group(self, z_threshold, min_t, max_t, window, inclusive, group_labels):
        """
        计算组内第一次收敛年份：使用组内频率和样本大小计算稀有度
        """
        from collections import defaultdict
        
        # 按组构建 suffix 频率表（重用 _compute_converged_by_group 的逻辑）
        group_sequences = defaultdict(list)
        for i, (seq, group) in enumerate(zip(self.sequences, group_labels)):
            group_sequences[group].append((i, seq))
        
        # 为每个组构建 suffix 频率表
        group_suffix_freq = {}
        group_sizes = {}
        for group, seq_list in group_sequences.items():
            group_sizes[group] = len(seq_list)
            freq_by_year = [defaultdict(int) for _ in range(self.T)]
            
            for _, seq in seq_list:
                for t in range(self.T):
                    suffix = tuple(seq[t:])
                    freq_by_year[t][suffix] += 1
            
            group_suffix_freq[group] = freq_by_year
        
        # 为每个个体计算组内收敛年份
        all_years = [None] * len(self.sequences)
        
        for group, seq_list in group_sequences.items():
            group_n = group_sizes[group]
            group_freq = group_suffix_freq[group]
            
            # 计算该组的稀有度矩阵
            rarity_matrix = []
            group_indices = []
            
            for orig_idx, seq in seq_list:
                group_indices.append(orig_idx)
                score = []
                for t in range(self.T):
                    suffix = tuple(seq[t:])
                    freq = group_freq[t][suffix] / group_n
                    score.append(-np.log(freq + 1e-10))
                rarity_matrix.append(score)
            
            # 计算 z 分数
            rarity_df = pd.DataFrame(rarity_matrix)
            rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
            rarity_z = rarity_z.replace([np.inf, -np.inf], np.nan).fillna(np.inf)
            
            # 寻找第一次收敛年份
            for i, orig_idx in enumerate(group_indices):
                z = rarity_z.iloc[i]
                year = None
                for t in range(min_t - 1, max_t):
                    if inclusive:
                        condition = all(z[t + k] <= -z_threshold for k in range(window))
                    else:
                        condition = all(z[t + k] < -z_threshold for k in range(window))
                    
                    if condition:
                        year = t + 1
                        break
                
                all_years[orig_idx] = year
        
        return all_years

    def compute_suffix_rarity_per_year(self, as_dataframe: bool = True, column_prefix: str = "t", zscore: bool = False):
        """
        Compute per-year suffix rarity scores for each individual.

        Definition (mirror of prefix rarity):
            rarity_{i,t} = -log( freq(suffix_{i,t}) / N ) >= 0

        Where suffix_{i,t} is the observed suffix from year t to end for person i,
        and N is total number of individuals. Higher means rarer (less typical).

        Parameters
        ----------
        as_dataframe : bool, default True
            If True, returns a pandas DataFrame with columns f"{column_prefix}1"..f"{column_prefix}T".
            If False, returns a NumPy array of shape (N, T).
        column_prefix : str, default "t"
            Column name prefix when returning a DataFrame.
        zscore : bool, default False
            If True, z-standardize the rarity scores column-wise (by year).

        Returns
        -------
        pandas.DataFrame or np.ndarray
            Per-year rarity scores (optionally z-scored).
        """
        N = len(self.sequences)
        rarity_matrix = []

        for seq in self.sequences:
            score_list = []
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
                score_list.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score_list)

        rarity_arr = np.array(rarity_matrix, dtype=float)

        if zscore:
            col_means = np.nanmean(rarity_arr, axis=0)
            col_stds = np.nanstd(rarity_arr, axis=0, ddof=1)  # 与 pandas DataFrame.std() 保持一致
            with np.errstate(invalid='ignore', divide='ignore'):
                rarity_arr = (rarity_arr - col_means) / col_stds

        if not as_dataframe:
            return rarity_arr

        columns = [f"{column_prefix}{t+1}" for t in range(self.T)]
        return pd.DataFrame(rarity_arr, columns=columns)

    def compute_suffix_rarity_score(self):
        """
        Compute cumulative suffix rarity score for each individual:
            rarity_score_i = sum_{t=1}^T -log( freq(suffix_{i,t}) / N )

        Higher scores indicate rarer, less typical future paths from each year onward.
        """
        rarity_scores = []
        N = len(self.sequences)

        for seq in self.sequences:
            score = 0.0
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
                score += -np.log(freq + 1e-10)
            rarity_scores.append(score)
        return rarity_scores

    def compute_standardized_rarity_score(self, min_t=1, max_t=None, window=1):
        """
        Compute standardized rarity scores for convergence classification and visualization.
        
        This method computes standardized rarity scores used for individual-level 
        convergence classification:
        standardized_score_i = min_t max_{k=0..window-1} z_{i,t+k}
        
        Where z_{i,t} are the year-wise standardized suffix rarity scores using column-wise 
        standardization with sample standard deviation (ddof=1, as computed by pandas).
        
        The standardized scores can be used with a threshold (e.g., z ≤ -1.5) to classify 
        individuals as converged/not converged, and are particularly useful for visualization.
        
        Note: For convergence (suffix tree), we look for LOW rarity (more typical patterns),
        so lower z-scores indicate convergence. This is opposite to prefix tree divergence.
        
        Parameters:
        -----------
        min_t : int, default=1
            Minimum year (1-indexed) after which convergence is considered valid.
        max_t : int, optional
            Maximum year (1-indexed) before which convergence is considered valid.
            If None, uses T-window+1.
        window : int, default=1
            Number of consecutive low-z years required
            
        Returns:
        --------
        List[float]
            Standardized rarity scores for each individual. Values ≤ -z_threshold indicate convergence.
            
        Notes:
        ------
        The standardization uses sample standard deviation (ddof=1) for each year column,
        which is consistent with pandas' default behavior for DataFrame.std().
        This is essentially the z-score normalized version of suffix rarity scores.
        For convergence detection, we look for the MINIMUM z-score (most typical behavior).
        """
        if max_t is None:
            max_t = self.T - window + 1
            
        N = len(self.sequences)
        
        # Step 1: Calculate rarity matrix
        rarity_matrix = []
        for seq in self.sequences:
            score = []
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
                score.append(-np.log(freq + 1e-10))
            rarity_matrix.append(score)

        # Step 2: Column-wise standardization (by year)
        rarity_df = pd.DataFrame(rarity_matrix)
        rarity_z = rarity_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        # Handle zero-variance years: NaN would make comparison fail, explicitly set to +inf to ensure not meeting convergence condition
        rarity_z = rarity_z.replace([np.inf, -np.inf], np.nan).fillna(np.inf)
        
        # Step 3: Compute standardized rarity score for each individual
        standardized_scores = []
        for i in range(N):
            z_scores = rarity_z.iloc[i]
            candidate_values = []
            
            # For each possible starting time t
            for t in range(min_t - 1, max_t):  # min_t-1 for 0-indexed
                # Find the maximum z-score within the window (for convergence, we want sustained low rarity)
                window_max = np.nanmax([z_scores[t + k] for k in range(window)])
                candidate_values.append(window_max)
            
            # Take the minimum across all starting times (most convergent period)
            if candidate_values:
                standardized_score = np.nanmin(candidate_values)
            else:
                standardized_score = np.nan
                
            standardized_scores.append(standardized_score)
        
        return standardized_scores

    def diagnose_convergence_calculation(self, z_threshold=1.5, max_t=None, window=1, inclusive=False, group_labels=None):
        """
        Diagnostic function to analyze convergence year calculation and identify 
        years with insufficient variance (std ≈ 0) that cannot trigger convergence.
        
        This is methodologically appropriate: when all individuals follow similar 
        trajectories in a given year, no convergence should be detected.
        
        Returns:
        --------
        dict: Diagnostic information including:
            - years_with_zero_variance: List of years where std ≈ 0
            - rarity_std_by_year: Standard deviation of rarity scores per year
            - n_individuals_with_convergence: Count of individuals with any convergence
            - convergence_year_distribution: Value counts of convergence years
        """
        if max_t is None:
            max_t = self.T - window + 1

        N = len(self.sequences)
        rarity_matrix = []

        # Calculate rarity scores (same as in compute_first_convergence_year)
        for seq in self.sequences:
            score = []
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
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
        
        # Count individuals with convergence
        convergence_years = self.compute_first_convergence_year(
            z_threshold=z_threshold, min_t=1, max_t=max_t, window=window,
            inclusive=inclusive, group_labels=group_labels
        )
        n_individuals_with_convergence = sum(1 for year in convergence_years if year is not None)
        
        # Distribution of convergence years
        convergence_year_counts = pd.Series(convergence_years).value_counts(dropna=False).sort_index()
        
        return {
            'years_with_zero_variance': years_with_zero_variance,
            'rarity_std_by_year': rarity_std_by_year.tolist(),
            'n_individuals_with_convergence': n_individuals_with_convergence,
            'convergence_year_distribution': convergence_year_counts.to_dict(),
            'total_individuals': N,
            'parameters_used': {
                'z_threshold': z_threshold,
                'max_t': max_t, 
                'window': window,
                'inclusive': inclusive,
                'group_labels': group_labels is not None
            }
        }

    def compute_path_uniqueness(self):
        """
        Count, for each individual, how many years t their suffix (from t to end)
        is unique in the population (frequency == 1). Uses suffix-based logic.
        """
        uniqueness_scores = []
        for seq in self.sequences:
            count = 0
            for t in range(self.T):
                suffix = tuple(seq[t:])
                if self.suffix_freq_by_year[t][suffix] == 1:
                    count += 1
            uniqueness_scores.append(count)
        return uniqueness_scores


def _removed_prefix_rarity_distribution_placeholder():
    return None


def plot_suffix_rarity_distribution(
    data,
    group_names=None,
    show_threshold=True,
    z_threshold=1.5,
    threshold_label=None,
    is_standardized_score=False,
    colors=None,
    figsize=(10, 6),
    save_as=None,
    dpi=300,
    show=True
):
    """
    Plot suffix rarity score distribution(s) with optional z-score threshold line.
    
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
        Custom label for threshold line. If None, uses appropriate default
    is_standardized_score : bool, default=False
        If True, treats data as standardized rarity scores (already z-scored) and draws 
        threshold line directly at -z_threshold. If False, calculates threshold 
        as mean - z_threshold * std of the raw data.
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
    # Single group (raw rarity scores)
    >>> plot_suffix_rarity_distribution(india_scores)
    
    # Multi-group comparison (raw scores)
    >>> data = {"India": india_scores, "US": us_scores}
    >>> plot_suffix_rarity_distribution(
    ...     data, 
    ...     show_threshold=True,
    ...     z_threshold=1.5,
    ...     save_as="rarity_comparison"
    ... )
    
    # Standardized rarity scores (correct threshold representation)
    >>> india_std_scores = indiv_convergence_india.compute_standardized_rarity_score(max_t=8, window=1)
    >>> us_std_scores = indiv_convergence_us.compute_standardized_rarity_score(max_t=8, window=1)
    >>> plot_suffix_rarity_distribution(
    ...     {"India": india_std_scores, "US": us_std_scores},
    ...     is_standardized_score=True,
    ...     z_threshold=1.5,
    ...     threshold_label="z = -1.5 (convergence boundary)",
    ...     save_as="standardized_rarity_comparison"
    ... )
    
    # Custom colors and no threshold
    >>> plot_suffix_rarity_distribution(
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
        default_colors = ["#A3BFD9", "#E8B88A", "#C6A5CF", "#A6C1A9", "#F4A460", "#87CEEB"]
        if isinstance(colors, dict):
            color_map = colors
        else:
            color_map = dict(zip(group_names, default_colors[:len(group_names)]))
    elif isinstance(colors, dict):
        color_map = colors
    else:
        color_map = dict(zip(group_names, colors))
    
    # Calculate per-group thresholds (mean - z * std) for convergence (low side)
    stats = {"per_group": {}}
    for g in group_names:
        if g in groups:
            arr = np.asarray(groups[g], dtype=float)
            mean_g = np.nanmean(arr)
            std_g = np.nanstd(arr, ddof=1)  # sample std to match pandas
            x_thresh_g = mean_g - z_threshold * std_g
            stats["per_group"][g] = {
                "mean": float(mean_g),
                "std": float(std_g),
                "threshold_value": float(x_thresh_g),
                "z_threshold": float(z_threshold),
                "is_group_relative": True
            }
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot distributions
    for group_name in group_names:
        if group_name in groups:
            scores = groups[group_name]
            color = color_map.get(group_name, "#1f77b4")
            sns.kdeplot(scores, label=group_name, fill=True, color=color, linewidth=2)
    
    # Add per-group threshold lines if requested (color-matched)
    if show_threshold:
        for g in group_names:
            if g in stats["per_group"]:
                xg = stats["per_group"][g]["threshold_value"]
                color = color_map.get(g, "#1f77b4")
                plt.axvline(xg, color=color, linestyle="--", linewidth=1.6)
                # Dynamic text positioning per group
                ax = plt.gca()
                y_max = ax.get_ylim()[1]
                text_y = y_max * 0.9
                lbl = threshold_label or f"z = -{z_threshold}"
                plt.text(xg, text_y, f"{g}: {lbl}", fontsize=10, ha="left", va="top", color=color)
    
    # Formatting
    if is_standardized_score:
        plt.xlabel("Standardized Suffix Rarity Score", fontsize=13)
    else:
        plt.xlabel("Suffix Rarity Score", fontsize=13)
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
        If None, automatically detects indicator columns (converged, first_convergence_year, 
        suffix_typicality_score, path_uniqueness, etc.)
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
    ...     indicator_columns=['converged', 'suffix_rarity_score', 'path_uniqueness'],
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
        # Common individual-level indicator patterns (convergence-focused)
        potential_indicators = [
            'converged', 'first_convergence_year', 'convergence_year',
            'suffix_rarity_score', 'path_uniqueness', 'rarity_score', 'uniqueness_score'
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
    if group_column is None:
        sample_size = len(df_clean)
    else:
        sizes = {}
        for g in df[group_column].unique():
            g_clean = df[df[group_column]==g][indicator_columns].apply(pd.to_numeric, errors='coerce').dropna()
            sizes[g] = len(g_clean)
        sample_size = sizes
    
    results['summary'] = {
        'method': correlation_method,
        'n_indicators': len(valid_cols),
        'indicators_included': valid_cols,
        'sample_size': sample_size
    }
    
    return results


def compute_path_uniqueness_by_group_suffix(sequences, group_labels):
        """
        Compute path uniqueness within each subgroup defined by group_labels using suffix-based approach.
        This is consistent with the convergence module's suffix-based logic.
        :param sequences: List of sequences.
        :param group_labels: List of group keys (same length as sequences), e.g., country, gender.
        :return: List of path uniqueness scores (same order as input).
        """
        from collections import defaultdict

        T = len(sequences[0])
        df = pd.DataFrame({
            "sequence": sequences,
            "group": group_labels
        })

        # Step 1: Precompute suffix frequency tables per group (changed from prefix to suffix)
        group_suffix_freq = {}
        for group, group_df in df.groupby("group"):
            suffix_freq = [defaultdict(int) for _ in range(T)]
            for seq in group_df["sequence"]:
                for t in range(T):
                    suffix = tuple(seq[t:])  # suffix from year t to end
                    suffix_freq[t][suffix] += 1
            group_suffix_freq[group] = suffix_freq

        # Step 2: Compute path uniqueness per individual using suffix logic
        uniqueness_scores = []
        for seq, group in zip(sequences, group_labels):
            suffix_freq = group_suffix_freq[group]
            count = 0
            for t in range(T):
                suffix = tuple(seq[t:])  # suffix from year t to end
                if suffix_freq[t][suffix] == 1:
                    count += 1
            uniqueness_scores.append(count)

        return uniqueness_scores


# Provide a default version for backward compatibility
def compute_path_uniqueness_by_group(sequences, group_labels):
    """
    Compute path uniqueness within each subgroup defined by group_labels.
    
    This is the default version using suffix-based approach (convergence logic).
    For explicit control, use compute_path_uniqueness_by_group_suffix() or 
    compute_path_uniqueness_by_group_prefix() from the prefix_tree module.
    
    :param sequences: List of sequences.
    :param group_labels: List of group keys (same length as sequences), e.g., country, gender.
    :return: List of path uniqueness scores (same order as input).
    """
    return compute_path_uniqueness_by_group_suffix(sequences, group_labels)
