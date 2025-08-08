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

    def compute_divergence_year(self, z_threshold=1.5, min_t=3, window=1):
        """
        Compute divergence year based on rarity score z-scores.

        :param z_threshold: Z-score threshold for defining divergence.
        :param min_t: Minimum year (1-indexed) considered valid for divergence.
        :param window: Number of consecutive high-z years to confirm divergence.
        :return: List of divergence years (1-indexed), or None if no divergence detected.
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
