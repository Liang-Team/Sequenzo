"""
@Author  : Yuqi Liang 梁彧祺
@File    : individual_level_indicators.py
@Time    : 08/08/2025 15:30
@Desc    : 
    This module provides methods for calculating individual-level indicators in sequence data analysis.
    It includes tools to assess sequence divergence, identify divergence timing, measure prefix rarity,
    and evaluate path uniqueness for individuals or groups. 
    
    These indicators help quantify how typical or unique an individual's sequence is within a population, 
    and can be used for both overall and subgroup analyses.

    TODO: path uniquess is duplicated with those in prefix_tree
"""
from collections import defaultdict
import math
import pandas as pd


class IndividualConvergence:
    def __init__(self, sequences):
        self.sequences = sequences
        self.T = len(sequences[0])
        self.suffix_freq_by_year = self._build_suffix_frequencies()

    def _build_suffix_frequencies(self):
        """Build frequency tables for suffixes starting at each year."""
        freq_by_year = [defaultdict(int) for _ in range(self.T)]
        for seq in self.sequences:
            for t in range(self.T):
                suffix = tuple(seq[t:])  # Suffix from year t to end
                freq_by_year[t][suffix] += 1
        return freq_by_year

    def compute_converged(self, z_threshold=1.5, max_t=None, window=1):
        """
        Compute binary convergence status based on suffix typicality z-scores.

        :param z_threshold: Z-score threshold above which an individual is considered converged.
        :param max_t: Maximum year (1-indexed) before which convergence is considered valid.
        :param window: Number of consecutive high-z years required (default: 1).
        :return: List of 0/1 flags indicating whether each individual converged.
        """
        if max_t is None:
            max_t = self.T - window + 1
        
        N = len(self.sequences)
        typicality_matrix = []

        for seq in self.sequences:
            scores = []
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
                scores.append(math.log(freq + 1e-10))  # Log for typicality
            typicality_matrix.append(scores)

        typicality_df = pd.DataFrame(typicality_matrix)
        typicality_z = typicality_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        flags = []
        for i in range(N):
            z = typicality_z.iloc[i]
            converged = 0
            for t in range(0, min(max_t, self.T - window + 1)):
                if all(z[t + k] > z_threshold for k in range(window)):
                    converged = 1
                    break
            flags.append(converged)
        return flags

    def compute_convergence_year(self, z_threshold=1.5, max_t=None, window=1):
        """
        Compute first convergence year based on suffix typicality z-scores.

        :param z_threshold: Z-score threshold for defining convergence.
        :param max_t: Maximum year (1-indexed) considered valid for convergence.
        :param window: Number of consecutive high-z years to confirm convergence.
        :return: List of convergence years (1-indexed), or None if no convergence detected.
        """
        if max_t is None:
            max_t = self.T - window + 1
            
        N = len(self.sequences)
        typicality_matrix = []

        for seq in self.sequences:
            scores = []
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
                scores.append(math.log(freq + 1e-10))
            typicality_matrix.append(scores)

        typicality_df = pd.DataFrame(typicality_matrix)
        typicality_z = typicality_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        years = []
        for i in range(N):
            z = typicality_z.iloc[i]
            year = None
            for t in range(0, min(max_t, self.T - window + 1)):
                if all(z[t + k] > z_threshold for k in range(window)):
                    year = t + 1  # Convert to 1-indexed
                    break
            years.append(year)
        return years

    def compute_suffix_typicality_score(self):
        """
        Compute cumulative suffix typicality scores for all individuals.
        
        :return: List of typicality scores (higher = more aligned with common outcomes).
        """
        typicality_scores = []
        N = len(self.sequences)

        for seq in self.sequences:
            score = 0.0
            for t in range(self.T):
                suffix = tuple(seq[t:])
                freq = self.suffix_freq_by_year[t][suffix] / N
                score += math.log(freq + 1e-10)  # Log for typicality (common = positive)
            typicality_scores.append(score)
        return typicality_scores

    def compute_path_typicality(self):
        """
        Compute path typicality as count of years with common suffixes.
        
        :return: List of counts indicating how many years have typical suffixes.
        """
        typicality_counts = []
        for seq in self.sequences:
            count = 0
            for t in range(self.T):
                suffix = tuple(seq[t:])
                if self.suffix_freq_by_year[t][suffix] > 1:  # Shared by multiple individuals
                    count += 1
            typicality_counts.append(count)
        return typicality_counts

    @staticmethod
    def compute_path_typicality_by_group(sequences, group_labels):
        """
        Compute path typicality within each subgroup defined by group_labels.
        
        :param sequences: List of sequences.
        :param group_labels: List of group keys (same length as sequences).
        :return: List of path typicality scores (same order as input).
        """
        T = len(sequences[0])
        df = pd.DataFrame({
            "sequence": sequences,
            "group": group_labels
        })

        # Build suffix frequency tables per group
        group_suffix_freq = {}
        for group, group_df in df.groupby("group"):
            suffix_freq = [defaultdict(int) for _ in range(T)]
            for seq in group_df["sequence"]:
                for t in range(T):
                    suffix = tuple(seq[t:])
                    suffix_freq[t][suffix] += 1
            group_suffix_freq[group] = suffix_freq

        # Compute path typicality per individual
        typicality_scores = []
        for seq, group in zip(sequences, group_labels):
            suffix_freq = group_suffix_freq[group]
            count = 0
            for t in range(T):
                suffix = tuple(seq[t:])
                if suffix_freq[t][suffix] > 1:  # Shared within group
                    count += 1
            typicality_scores.append(count)

        return typicality_scores

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
