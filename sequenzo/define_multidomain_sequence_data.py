"""
@Author  : Yuqi Liang 梁彧祺
@File    : define_multidomain_sequence_data.py
@Time    : 14/04/2025 17:27
@Desc    : 
    This module defines the `MultidomainSequenceData` class, a specialized container for handling
    multi-channel (multidomain) sequence data in social sequence analysis.

    Unlike the standard `SequenceData` class which represents a single-domain trajectory (e.g., employment history),
    `MultidomainSequenceData` is designed to work with multiple domains simultaneously (e.g., employment, marriage, parenthood).

    Main responsibilities of this class:
    - Accept a list of multiple `SequenceData` objects, each representing one domain.
    - Combine the sequences into a new multidomain representation using token concatenation (e.g., "employed+married+nochild").
    - Automatically assign combined-state color maps and legends.
    - Provide `.values`, `.to_dataframe()`, `.get_colormap()`, and `.get_legend()` methods
      that are fully compatible with downstream plotting functions (e.g., `plot_sequence_index`).

    Why this is a separate class instead of using `SequenceData`:
    - `SequenceData` is designed for a single categorical time series; extending it to multiple domains
      would compromise its clarity and simplicity.
    - A multidomain sequence needs fundamentally different construction, color assignment, and visualization logic.
    - Keeping `MultidomainSequenceData` independent improves modularity and makes domain-specific operations easier to manage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

__all__ = ['MultidomainSequenceData']


class MultidomainSequenceData:
    """
    A class to combine and manage multiple SequenceData objects into a multidomain sequence structure.

    This is equivalent to R's `seqMD()` + metadata wrapping.
    """

    def __init__(self,
                 domains: list,  # List of SequenceData instances
                 sep: str = "+",
                 labels: list[str] = None,
                 domain_names: list[str] = None,
                 validate_ids: bool = True):
        """
        :param domains: A list of SequenceData objects (one per domain)
        :param sep: String used to join domain-level states (default: '+')
        :param labels: Optional combined state labels (will be auto-generated otherwise)
        :param domain_names: Optional names for domains
        :param validate_ids: Whether to check that all domains share same individuals in same order
        """
        self.domains = domains
        self.sep = sep
        self.n_domains = len(domains)
        self.domain_names = domain_names or [f"Domain{i+1}" for i in range(self.n_domains)]

        if validate_ids:
            self._validate_individual_ids()

        self.ids = domains[0].ids
        self.seqdata = self._combine_domains()

        self.states = sorted(set(self.seqdata.stack().dropna().unique()))
        self.labels = labels or self.states
        self.color_map = self._assign_combined_colors()

        print(f"[>] MultidomainSequenceData initialized with {self.n_domains} domains, {len(self.seqdata)} sequences.")

    def _validate_individual_ids(self):
        """Ensure all domains have the same individuals in same order."""
        base_ids = self.domains[0].ids
        for i, d in enumerate(self.domains[1:], start=2):
            if not np.array_equal(d.ids, base_ids):
                raise ValueError(f"Domain {i} has mismatching individual IDs.")

    def _combine_domains(self) -> pd.DataFrame:
        """Generates the multidomain sequence matrix."""
        sequences = [d.to_dataframe().astype(str) for d in self.domains]
        combined = sequences[0].copy()

        for df in sequences[1:]:
            combined = combined + self.sep + df

        combined.index = self.ids
        return combined

    @property
    def values(self) -> np.ndarray:
        """Return combined sequences as NumPy array of strings."""
        return self.seqdata.values.astype(str)

    def to_dataframe(self) -> pd.DataFrame:
        """Return combined sequence data as pandas DataFrame."""
        return self.seqdata.copy()

    def _assign_combined_colors(self):
        """Assign colors to combination states using a colormap."""
        import seaborn as sns
        states = self.states
        n_states = len(states)

        palette = sns.color_palette("Spectral", n_colors=n_states)
        palette = list(reversed(palette))

        return {state: palette[i] for i, state in enumerate(states)}

    def get_colormap(self) -> ListedColormap:
        """Return matplotlib colormap for visualization."""
        return ListedColormap([self.color_map[s] for s in self.states])

    def get_legend(self):
        """Return legend handles and labels."""
        handles = [plt.Rectangle((0, 0), 1, 1, color=self.color_map[state], label=state) for state in self.states]
        return handles, self.states

    def describe(self):
        """Print basic info about multidomain sequence data."""
        print(f"[>] Number of sequences: {len(self.seqdata)}")
        print(f"[>] Number of time points: {self.seqdata.shape[1]}")
        print(f"[>] Alphabet size: {len(self.states)}")
        print(f"[>] Domains: {self.domain_names}")
