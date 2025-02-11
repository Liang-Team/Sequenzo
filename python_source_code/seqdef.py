"""
@Author  : 梁彧祺
@File    : xinyi_original_seqdef.py
@Time    : 05/02/2025 12:47
@Desc    : Optimized SequenceData class with integrated color scheme & legend handling.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class SequenceData:
    """
    A class for defining and processing a sequence dataset for social sequence analysis.

    This class provides:
    - Sequence extraction & missing value handling.
    - Automatic alphabet (state space) management.
    - Efficient sequence-to-numeric conversion.
    - Color mapping & legend storage for visualization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        var: list,
        states: list,
        alphabet: list = None,
        labels: list = None,
        id_col: str = None,
        weights: np.ndarray = None,
        start: int = 1,
        missing_handling: dict = None,
        void: str = "%",
        nr: str = "*",
        cpal: list = None
    ):
        """
        Initialize the SequenceData object.

        :param data: DataFrame containing sequence data.
        :param var: List of columns containing sequences.
        :param states: List of unique states (categories).
        :param alphabet: Optional predefined state space.
        :param labels: Labels for states (optional, for visualization).
        :param id_col: Column name for row identifiers.
        :param weights: Sequence weights (optional).
        :param start: Starting time index (default: 1).
        :param missing_handling: Dict specifying handling for missing values (left, right, gaps).
        :param void: Symbol for void elements (default: "%").
        :param nr: Symbol for missing values (default: "*").
        :param cpal: Custom color palette for visualization.
        """
        self.data = data.copy()
        self.var = var
        self.states = states
        self.alphabet = alphabet or sorted(set(data[var].stack().dropna().unique()))
        self.labels = labels
        self.id_col = id_col
        self.weights = weights
        self.start = start
        self.missing_handling = missing_handling or {"left": np.nan, "right": "DEL", "gaps": np.nan}
        self.void = void
        self.nr = nr
        self.cpal = cpal

        # Validate parameters
        self._validate_parameters()

        # Extract & process sequences
        self.seqdata = self._extract_sequences()
        self._process_missing_values()
        self._convert_states()

        # Assign colors & save legend
        self._assign_colors()

        # Automatically print dataset overview
        print("\n✅ SequenceData initialized successfully! Here's a summary:")
        self.describe()

    @property
    def values(self):
        """Returns sequence data as a NumPy array, similar to xinyi_original_seqdef()."""
        return self.seqdata.to_numpy()

    def __repr__(self):
        return f"SequenceData({len(self.seqdata)} sequences, Alphabet: {self.alphabet})"

    def _validate_parameters(self):
        """Ensures correct input parameters."""
        if not self.states:
            raise ValueError("❌ 'states' must be provided.")
        if self.alphabet and set(self.alphabet) != set(self.states):
            raise ValueError("❌ 'alphabet' must match 'states'.")
        if self.labels and len(self.labels) != len(self.states):
            raise ValueError("❌ 'labels' must match the length of 'states'.")

    def _extract_sequences(self) -> pd.DataFrame:
        """Extracts only relevant sequence columns."""
        return self.data[self.var].copy()

    def _process_missing_values(self):
        """Handles missing values based on the specified rules."""
        left, right, gaps = self.missing_handling.values()

        # Fill left-side missing values
        if not pd.isna(left) and left != "DEL":
            self.seqdata.fillna(left, inplace=True)

        # Process right-side missing values
        if right == "DEL":
            self.seqdata = self.seqdata.apply(lambda row: row.dropna().reset_index(drop=True), axis=1)

        # Process gaps (internal missing values)
        if not pd.isna(gaps) and gaps != "DEL":
            self.seqdata.replace(self.nr, gaps, inplace=True)

    def _convert_states(self):
        """Converts categorical states into numerical values for processing."""
        unique_states = sorted(set(self.seqdata.stack().dropna().unique()))

        # Create mapping
        self.state_mapping = {state: idx for idx, state in enumerate(unique_states)}

        # Convert sequences to numeric values
        self.seqdata = self.seqdata.map(lambda x: self.state_mapping.get(x, np.nan))

    def _assign_colors(self, reverse_colors=True):
        """Assigns a color palette using the Spectral scheme by default."""
        num_states = len(self.states)
        spectral_colors = sns.color_palette("Spectral", num_states)

        if reverse_colors:
            spectral_colors = list(reversed(spectral_colors))

        self.color_map = {state: spectral_colors[i] for i, state in enumerate(self.states)}

        # Save legend
        self.legend_handles = [plt.Rectangle((0, 0), 1, 1, color=self.color_map[state], label=state) for state in
                               self.states]

    def get_colormap(self):
        """Returns a ListedColormap for visualization."""
        return ListedColormap([self.color_map[state] for state in self.states])

    def describe(self):
        """Prints an overview of the sequence dataset."""
        print(f"🔍 Number of sequences: {len(self.seqdata)}")
        print(f"📏 Min/Max sequence length: {self.seqdata.notna().sum(axis=1).min()} / {self.seqdata.notna().sum(axis=1).max()}")
        print(f"🔤 Alphabet: {self.alphabet}")

    def get_color_map(self):
        """Returns the color map for visualization."""
        return ListedColormap([self.color_map[state] for state in self.alphabet])

    def get_legend(self):
        """Returns the legend handles and labels for visualization."""
        return [handle for handle in self.legend_handles], self.states

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the processed sequence dataset as a DataFrame."""
        return self.seqdata

    def plot_legend(self, save_as='legend', dpi=200):
        """Displays the saved legend for sequence state colors."""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.legend(handles=self.legend_handles, loc='center', title="States", fontsize=10)
        ax.axis('off')

        if save_as:
            plt.show()
            plt.savefig(save_as, dpi=dpi)
        else:
            plt.tight_layout()
            plt.show()


# Example Usage
if __name__ == "__main__":
    df = pd.read_csv('/test_data/data_2.7w.csv')

    sequence_data = SequenceData(
        data=df,
        var=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'],  # Specify sequence columns
        states=['data', 'math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    )

    sequence_data.describe()
    processed_sequence = sequence_data.to_dataframe()
    print(processed_sequence)

    # Display legend
    sequence_data.plot_legend()
