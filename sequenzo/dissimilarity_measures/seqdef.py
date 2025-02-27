"""
@Author  : 梁彧祺
@File    : seqdef.py
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
        time: list,
        states: list,
        alphabet: list = None,
        labels: list = None,
        ids: list = None,
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
        :param time: List of columns containing sequences.
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
        self.time = time
        # Clean the labels of time steps instead of keeping "C1", ..."C10"
        self.cleaned_time = [str(i + 1) for i in range(len(time))]
        self.states = states
        self.alphabet = alphabet or sorted(set(data[time].stack().dropna().unique()))
        self.labels = labels
        self.ids = ids
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
        # check states, alphabet, labels
        if not self.states:
            raise ValueError("❌ 'states' must be provided.")
        if self.alphabet and set(self.alphabet) != set(self.states):
            raise ValueError("❌ 'alphabet' must match 'states'.")
        if self.labels and len(self.labels) != len(self.states):
            raise ValueError("❌ 'labels' must match the length of 'states'.")

        # check ids
        if self.ids is not None:
            if len(self.ids) != len(self.data):
                raise ValueError("❌ 'ids' must match the length of 'data'.")

            if len(np.unique(self.ids)) != len(self.ids):
                raise ValueError("❌ 'ids' must be unique.")

        # check weights
        if self.weights is not None:
            if len(self.weights) != len(self.data):
                raise ValueError("❌ 'weights' must match the length of 'data'.")
        else:
            self.weights = np.ones(self.data.shape[0])

    def _extract_sequences(self) -> pd.DataFrame:
        """Extracts only relevant sequence columns."""
        return self.data[self.time].copy()

    def _process_missing_values(self):
        """Handles missing values based on the specified rules."""
        # left, right, gaps = self.missing_handling.values()
        #
        # # Fill left-side missing values
        # if not pd.isna(left) and left != "DEL":
        #     self.seqdata.fillna(left, inplace=True)
        #
        # # Process right-side missing values
        # if right == "DEL":
        #     self.seqdata = self.seqdata.apply(lambda row: row.dropna().reset_index(drop=True), axis=1)
        #
        # # Process gaps (internal missing values)
        # if not pd.isna(gaps) and gaps != "DEL":
        #     self.seqdata.replace(self.nr, gaps, inplace=True)

        self.ismissing = self.seqdata.isna().any().any()

    def _convert_states(self):
        """Converts categorical states into numerical values for processing."""
        unique_states = sorted(set(self.seqdata.stack().dropna().unique()))

        # with missing data
        if self.seqdata.isna().any().any():
            self.state_mapping = {state: idx + 1 for idx, state in enumerate(unique_states)}    # Create mapping
            self.seqdata = self.seqdata.applymap(lambda x: self.state_mapping.get(x, 0))    # Convert sequences to numeric values

        # without missing data
        else:
            self.state_mapping = {state: idx for idx, state in enumerate(unique_states)}
            self.seqdata = self.seqdata.applymap(lambda x: self.state_mapping.get(x, np.nan))

        if self.ids is not None:
            self.seqdata.index = self.ids

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

        if self.seqdata.isna().any().any():
            lengths = self.seqdata.apply(lambda row: (row != 0).sum(), axis=1)
            print(f"📏 Min/Max sequence length: {lengths.min()} / {lengths.max()}")
        else:
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
    df = pd.read_csv('../../../seqdef/missing_data_1000.csv')

    sequence_data = SequenceData(
        data=df,
        time=['C1', 'C2', 'C3', 'C4', 'C5'],  # Specify sequence columns
        ids=df['worker_id'].tolist(),
        states=["Non-computing", "Non-technical computing", "Technical computing"]
    )

    processed_sequence = sequence_data.seqdata
    print(processed_sequence)
    # processed_sequence.to_csv('missing_sequence_data.csv')

    # Display legend
    # sequence_data.plot_legend()