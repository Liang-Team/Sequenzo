"""
@Author  : 梁彧祺
@File    : define_sequence_data.py
@Time    : 05/02/2025 12:47
@Desc    :

    Optimized SequenceData class with integrated color scheme & legend handling.

    Note on `states` and `alphabet`:

    In traditional sequence analysis tools (e.g., TraMineR), the `alphabet` refers to the full set of distinct states
    found in the data and is often inferred automatically from the observed sequences.

    However, in this implementation, we require the user to explicitly provide the set of `states`. This explicit control
    is essential for ensuring consistent ordering of states, reproducibility of visualizations, and compatibility across
    sequence datasets — especially when certain states may not appear in a given subset of the data.

    As a result, `alphabet` is automatically set to `states` upon initialization, and kept as a semantic alias for clarity
    and potential compatibility. Users should treat `states` as the definitive state space and are not required to provide
    `alphabet` separately.

    # ----------------------------------------------------------------------
    # [Hint] Handling the ID column for sequence analysis
    # ----------------------------------------------------------------------

    # STEP 1: Check if your DataFrame already has a column representing unique entity IDs
    # For example, check if "Entity ID" or "country" or any other identifier exists:
    print(df.columns)

    # If your data already has an ID column (e.g., 'Entity ID'), you can directly use it:
    seq = SequenceData(df, id_col='Entity ID', time=..., states=..., time_type='year')

    # ----------------------------------------------------------------------
    # STEP 2: If your data has NO ID column, use the helper function below
    # ----------------------------------------------------------------------
    from sequenzo.utils import assign_unique_ids

    # This will insert a new ID column named 'Entity ID' as the first column
    df = assign_unique_ids(df, id_col_name='Entity ID')

    # Optional: Save it for future use to avoid repeating this step
    df.to_csv('your_dataset_with_ids.csv', index=False)

    # Then you can use it like this:
    seq = SequenceData(df, id_col='Entity ID', time=..., states=..., time_type='year')

"""
# Only applicable to Python 3.7+, add this line to defer type annotation evaluation
from __future__ import annotations
# Define the public API at the top of the file
__all__ = ['SequenceData']

# Global variables and other imports that do not depend on pandas are placed here
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from docutils.parsers.rst import states
from matplotlib.colors import ListedColormap
import re


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
        time_type: str,
        time: list,
        states: list,
        labels: list = None,
        id_col: str = None,
        weights: np.ndarray = None,
        start: int = 1,
        missing_handling: dict = None,
        void: str = "%",
        nr: str = "*",
        custom_colors: list = None
    ):
        """
        Initialize the SequenceData object.

        :param data: DataFrame containing sequence data.
        :param time: List of columns containing time labels.
        :param time_type: Type of time labels (e.g., "year", "age").
        :param states: List of unique states (categories).
        :param alphabet: Optional predefined state space.
        :param labels: Labels for states (optional, for visualization).
        :param id_col: Column name for row identifiers, which is very important for hierarchical clustering.
        :param weights: Sequence weights (optional).
        :param start: Starting time index (default: 1).
        :param missing_handling: Dict specifying handling for missing values (left, right, gaps).
        :param void: Symbol for void elements (default: "%").
        :param nr: Symbol for missing values (default: "*").
        :param custom_colors: Custom color palette for visualization.
        """
        # Ensure time_type is either "year" or "age"
        if time_type not in ["year", "age"]:
            raise ValueError("time_type must be either 'year' or 'age'")

        # Import pandas here instead of the top of the file
        import pandas as pd

        self.data = data.copy()
        self.time = time

        if time_type == "year":
            self.time_type = "year"
        elif time_type == "age":
            self.time_type = "age"

        # Remove all non-numeric characters from the year labels, e.g., "Year2020" -> "2020", or "C1" -> "1"
        # self.cleaned_time = [re.sub(r'\D', '', str(year)) for year in time]
        # No longer support this feature as we encourage users to clean the time variables.
        # TODO: might implement a helper function for users to clean up their time variables.
        self.cleaned_time = time
        self.states = states.copy()
        self.alphabet = states.copy() or sorted(set(data[time].stack().unique()))
        self.labels = labels or states.copy()
        self.id_col = id_col
        self.ids = np.array(data[id_col].values) if self.id_col else data.index
        self.weights = weights
        self.start = start
        # TODO 这个没有用，要看看是否需要去除
        self.missing_handling = missing_handling or {"left": np.nan, "right": "DEL", "gaps": np.nan}
        self.void = void
        self.nr = nr
        self.custom_colors = custom_colors

        # Validate parameters
        self._validate_parameters()

        # Extract & process sequences
        self.seqdata = self._extract_sequences()
        self._process_missing_values()

        # The following two lines of code are for visualization
        self.state_to_label = dict(zip(self.states, self.labels))
        self.label_to_state = dict(zip(self.labels, self.states))

        self._convert_states()

        # Assign colors & save legend
        self._assign_colors()

        # Automatically print dataset overview
        print("\n[>] SequenceData initialized successfully! Here's a summary:")
        self.describe()

    @property
    def values(self):
        """Returns sequence data as a NumPy array, similar to xinyi_original_seqdef()."""
        return self.seqdata.to_numpy(dtype=np.int32)

    def __repr__(self):
        return f"SequenceData({len(self.seqdata)} sequences, States: {self.states})"

    def _validate_parameters(self):
        """Ensures correct input parameters and checks consistency with data."""
        # Check states, alphabet, labels
        if not self.states:
            raise ValueError("'states' must be provided.")

        # Validate that states are present in the actual data values
        data_values = set(self.data[self.time].stack().unique())
        unmatched_states = [s for s in self.states if s not in data_values]

        if unmatched_states:
            raise ValueError(
                f"[!] The following provided 'states' are not found in the data: {unmatched_states}\n"
                f"    Hint: Check spelling or formatting. Data contains these unique values: {sorted(data_values)}"
            )

        # ----------------
        # Check if ID column is provided and valid
        if self.id_col is None or self.id_col not in self.data.columns:
            raise ValueError(
                f"[!] You must specify a valid `id_col` parameter that exists in your dataset.\n"
                f"    ID is required to uniquely identify each sequence (e.g., individuals).\n"
                f"    → Hint: If your data does not have an ID column yet, you can use the helper function:\n\n"
                f"        from sequenzo.utils import assign_unique_ids\n"
                f"        df = assign_unique_ids(df, id_col_name='Entity ID')\n"
                f"        df.to_csv('your_dataset_with_ids.csv', index=False)\n\n"
                f"    This will permanently assign unique IDs to your dataset for future use."
            )

        self.ids = np.array(self.data[self.id_col].values)

        # Validate ID uniqueness and length
        if len(self.ids) != len(self.data):
            raise ValueError(f"[!] Length of ID column ('{self.id_col}') must match number of rows in the dataset.")
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError(f"[!] IDs in column '{self.id_col}' must be unique.")

        # ----------------
        if self.alphabet and set(self.alphabet) != set(self.states):
            raise ValueError("'alphabet' must match 'states'.")

        if self.labels:
            if len(self.labels) != len(self.states):
                raise ValueError("'labels' must match the length of 'states'.")

            # Ensure labels are all strings
            non_string_labels = [label for label in self.labels if not isinstance(label, str)]
            if non_string_labels:
                raise TypeError(
                    f"[!] All elements in 'labels' must be strings for proper visualization (e.g., for legends or annotations).\n"
                    f"    Detected non-string labels: {non_string_labels}\n"
                    f"    Example fix: instead of using `labels = [1, 2, 3]`, use `labels = ['Single', 'Married', 'Divorced']`."
                )

        # Check weights
        if self.weights is not None:
            if len(self.weights) != len(self.data):
                raise ValueError("'weights' must match the length of 'data'.")
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

        if self.ismissing:
            # 判断 states 中是否已经含有 Missing（无论是字符串还是 np.nan）
            if "Missing" not in self.states and not any(pd.isna(s) for s in self.states):
                # 自动判断 states 是字符串型还是数字型
                example_missing = "'Missing'" if all(isinstance(s, str) for s in self.states) else "np.nan"
                quote = "" if example_missing == "np.nan" else "'"

                print(
                    "[!] Detected missing values (empty cells) in the sequence data.\n"
                    f"    → Automatically added {example_missing} to `states` and `labels` for compatibility.\n"
                    "    However, it's strongly recommended to manually include it when defining `states` and `labels`.\n"
                    "    For example:\n\n"
                    f"        states = [{quote}At Home{quote}, {quote}Left Home{quote}, {example_missing}]\n"
                    f"        labels = [{quote}At Home{quote}, {quote}Left Home{quote}, {quote}Missing{quote}]\n\n"
                    "    This ensures consistent color mapping and avoids unexpected visualization errors."
                )

                # 添加 missing 到 states 和 labels
                if example_missing == "'Missing'":
                    self.states.append("Missing")
                    self.labels.append("Missing")
                else:
                    self.states.append(np.nan)
                    self.labels.append("Missing")

    def _convert_states(self):
        """
        Converts categorical states into numerical values for processing.
        Note that the order has to be the same as when the user defines the states of the class,
        as it is very important for visualization.
        Otherwise, the colors will be assigned incorrectly.

        For instance, self.states = ['Very Low', 'Low', 'Middle', 'High', 'Very High'], as the user defines when defining the class
        but the older version here is {'High': 1, 'Low': 2, 'Middle': 3, 'Very High': 4, 'Very Low': 5}
        """
        correct_order = self.states

        # Create the state mapping with correct order
        self.state_mapping = {original_state: i + 1 for i, original_state in enumerate(self.states)}
        # 保留下面的映射关系，这样后面 legend 和绘图都能用 numeric 编码了
        self.inverse_state_mapping = {v: k for k, v in self.state_mapping.items()}

        # Apply the mapping
        # If there are missing values, replace them with the last index + 1
        # And update the additional missing value as a new state in self.state and self.alphabet
        try:
            self.seqdata = self.seqdata.map(lambda x: self.state_mapping.get(x, len(self.states)))
        except AttributeError:
            self.seqdata = self.seqdata.applymap(lambda x: self.state_mapping.get(x, len(self.states)))

        if self.ids is not None:
            self.seqdata.index = self.ids

    def _assign_colors(self, reverse_colors=True):
        """Assigns a color palette using user-defined or default Spectral palette."""
        num_states = len(self.states)

        if self.custom_colors:
            if len(self.custom_colors) != num_states:
                raise ValueError("Length of custom_colors must match number of states.")
            color_list = self.custom_colors
        else:
            if num_states <= 20:
                color_list = sns.color_palette("Spectral", num_states)
            else:
                color_list = sns.color_palette("cubehelix", num_states)

            if reverse_colors:
                color_list = list(reversed(color_list))

        # self.color_map = {state: color_list[i] for i, state in enumerate(self.states)}
        # 这样所有 color map key 是 1, 2, 3...，就可以和 imshow(vmin=1, vmax=N) 对齐
        self.color_map = {i + 1: color_list[i] for i in range(num_states)}

        # 构造以 label 为 key 的 color_map（用于 legend）
        self.color_map_by_label = {
            self.state_to_label[state]: self.color_map[self.state_mapping[state]]
            for state in self.states
        }

    def get_colormap(self):
        """Returns a ListedColormap for visualization."""
        # return ListedColormap([self.color_map[state] for state in self.states])
        return ListedColormap([self.color_map[i + 1] for i in range(len(self.states))])

    def describe(self):
        """
        Prints an overview of the sequence dataset.

        # NOTE:
            # Printing 'missing_index' directly may cause issues in Jupyter Notebook/Lab if the list is too long.
            # For example, if there are thousands of sequences with missing values, the full list can easily exceed
            # the IOPub data rate limit (1MB/sec by default), which will interrupt output to the client.
            # To avoid this, it's safer to only display a subset (e.g., the first 10) or add a 'verbose' flag to control output.
        """
        print(f"[>] Number of sequences: {len(self.seqdata)}")
        print(f"[>] Number of time points: {self.n_steps}")

        if self.ismissing:
            lengths = self.seqdata.apply(lambda row: (row != len(self.states)).sum(), axis=1)
            print(f"[>] Min/Max sequence length: {lengths.min()} / {lengths.max()}")

            # Identify missing values and related IDs
            missing_locs = self.seqdata.stack()[self.seqdata.stack() == len(self.states)].index.get_level_values(0)
            missing_count = len(missing_locs)
            unique_missing_ids = missing_locs.unique().tolist()
            print(f"[>] There are {missing_count} missing values across {len(unique_missing_ids)} sequences.")
            print(f"    First few missing sequence IDs: {unique_missing_ids[:10]} ...")

            # Find and display sequences with the most missing points
            missing_counts = self.seqdata.isin([len(self.states)]).sum(axis=1)
            most_missing = missing_counts[missing_counts > 0].sort_values(ascending=False).head(5)
            print("[>] Top sequences with the most missing time points:")
            print("    (Each row shows a sequence ID and its number of missing values)\n")
            print(most_missing.rename("Missing Count").to_frame().rename_axis("Sequence ID"))

        else:
            print(
                f"[>] Min/Max sequence length: {self.seqdata.notna().sum(axis=1).min()} / {self.seqdata.notna().sum(axis=1).max()}")

        print(f"[>] States: {self.states}")
        print(f"[>] Labels: {self.labels}")

    def get_legend(self):
        """Returns the legend handles and labels for visualization."""
        # self.legend_handles = [plt.Rectangle((0, 0), 1, 1,
        #                                      color=self.color_map[state],
        #                                      label=label)
        #                        for state, label in zip(self.states, self.labels)]
        # return [handle for handle in self.legend_handles], self.labels

        self.legend_handles = [
            plt.Rectangle((0, 0), 1, 1,
                          color=self.color_map[i + 1],
                          label=self.labels[i])
            for i in range(len(self.states))
        ]
        return self.legend_handles, self.labels

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the processed sequence dataset as a DataFrame."""
        return self.seqdata

    def plot_legend(self, save_as=None, dpi=200):
        """Displays the saved legend for sequence state colors."""
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.legend(handles=self.legend_handles, loc='center', title="States", fontsize=10)
        ax.axis('off')

        if save_as:
            plt.savefig(save_as, dpi=dpi)
            plt.show()
        else:
            plt.tight_layout()
            plt.show()

    # ------------------------------
    # The following are for multidomain sequence analysis, especially for seqdomassoc()

    @property
    def n_sequences(self):
        """Returns number of sequences (rows)."""
        return self.seqdata.shape[0]

    @property
    def n_steps(self):
        """Returns sequence length (columns)."""
        return self.seqdata.shape[1]

    @property
    def alphabet(self):
        """Returns state alphabet."""
        return self._alphabet

    @alphabet.setter
    def alphabet(self, val):
        self._alphabet = val

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        self._weights = val

    def flatten(self) -> np.ndarray:
        """Flatten all sequences into a 1D array (row-wise)."""
        return self.seqdata.values.flatten()

    def flatten_weights(self) -> np.ndarray:
        """
        Repeat weights across sequence length for 1D alignment with flatten().
        E.g., 5 sequences × 10 steps → repeat each weight 10 times.
        """
        return np.repeat(self.weights, self.n_steps)

    def to_numeric(self) -> np.ndarray:
        """Returns integer-coded sequence data as NumPy array."""
        return self.seqdata.to_numpy(dtype=np.int32)

    def get_xtabs(self, other: SequenceData, weighted=True) -> np.ndarray:
        """
        NumPy-only version of get_xtabs.
        Returns a raw NumPy matrix: shape (len(alphabet1), len(alphabet2))
        """
        if self.n_sequences != other.n_sequences or self.n_steps != other.n_steps:
            raise ValueError("Both SequenceData objects must have same shape.")

        v1 = self.flatten()
        v2 = other.flatten()

        # Equivalent to self.alphabet,
        # but alphabet cannot be used directly, because it does not account for missing values
        n1 = len(self.states)
        n2 = len(other.states)

        table = np.zeros((n1, n2), dtype=np.float64)

        if weighted:
            w = self.flatten_weights()
            # Safe increment using integer indices
            # Numpy's index starts from 0, thus it is important to reduce by 1
            np.add.at(table, (v1 - 1, v2 - 1), w)
        else:
            np.add.at(table, (v1 - 1, v2 - 1), 1)

        return table





