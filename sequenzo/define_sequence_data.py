"""
@Author  : 梁彧祺 Yuqi Liang, 李欣怡 Xinyi Li
@File    : define_sequence_data.py
@Time    : 05/02/2025 12:47
@Desc    :

    Optimized SequenceData class with integrated color scheme & legend handling.

    Note on `states` and `alphabet`:

    In traditional sequence analysis tools (e.g., TraMineR), the `alphabet` refers to the full set of distinct states
    found in the data and is often inferred automatically from the observed sequences.

    However, in this implementation, we require the user to explicitly provide the set of `states`. This explicit control
    is essential for ensuring consistent ordering of states, reproducibility of visualizations, and compatibility across
    sequence datasets - especially when certain states may not appear in a given subset of the data.

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
    seq = SequenceData(df, id_col='Entity ID', time=..., states=...)

    # ----------------------------------------------------------------------
    # STEP 2: If your data has NO ID column, use the helper function below
    # ----------------------------------------------------------------------
    from sequenzo.utils import assign_unique_ids

    # This will insert a new ID column named 'Entity ID' as the first column
    df = assign_unique_ids(df, id_col_name='Entity ID')

    # Optional: Save it for future use to avoid repeating this step
    df.to_csv('your_dataset_with_ids.csv', index=False)

    # Then you can use it like this:
    seq = SequenceData(df, id_col='Entity ID', time=..., states=...)

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
from typing import Union


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
        labels: list = None,
        id_col: str = None,
        weights: np.ndarray = None,
        start: int = 1,
        custom_colors: list = None,
        additional_colors: dict = None,
        missing_values: Union[None, int, float, str, list] = None
    ):
        """
        Initialize the SequenceData object.

        :param data: DataFrame containing sequence data.
        :param time: List of columns containing time labels.
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
            If provided, should be a list of colors matching the number of states.
            Colors can be hex strings (e.g., "#FF5733") or RGB tuples.
        :param additional_colors: Dictionary to specify additional custom colors for specific states 
            while keeping the default palette for others. This is useful when you want to keep default colors 
            but assign custom colors to specific states (e.g., {"Other": "#BDBDBD"} to make "Other" gray).
            Format: {state_name: color}, where color can be hex string (e.g., "#BDBDBD") or RGB tuple.
            Example: additional_colors={"Other": "#BDBDBD", "Missing": "#E0E0E0"}
        :param missing_values: Custom missing value indicators. Can be:
            - None (default): Auto-detect missing values (NaN, string "Missing")
            - Single value: e.g., 99, 9, 1000, "Missing"
            - List: e.g., [99, 9, 1000] or ["Missing", "N/A"]
            The system will also check for pandas NaN and string "Missing" (case-insensitive)
            and warn if other missing values are detected.
        """
        # Import pandas here instead of the top of the file
        import pandas as pd

        self.data = data.copy()
        self.time = time

        # Remove all non-numeric characters from the year labels, e.g., "Year2020" -> "2020", or "C1" -> "1"
        # self.cleaned_time = [re.sub(r'\D', '', str(year)) for year in time]
        # No longer support this feature as we encourage users to clean the time variables.
        # TODO: might implement a helper function for users to clean up their time variables.
        self.cleaned_time = time
        self.states = states.copy()
        self.alphabet = states.copy() or sorted(set(data[time].stack().unique()))
        self.labels = labels or [str(s) for s in states]
        self.id_col = id_col
        self.ids = np.array(self.data[self.id_col].values) if self.id_col else data.index
        self.weights = weights
        self._weights_provided = weights is not None  # Track if weights were originally provided
        self.start = start
        self.custom_colors = custom_colors
        self.additional_colors = additional_colors or {}
        
        # Process missing_values parameter: convert to list format
        if missing_values is None:
            self.missing_values = []
        elif isinstance(missing_values, (list, tuple)):
            self.missing_values = list(missing_values)
        else:
            self.missing_values = [missing_values]
        
        # Track original number of states before processing missing values
        # This helps us determine if custom_colors needs adjustment
        self._original_num_states = len(self.states)
        self._missing_auto_added = False  # Track if Missing was automatically added

        # Validate parameters
        self._validate_parameters()
        
        # Validate additional_colors if provided
        if self.additional_colors:
            if self.custom_colors:
                raise ValueError(
                    "[!] You cannot use both 'custom_colors' and 'additional_colors' at the same time.\n"
                    "    -> Use 'custom_colors' to specify all colors, or\n"
                    "    -> Use 'additional_colors' to assign custom colors to specific states while keeping default colors."
                )
            # Check that all states in additional_colors exist in self.states
            invalid_states = [state for state in self.additional_colors.keys() if state not in self.states]
            if invalid_states:
                raise ValueError(
                    f"[!] The following states in 'additional_colors' are not found in 'states': {invalid_states}\n"
                    f"    Available states: {self.states}"
                )

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

        # Get all unique values from the data (including NaN)
        # stack() drops NaN by default, so we need to check separately
        data_values_no_nan = set(self.data[self.time].stack().unique())
        # Check if there are any NaN values in the data
        has_nan_in_data = self.data[self.time].isna().any().any()
        
        # Combine all data values (including NaN indicator if present)
        all_data_values = data_values_no_nan.copy()
        if has_nan_in_data:
            all_data_values.add(np.nan)
        
        # Validate that states are present in the actual data values
        states_clean = [s for s in self.states if not pd.isna(s)]    # stack() removes nan values, so if states contains np.nan, it will cause an error
        unmatched_states = [s for s in states_clean if s not in data_values_no_nan]

        if unmatched_states:
            raise ValueError(
                f"[!] The following provided 'states' are not found in the data: {unmatched_states}\n"
                f"    Hint: Check spelling or formatting. Data contains these unique values: {sorted([v for v in data_values_no_nan if not pd.isna(v)])}"
            )
        
        # Validate that all data values are present in the provided states (complete state space check)
        # Exclude missing values from this check (NaN and user-specified missing_values)
        states_set = set(self.states)
        # Check for NaN in states
        has_nan_in_states = any(pd.isna(s) for s in self.states)
        
        # Get missing value indicators to exclude from the check
        missing_indicators = set()
        if has_nan_in_states:
            missing_indicators.add(np.nan)
        # Add user-specified missing_values
        for mv in self.missing_values:
            if pd.isna(mv):
                missing_indicators.add(np.nan)
            else:
                missing_indicators.add(mv)
        # Also check for string "Missing" (case-insensitive) in states
        for s in self.states:
            if isinstance(s, str) and s.lower() == 'missing':
                missing_indicators.add(s)
        
        # Find data values that are not in states and not missing values
        missing_from_states = []
        for dv in all_data_values:
            # Skip if it's a missing value indicator
            if pd.isna(dv):
                # NaN is only allowed if it's in states or in missing_indicators
                if np.nan not in missing_indicators and not has_nan_in_states:
                    missing_from_states.append("NaN")
            elif dv in missing_indicators:
                continue  # This is a known missing value, skip
            elif dv not in states_set:
                missing_from_states.append(dv)
        
        if missing_from_states:
            # Format the error message nicely
            data_values_display = sorted([v for v in data_values_no_nan if not pd.isna(v)])
            if has_nan_in_data:
                data_values_display.append("NaN")
            
            raise ValueError(
                f"[!] The following values found in the data are not included in your 'states' list: {missing_from_states}\n"
                f"    Your provided states: {self.states}\n"
                f"    All unique values in data: {data_values_display}\n"
                f"    Hint: You must include ALL unique values from the data in your 'states' parameter.\n"
                f"    Missing values (NaN or user-specified) are automatically handled, but all other data values must be in 'states'."
            )

        # ----------------
        # Check if ID column is provided and valid
        if self.id_col is not None and self.id_col not in self.data.columns:
            raise ValueError(
                f"[!] You must specify a valid `id_col` parameter that exists in your dataset.\n"
                f"    ID is required to uniquely identify each sequence (e.g., individuals).\n"
                f"    -> Hint: If your data does not have an ID column yet, you can use the helper function:\n\n"
                f"        from sequenzo.utils import assign_unique_ids\n"
                f"        df = assign_unique_ids(df, id_col_name='Entity ID')\n"
                f"        df.to_csv('your_dataset_with_ids.csv', index=False)\n\n"
                f"    This will permanently assign unique IDs to your dataset for future use."
            )

        # Because it is already implemented at initialization time
        # self.ids = np.array(self.data[self.id_col].values)

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
                # Provide detailed error message showing what's missing or extra
                states_len = len(self.states)
                labels_len = len(self.labels)
                
                if labels_len < states_len:
                    missing_count = states_len - labels_len
                    error_msg = (
                        f"[!] 'labels' length ({labels_len}) is shorter than 'states' length ({states_len}).\n"
                        f"    Missing {missing_count} label(s).\n"
                        f"    Your states: {self.states}\n"
                        f"    Your labels: {self.labels}\n"
                        f"    Hint: You need to provide {states_len} labels, one for each state.\n"
                        f"    Example: labels = {[str(s) for s in self.states]}"
                    )
                else:
                    extra_count = labels_len - states_len
                    error_msg = (
                        f"[!] 'labels' length ({labels_len}) is longer than 'states' length ({states_len}).\n"
                        f"    You have {extra_count} extra label(s).\n"
                        f"    Your states: {self.states}\n"
                        f"    Your labels: {self.labels}\n"
                        f"    Hint: You should provide exactly {states_len} labels, one for each state.\n"
                        f"    The extra labels are: {self.labels[states_len:]}"
                    )
                
                raise ValueError(error_msg)

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
        """Handles missing values based on the specified rules and user-defined missing_values."""
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

        # Collect all detected missing value indicators
        detected_missing = []
        
        # Check for pandas NaN values
        has_pandas_nan = self.seqdata.isna().any().any()
        if has_pandas_nan:
            detected_missing.append("NaN (pandas)")
        
        # Check for user-specified missing_values in the data
        user_missing_found = []
        for mv in self.missing_values:
            if pd.isna(mv):
                # Handle NaN in missing_values list
                if has_pandas_nan and "NaN (pandas)" not in user_missing_found:
                    user_missing_found.append("NaN (pandas)")
            else:
                # Check if this missing value exists in the data
                if (self.seqdata == mv).any().any():
                    user_missing_found.append(mv)
        
        # Check for string "Missing" (case-insensitive) as missing indicator
        # This handles cases where missing values are represented as the string "Missing" instead of NaN
        # Only check if not already in user-specified missing_values
        has_string_missing = False
        string_missing_variants = []
        
        # Check if "Missing" (case-insensitive) is already in user-specified missing_values
        has_missing_string_in_user_spec = any(
            isinstance(mv, str) and mv.lower() == 'missing' for mv in self.missing_values
        )
        
        if not has_missing_string_in_user_spec:
            try:
                # Check case-insensitive "missing" strings
                missing_mask = self.seqdata.astype(str).str.lower() == 'missing'
                if missing_mask.any().any():
                    has_string_missing = True
                    # Find actual string values (preserving case)
                    actual_values = self.seqdata[missing_mask].dropna().unique()
                    string_missing_variants = [str(v) for v in actual_values if str(v).lower() == 'missing']
            except (AttributeError, TypeError):
                # If conversion fails, check column by column
                try:
                    for col in self.seqdata.columns:
                        col_mask = self.seqdata[col].astype(str).str.lower() == 'missing'
                        if col_mask.any():
                            has_string_missing = True
                            actual_values = self.seqdata.loc[col_mask, col].unique()
                            for v in actual_values:
                                variant = str(v)
                                if variant.lower() == 'missing' and variant not in string_missing_variants:
                                    string_missing_variants.append(variant)
                except:
                    pass
        
        if has_string_missing:
            # Add unique string variants to detected missing (only if not already specified by user)
            for variant in string_missing_variants:
                if variant not in detected_missing and variant not in user_missing_found:
                    detected_missing.append(variant)
        
        # Combine user-specified and auto-detected missing values
        all_missing_values = list(set(self.missing_values + detected_missing))
        # Remove NaN placeholders and add actual NaN check
        if has_pandas_nan:
            all_missing_values = [mv for mv in all_missing_values if mv != "NaN (pandas)"] + [np.nan]
        else:
            all_missing_values = [mv for mv in all_missing_values if mv != "NaN (pandas)"]
        
        # Check if there are any missing values at all
        has_any_missing = False
        if has_pandas_nan:
            has_any_missing = True
        elif user_missing_found:
            has_any_missing = True
        elif has_string_missing:
            has_any_missing = True
        else:
            # Check if any user-specified missing_values exist in data
            for mv in self.missing_values:
                if not pd.isna(mv):
                    if (self.seqdata == mv).any().any():
                        has_any_missing = True
                        break
        
        self.ismissing = has_any_missing
        
        # Warn user if other missing values were detected beyond what they specified
        if self.missing_values and detected_missing:
            other_missing = [mv for mv in detected_missing if mv not in [str(m) for m in self.missing_values] and mv != "NaN (pandas)"]
            if other_missing or (has_pandas_nan and not any(pd.isna(mv) for mv in self.missing_values)):
                print(
                    f"[!] Warning: Detected additional missing value indicators in your data beyond those you specified.\n"
                    f"    You specified: {self.missing_values}\n"
                    f"    Additional missing values found: {other_missing + (['NaN'] if has_pandas_nan and not any(pd.isna(mv) for mv in self.missing_values) else [])}\n"
                    f"    Recommendation: Include these in the `missing_values` parameter for complete handling.\n"
                    f"    Example: missing_values={self.missing_values + other_missing + (['NaN'] if has_pandas_nan and not any(pd.isna(mv) for mv in self.missing_values) else [])}"
                )
        
        # Determine the canonical missing representation for states/labels
        # This will be used when adding missing to states if needed
        canonical_missing_value = None
        if has_pandas_nan:
            canonical_missing_value = np.nan
        elif string_missing_variants:
            # Use the first variant (usually "Missing")
            canonical_missing_value = string_missing_variants[0]
        elif user_missing_found:
            # Use the first user-specified missing value that was found
            canonical_missing_value = user_missing_found[0]
        elif self.missing_values:
            # Use the first user-specified missing value
            canonical_missing_value = self.missing_values[0]
        
        if self.ismissing:
            # Check if states already contains any form of "Missing" or np.nan
            # Check if states contains any representation of missing values
            has_missing_state = False
            for state in self.states:
                if pd.isna(state):
                    has_missing_state = True
                    break
                elif isinstance(state, str):
                    # Check if state matches any missing value (case-insensitive for strings)
                    state_lower = state.lower()
                    if state_lower == "missing" or state in self.missing_values or state in user_missing_found:
                        has_missing_state = True
                        break
                elif state in self.missing_values or state in user_missing_found:
                    has_missing_state = True
                    break
            
            # Also check labels
            has_missing_label = any(
                label.lower() == "missing" or label in self.missing_values or label in user_missing_found
                for label in self.labels if isinstance(label, str)
            ) or any(pd.isna(label) for label in self.labels)
            
            if not has_missing_state and canonical_missing_value is not None:
                # Automatically determine if states are string type or numeric type
                if pd.isna(canonical_missing_value):
                    example_missing = "np.nan"
                    quote = ""
                    missing_state_value = np.nan
                else:
                    example_missing = f"'{canonical_missing_value}'" if isinstance(canonical_missing_value, str) else str(canonical_missing_value)
                    quote = "'" if isinstance(canonical_missing_value, str) else ""
                    missing_state_value = canonical_missing_value

                # Build description of missing types found
                missing_types = []
                if has_pandas_nan:
                    missing_types.append("NaN (pandas)")
                if string_missing_variants:
                    missing_types.extend([f"'{v}'" for v in string_missing_variants])
                if user_missing_found:
                    missing_types.extend([str(v) for v in user_missing_found if v not in string_missing_variants and not pd.isna(v)])
                missing_type_desc = ", ".join(missing_types) if missing_types else "missing values"
                
                missing_values_desc = ""
                if self.missing_values:
                    missing_values_desc = f"\n    You specified missing_values={self.missing_values}."
                
                print(
                    f"[!] Detected missing values ({missing_type_desc}) in the sequence data.{missing_values_desc}\n"
                    f"    -> Automatically added {example_missing} to `states` and `labels` for compatibility.\n"
                    "    However, it's strongly recommended to manually include it when defining `states` and `labels`.\n"
                    "    For example:\n\n"
                    f"        states = [{quote}At Home{quote}, {quote}Left Home{quote}, {example_missing}]\n"
                    f"        labels = [{quote}At Home{quote}, {quote}Left Home{quote}, {quote}Missing{quote}]\n\n"
                    "    This ensures consistent color mapping and avoids unexpected visualization errors."
                )

                # Add missing to states
                self.states.append(missing_state_value)
                    
                # Always ensure labels has the same length as states after appending missing state
                # Strategy: 
                # 1. If labels already has "Missing", we need to ensure it's removed and re-added at the end
                # 2. We need to preserve labels for the original states (before adding missing)
                # 3. If labels length matches original states length, just replace any "Missing" and append
                # 4. If labels has extra elements, take only the first N (where N = original states count)
                
                # Remove any existing "Missing" labels (case-insensitive)
                labels_without_missing = [label for label in self.labels
                                          if not (isinstance(label, str) and label.lower() == "missing")]
                
                # Ensure we have the correct number of labels for non-missing states
                # If labels_without_missing has fewer elements than original states, we're missing some labels
                # If it has more, we take only the first N that match original states
                if len(labels_without_missing) < self._original_num_states:
                    # Not enough labels - this is unusual but we'll pad with generic labels
                    while len(labels_without_missing) < self._original_num_states:
                        labels_without_missing.append(f"State {len(labels_without_missing) + 1}")
                elif len(labels_without_missing) > self._original_num_states:
                    # Too many labels - take only the first N
                    labels_without_missing = labels_without_missing[:self._original_num_states]
                
                # Append "Missing" label at the end to match the appended missing state
                self.labels = labels_without_missing + ["Missing"]
                
                # Verify lengths match (safety check)
                if len(self.states) != len(self.labels):
                    raise ValueError(
                        f"Internal error: Length mismatch after adding missing state. "
                        f"States length: {len(self.states)}, Labels length: {len(self.labels)}. "
                        f"States: {self.states}, Labels: {self.labels}. "
                        f"Original num states: {self._original_num_states}"
                    )
                
                # Mark that Missing was automatically added
                self._missing_auto_added = True
                

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
        # Keep the inverse mapping so that legends and plots can use numeric encoding
        self.inverse_state_mapping = {v: k for k, v in self.state_mapping.items()}

        # Apply the mapping
        # Handle missing values: replace with the last index (which should be the missing state)
        # Also handle user-specified missing_values that might not be in state_mapping
        def map_value(x):
            # First check if it's in the state mapping
            if x in self.state_mapping:
                return self.state_mapping[x]
            # Check if it's a pandas NaN
            if pd.isna(x):
                return len(self.states)  # Last state should be missing
            # Check if it's in user-specified missing_values
            if x in self.missing_values or str(x).lower() == 'missing':
                # If missing value is in states, use its mapping; otherwise use last index
                if x in self.states:
                    return self.state_mapping.get(x, len(self.states))
                else:
                    return len(self.states)
            # If not found, use last index as fallback (treat as missing)
            return len(self.states)
        
        try:
            self.seqdata = self.seqdata.map(map_value)
        except AttributeError:
            self.seqdata = self.seqdata.applymap(map_value)

        if self.ids is not None:
            self.seqdata.index = self.ids

    def _assign_colors(self, reverse_colors=True):
        """Assigns a color palette using user-defined or default Spectral palette.
        
        If missing values are present, automatically assigns a fixed gray color (#cfcccc)
        to missing values and uses the existing color scheme for non-missing states.
        """
        num_states = len(self.states)
        
        # Check if missing values are present
        has_missing = self.ismissing
        missing_gray_color = (0.811765, 0.8, 0.8)  # Fixed gray color for missing values (#cfcccc)
        
        if has_missing:
            # Count non-missing states for color palette generation
            non_missing_states = num_states - 1
            
            if self.custom_colors:
                # If user provided custom colors, check if they account for missing values
                if len(self.custom_colors) == num_states:
                    # User provided colors for all states including missing - use as is
                    color_list = self.custom_colors
                elif len(self.custom_colors) == non_missing_states:
                    # User provided colors only for non-missing states - add gray for missing
                    color_list = self.custom_colors + [missing_gray_color]
                    if self._missing_auto_added:
                        print(
                            f"[!] Automatically added gray color (#cfcccc) for missing values.\n"
                            f"    -> You provided {len(self.custom_colors)} colors for {self._original_num_states} states, "
                            f"but Missing was automatically added.\n"
                            f"    -> Added gray (#cfcccc) as the color for Missing state."
                        )
                elif self._missing_auto_added and len(self.custom_colors) == self._original_num_states:
                    # Missing was automatically added, and user provided colors for original states
                    # Automatically add gray for the missing state
                    color_list = self.custom_colors + [missing_gray_color]
                    print(
                        f"[!] Automatically added gray color (#cfcccc) for missing values.\n"
                        f"    -> You provided {len(self.custom_colors)} colors for {self._original_num_states} states, "
                        f"but Missing was automatically added.\n"
                        f"    -> Added gray (#cfcccc) as the color for Missing state."
                    )
                else:
                    raise ValueError(
                        f"Length of custom_colors ({len(self.custom_colors)}) must match "
                        f"either total states ({num_states}) or non-missing states ({non_missing_states}).\n"
                        f"Hint: If Missing was automatically added, you can either:\n"
                        f"  1. Include 'Missing' in your states and labels when creating SequenceData, or\n"
                        f"  2. Provide {non_missing_states} colors (without Missing) and we'll add gray automatically."
                    )
            else:
                # Generate colors for non-missing states and add gray for missing
                if non_missing_states <= 20:
                    non_missing_color_list = sns.color_palette("Spectral", non_missing_states)
                else:
                    # Use a more elegant color palette for many states - combination of viridis and pastel colors
                    if non_missing_states <= 40:
                        # Use viridis for up to 40 states (more colorful than cubehelix)
                        non_missing_color_list = sns.color_palette("viridis", non_missing_states)
                    else:
                        # For very large state counts, use a custom palette combining multiple schemes
                        viridis_colors = sns.color_palette("viridis", min(non_missing_states // 2, 20))
                        pastel_colors = sns.color_palette("Set3", min(non_missing_states // 2, 12))
                        tab20_colors = sns.color_palette("tab20", min(non_missing_states // 3, 20))
                        
                        # Combine and extend the palette
                        combined_colors = viridis_colors + pastel_colors + tab20_colors
                        # If we need more colors, cycle through the combined palette
                        while len(combined_colors) < non_missing_states:
                            combined_colors.extend(combined_colors[:min(len(combined_colors), non_missing_states - len(combined_colors))])
                        
                        non_missing_color_list = combined_colors[:non_missing_states]

                if reverse_colors:
                    non_missing_color_list = list(reversed(non_missing_color_list))
                
                # Add fixed gray color for missing values at the end
                color_list = list(non_missing_color_list) + [missing_gray_color]
        else:
            # No missing values - use original logic
            if self.custom_colors:
                if len(self.custom_colors) != num_states:
                    raise ValueError("Length of custom_colors must match number of states.")
                color_list = self.custom_colors
            else:
                if num_states <= 20:
                    color_list = sns.color_palette("Spectral", num_states)
                else:
                    # Use a more elegant color palette for many states - combination of viridis and pastel colors
                    if num_states <= 40:
                        # Use viridis for up to 40 states (more colorful than cubehelix)
                        color_list = sns.color_palette("viridis", num_states)
                    else:
                        # For very large state counts, use a custom palette combining multiple schemes
                        viridis_colors = sns.color_palette("viridis", min(num_states // 2, 20))
                        pastel_colors = sns.color_palette("Set3", min(num_states // 2, 12))
                        tab20_colors = sns.color_palette("tab20", min(num_states // 3, 20))
                        
                        # Combine and extend the palette
                        combined_colors = viridis_colors + pastel_colors + tab20_colors
                        # If we need more colors, cycle through the combined palette
                        while len(combined_colors) < num_states:
                            combined_colors.extend(combined_colors[:min(len(combined_colors), num_states - len(combined_colors))])
                        
                        color_list = combined_colors[:num_states]

                if reverse_colors:
                    color_list = list(reversed(color_list))

        # Apply additional_colors if specified (assign custom colors to specific states while keeping default colors)
        if self.additional_colors:
            color_list = list(color_list)  # Make a copy to avoid modifying original
            for state, custom_color in self.additional_colors.items():
                if state in self.states:
                    state_index = self.states.index(state)
                    # Convert hex string to RGB tuple if needed
                    if isinstance(custom_color, str) and custom_color.startswith('#'):
                        # Convert hex to RGB tuple (values 0-1)
                        hex_color = custom_color.lstrip('#')
                        rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                        color_list[state_index] = rgb
                    elif isinstance(custom_color, (tuple, list)) and len(custom_color) == 3:
                        # If RGB values are 0-255, convert to 0-1
                        if all(0 <= v <= 255 for v in custom_color):
                            color_list[state_index] = tuple(v / 255.0 for v in custom_color)
                        else:
                            # Assume already 0-1 range
                            color_list[state_index] = tuple(custom_color)
                    else:
                        color_list[state_index] = custom_color

        # self.color_map = {state: color_list[i] for i, state in enumerate(self.states)}
        # This way all color map keys are 1, 2, 3..., which aligns with imshow(vmin=1, vmax=N)
        self.color_map = {i + 1: color_list[i] for i in range(num_states)}

        # Construct color_map with label as key (for legend)
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
        
        # Display weights information if weights were originally provided
        if self._weights_provided:
            weight_mean = np.mean(self.weights)
            weight_std = np.std(self.weights)
            print(f"[>] Weights: Provided (total weight={sum(self.weights):.3f}, mean={weight_mean:.3f}, std={weight_std:.3f})")
        else:
            print(f"[>] Weights: Not provided")

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
        # Ensure legend handles exist even if get_legend() wasn't called
        legend_handles = getattr(self, "legend_handles", None)
        if not legend_handles:
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, color=self.color_map[i + 1], label=self.labels[i]
                              ) for i in range(len(self.states))
            ]
            self.legend_handles = legend_handles

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.legend(handles=legend_handles, loc='center', title="States", fontsize=10)
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
    def sequences(self):
        """Returns sequences as a list of lists (one list per sequence)."""
        return [list(row) for row in self.seqdata.values]

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
        E.g., 5 sequences x 10 steps -> repeat each weight 10 times.
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

    def check_uniqueness_rate(self, weighted: bool = False):
        """
        Compute uniqueness statistics of the sequences.

        Returns:
            dict with keys:
                - n_sequences: total number of sequences (unweighted count)
                - n_unique: number of unique sequence patterns
                - uniqueness_rate: n_unique / n_sequences
                - weighted_total: total weighted count (only if weighted=True)
                - weighted_uniqueness_rate: n_unique / weighted_total (only if weighted=True)
                
        Parameters:
            weighted: if True, use sequence weights to calculate weighted frequencies and uniqueness rates;
                     if False, use simple counts (default behavior for backward compatibility).
        """
        import numpy as np
        import pandas as pd

        A = self.to_numeric()                       # shape (n, m), int32
        n, m = A.shape

        # Use a byte-level view to let np.unique work row-wise efficiently
        A_contig = np.ascontiguousarray(A)
        row_view = A_contig.view(np.dtype((np.void, A_contig.dtype.itemsize * m))).ravel()

        # Get unique patterns 
        uniq, inverse = np.unique(row_view, return_inverse=True)

        n_unique = uniq.size
        uniqueness_rate = float(n_unique) / float(n) if n > 0 else np.nan

        # Build simplified result dictionary with only essential statistics
        result = {
            "n_sequences": int(n),
            "n_unique": int(n_unique),
            "uniqueness_rate": uniqueness_rate
        }

        # Add weighted statistics if requested
        if weighted:
            weighted_total = float(np.sum(self.weights))
            weighted_uniqueness_rate = float(n_unique) / weighted_total if weighted_total > 0 else np.nan
            result["weighted_total"] = weighted_total
            result["weighted_uniqueness_rate"] = weighted_uniqueness_rate

        return result

    def show_color_palette(self, palette_name: str = 'default', save_as: str = None, dpi: int = 200):
        """
        Instance method to show the default color palette for the current number of states.
        This is a convenience method that calls show_default_color_palette() with the number of states
        from this SequenceData instance.
        
        Parameters:
        -----------
        palette_name : str, default='default'
            Name of the color palette to use. See show_default_color_palette() for available options.
        save_as : str, optional
            If provided, save the color preview figure to this file path.
        dpi : int, default=200
            Resolution for saving the figure (if save_as is provided).
        
        Returns:
        --------
        dict : Dictionary with keys:
            - 'colors': List of RGB tuples (0-1 range)
            - 'hex_colors': List of hex color codes (e.g., "#FF5733")
            - 'rgb_255': List of RGB tuples (0-255 range)
        
        Example:
        --------
        # Show color palette for this SequenceData instance
        seq_data = SequenceData(...)
        color_info = seq_data.show_color_palette()
        
        # Show with a specific palette template
        color_info = seq_data.show_color_palette(palette_name='viridis')
        """
        return SequenceData.show_default_color_palette(
            n_states=len(self.states), 
            reverse_colors=True, 
            palette_name=palette_name,
            save_as=save_as, 
            dpi=dpi
        )

    @staticmethod
    def _get_available_palette_names():
        """
        Get list of available color palette names that can be used with show_default_color_palette.
        
        Returns:
        --------
        list : List of available palette names (strings)
        """
        # Common seaborn color palettes
        available_palettes = [
            'default',  # Uses automatic selection based on n_states (Spectral/viridis/combined)
            'Spectral',
            'viridis',
            'Set3',
            'tab20',
            'deep',
            'muted',
            'pastel',
            'bright',
            'dark',
            'colorblind',
            'husl',
            'hls',
            'coolwarm',
            'RdYlGn',
            'RdYlBu',
            'RdBu',
            'PiYG',
            'PRGn',
            'BrBG',
            'Set1',
            'Set2',
            'Paired',
            'Accent',
            'Dark2',
        ]
        return available_palettes
    
    @staticmethod
    def _generate_color_list(n_states: int, palette_name: str = 'default', reverse_colors: bool = True):
        """
        Generate color list based on palette name and number of states.
        
        Parameters:
        -----------
        n_states : int
            Number of states (colors) to generate.
        palette_name : str, default='default'
            Name of the color palette to use. Use 'default' for automatic selection.
            Available palettes: see _get_available_palette_names()
        reverse_colors : bool, default=True
            Whether to reverse the color order.
        
        Returns:
        --------
        list : List of RGB tuples (0-1 range)
        """
        if palette_name == 'default':
            # Use the original logic for default palette selection
            if n_states <= 20:
                color_list = sns.color_palette("Spectral", n_states)
            else:
                if n_states <= 40:
                    color_list = sns.color_palette("viridis", n_states)
                else:
                    viridis_colors = sns.color_palette("viridis", min(n_states // 2, 20))
                    pastel_colors = sns.color_palette("Set3", min(n_states // 2, 12))
                    tab20_colors = sns.color_palette("tab20", min(n_states // 3, 20))
                    combined_colors = viridis_colors + pastel_colors + tab20_colors
                    while len(combined_colors) < n_states:
                        combined_colors.extend(combined_colors[:min(len(combined_colors), n_states - len(combined_colors))])
                    color_list = combined_colors[:n_states]
        else:
            # Use specified palette name
            try:
                color_list = sns.color_palette(palette_name, n_states)
            except ValueError:
                # If palette doesn't support n_states directly, try to generate more colors
                try:
                    # Try to get a base palette and extend it
                    base_palette = sns.color_palette(palette_name)
                    color_list = []
                    while len(color_list) < n_states:
                        color_list.extend(base_palette)
                    color_list = color_list[:n_states]
                except Exception as e:
                    raise ValueError(f"Invalid palette name '{palette_name}'. Available palettes: {', '.join(SequenceData._get_available_palette_names())}") from e
        
        if reverse_colors:
            color_list = list(reversed(color_list))
        
        return color_list
    
    @staticmethod
    def _convert_rgb_to_hex_and_255(color_list):
        """
        Convert list of RGB tuples (0-1 range) to hex codes and RGB (0-255 range).
        
        Parameters:
        -----------
        color_list : list
            List of RGB tuples in 0-1 range.
        
        Returns:
        --------
        tuple : (hex_colors, rgb_255_list) where:
            - hex_colors: List of hex color codes (e.g., "#FF5733")
            - rgb_255_list: List of RGB tuples (0-255 range)
        """
        hex_colors = []
        rgb_255_list = []
        for rgb in color_list:
            # Convert from 0-1 to 0-255
            rgb_255 = tuple(int(c * 255) for c in rgb)
            rgb_255_list.append(rgb_255)
            # Convert to hex
            hex_color = f"#{rgb_255[0]:02X}{rgb_255[1]:02X}{rgb_255[2]:02X}"
            hex_colors.append(hex_color)
        return hex_colors, rgb_255_list
    
    @staticmethod
    def show_default_color_palette(n_states: int, reverse_colors: bool = True, palette_name: str = 'default', save_as: str = None, dpi: int = 200):
        """
        Display the default color palette that would be used for a given number of states.
        This is useful for viewing default colors and copying hex codes to create custom_colors.
        
        Parameters:
        -----------
        n_states : int
            Number of states (colors) to generate.
        reverse_colors : bool, default=True
            Whether to reverse the color order (same as default behavior in SequenceData).
        palette_name : str, default='default'
            Name of the color palette to use. Use 'default' to use the automatic palette selection
            (Spectral for ≤20 states, viridis for 21-40 states, combined for >40 states).
            Available palettes: 'default', 'Spectral', 'viridis', 'Set3', 'tab20', 'deep', 'muted', 
            'pastel', 'bright', 'dark', 'colorblind', 'husl', 'hls', 'coolwarm', 'RdYlGn', 'RdYlBu',
            'RdBu', 'PiYG', 'PRGn', 'BrBG', 'Set1', 'Set2', 'Paired', 'Accent', 'Dark2', etc.
            Call SequenceData._get_available_palette_names() to see all available options.
        save_as : str, optional
            If provided, save the color preview figure to this file path.
        dpi : int, default=200
            Resolution for saving the figure (if save_as is provided).
        
        Returns:
        --------
        dict : Dictionary with keys:
            - 'colors': List of RGB tuples (0-1 range)
            - 'hex_colors': List of hex color codes (e.g., "#FF5733")
            - 'rgb_255': List of RGB tuples (0-255 range)
        
        Example:
        --------
        # View default colors for 13 states (call via class)
        color_info = SequenceData.show_default_color_palette(13)
        
        # View a specific palette template
        color_info = SequenceData.show_default_color_palette(13, palette_name='viridis')
        
        # Or call via instance (which will use the instance's number of states)
        seq_data = SequenceData(...)
        color_info = seq_data.show_color_palette()
        
        # Then you can copy the hex_colors to use as custom_colors
        custom_colors = color_info['hex_colors']
        """
        # Generate colors using the specified palette
        color_list = SequenceData._generate_color_list(n_states, palette_name, reverse_colors)
        
        # Convert RGB (0-1) to hex and RGB (0-255)
        hex_colors, rgb_255_list = SequenceData._convert_rgb_to_hex_and_255(color_list)
        
        # Print header with palette information
        print(f"\n{'='*80}")
        palette_display_name = "Default (automatic selection)" if palette_name == 'default' else palette_name
        print(f"Color Palette: {palette_display_name} for {n_states} States")
        print(f"{'='*80}\n")
        
        # Show available palette names if using default
        if palette_name == 'default':
            available_palettes = SequenceData._get_available_palette_names()
            print("Available color palette templates:")
            print("  " + ", ".join(available_palettes))
            print("\n  You can specify a palette template by using the 'palette_name' parameter.")
            print("  Example: show_default_color_palette(13, palette_name='viridis')\n")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, max(6, n_states * 0.5)))
        
        for i, (hex_color, rgb, rgb_255) in enumerate(zip(hex_colors, color_list, rgb_255_list)):
            # Draw color swatch
            y_pos = n_states - i - 1
            rect = plt.Rectangle((0, y_pos), 1, 0.8, facecolor=rgb, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(1.1, y_pos + 0.4, f"{i+1:2d}. {hex_color} | RGB{rgb_255}", 
                   va='center', fontsize=10, fontfamily='monospace')
        
        print(f"{'='*80}")
        print("\nTo use these colors as custom_colors, copy the hex codes:")
        print("  custom_colors = " + str(hex_colors))
        print("\nOr use additional_colors to assign custom colors to specific states:")
        print("  additional_colors = {'Other': '#BDBDBD'}  # Assign gray color to 'Other' state")
        print(f"{'='*80}\n")
        
        # Configure plot
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.5, n_states)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        title_text = f"Color Palette: {palette_display_name} ({n_states} States)"
        ax.set_title(title_text, fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_as:
            plt.savefig(save_as, dpi=dpi, bbox_inches='tight')
            print(f"[>] Color palette saved to: {save_as}")
        
        plt.show()
        
        return {
            'colors': color_list,  # RGB tuples (0-1 range)
            'hex_colors': hex_colors,  # Hex codes
            'rgb_255': rgb_255_list  # RGB tuples (0-255 range)
        }

    @staticmethod
    def get_default_color_palette(n_states: int, reverse_colors: bool = True, palette_name: str = 'default', return_format: str = 'hex'):
        """
        Get the default color palette for a given number of states.
        This returns the colors without displaying them (useful for programmatic use).
        
        Parameters:
        -----------
        n_states : int
            Number of states (colors) to generate.
        reverse_colors : bool, default=True
            Whether to reverse the color order (same as default behavior in SequenceData).
        palette_name : str, default='default'
            Name of the color palette to use. See show_default_color_palette() for available options.
        return_format : str, default='hex'
            Format to return colors in. Options:
            - 'hex': List of hex color codes (e.g., "#FF5733")
            - 'rgb': List of RGB tuples (0-1 range, for matplotlib)
            - 'rgb255': List of RGB tuples (0-255 range)
        
        Returns:
        --------
        list : List of colors in the requested format.
        
        Example:
        --------
        # Get hex codes for 13 states using default palette
        hex_colors = SequenceData.get_default_color_palette(13, return_format='hex')
        
        # Get hex codes using a specific palette template
        hex_colors = SequenceData.get_default_color_palette(13, palette_name='viridis', return_format='hex')
        
        # Use them as custom_colors
        seq = SequenceData(df, time=..., states=..., custom_colors=hex_colors)
        """
        # Generate colors using the specified palette
        color_list = SequenceData._generate_color_list(n_states, palette_name, reverse_colors)
        
        if return_format == 'rgb':
            return color_list
        elif return_format == 'hex':
            hex_colors, _ = SequenceData._convert_rgb_to_hex_and_255(color_list)
            return hex_colors
        elif return_format == 'rgb255':
            _, rgb_255_list = SequenceData._convert_rgb_to_hex_and_255(color_list)
            return rgb_255_list
        else:
            raise ValueError(f"return_format must be 'hex', 'rgb', or 'rgb255', got '{return_format}'")





