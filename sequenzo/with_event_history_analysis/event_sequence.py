"""
@Author  : Yuqi Liang 梁彧祺
@File    : event_sequence.py
@Time    : 09/02/2026 11:14
@Desc    : Event Sequence Analysis 

What is Event Sequence Analysis?
--------------------------------
Event sequence analysis focuses on "when changes occur" rather than "what states persist".
Unlike state sequences that record persistent states at each time point, event sequences
record discrete events (changes) with timestamps. This allows us to:
- Discover frequent event patterns (e.g., "Graduate → Find Job → Promotion")
- Compare event patterns between groups (e.g., male vs female career paths)
- Analyze the timing and ordering of events
- Identify discriminating subsequences that distinguish different populations

Function Mapping: Sequenzo ↔ TraMineR
--------------------------------------
Sequenzo Function Name              TraMineR Equivalent    Description
----------------------------         -------------------    -----------
create_event_sequences()             seqecreate()          Create event sequence objects from TSE data or state sequences
find_frequent_subsequences()         seqefsub()            Find frequently occurring event patterns above a support threshold
count_subsequence_occurrences()      seqeapplysub()        Count how many times each subsequence appears in each sequence
compare_groups()                     seqecmpgroup()        Compare groups to find discriminating subsequences (chi-square tests)
plot_event_sequences()                plot.seqelist()       Visualize event sequences (see event_sequence_visualization.py)
plot_subsequence_frequencies()       plot.subseqelist()     Plot bar chart of subsequence frequencies (see event_sequence_visualization.py)

Key Classes:
------------
- EventSequence: Represents a single event sequence for one individual
- EventSequenceList: Collection of event sequences (equivalent to TraMineR's seqelist)
- EventSequenceConstraint: Time constraints for subsequence search (age windows, gaps, etc.)
- SubsequenceList: List of frequent subsequences with metadata (equivalent to TraMineR's subseqelist)

Example Usage:
--------------
    >>> import pandas as pd
    >>> from sequenzo.with_event_history_analysis import create_event_sequences, find_frequent_subsequences
    >>> 
    >>> # Create event sequences from TSE format data
    >>> tse_data = pd.DataFrame({
    ...     'id': [1, 1, 2, 2],
    ...     'timestamp': [18, 22, 17, 20],
    ...     'event': ['EnterUni', 'Graduate', 'EnterUni', 'Graduate']
    ... })
    >>> eseq = create_event_sequences(data=tse_data)
    >>> 
    >>> # Find frequent subsequences
    >>> fsubseq = find_frequent_subsequences(eseq, min_support=2)
    >>> print(fsubseq.data)

References:
-----------
- TraMineR Documentation: http://traminer.unige.ch
- Ritschard, G., Bürgin, R., & Studer, M. (2013). 
- Exploratory mining of life event histories. In Contemporary issues in exploratory data mining in the behavioral sciences (pp. 221-253). Routledge.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Tuple
from collections import defaultdict
import warnings

# Try to import scipy for statistical tests
try:
    from scipy.stats import chi2_contingency
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. compare_groups will not work properly.", ImportWarning)


class EventSequence:
    """
    Represents a single event sequence for one individual.
    
    Attributes:
        id: Individual identifier
        timestamps: Array of event timestamps
        events: Array of event codes (integers)
        dictionary: Event dictionary mapping codes to names
    """
    
    def __init__(self, id: int, timestamps: np.ndarray, events: np.ndarray, 
                 dictionary: List[str]):
        """
        Initialize an event sequence.
        
        Args:
            id: Individual identifier
            timestamps: Array of event timestamps (sorted)
            events: Array of event codes (integers)
            dictionary: List of event names (alphabet)
        """
        self.id = id
        self.timestamps = np.array(timestamps, dtype=np.float64)
        self.events = np.array(events, dtype=np.int32)
        self.dictionary = dictionary
        
        # Ensure timestamps and events are sorted
        if len(self.timestamps) > 0:
            sort_idx = np.argsort(self.timestamps)
            self.timestamps = self.timestamps[sort_idx]
            self.events = self.events[sort_idx]
    
    def __len__(self):
        """Return the number of events in the sequence."""
        return len(self.timestamps)
    
    def __repr__(self):
        """String representation of the event sequence."""
        if len(self.timestamps) == 0:
            return f"EventSequence(id={self.id}, empty)"
        
        event_strs = []
        for t, e in zip(self.timestamps, self.events):
            event_name = self.dictionary[e - 1] if 0 < e <= len(self.dictionary) else f"Event{e}"
            event_strs.append(f"{t}:{event_name}")
        
        return f"EventSequence(id={self.id}, events=[{', '.join(event_strs)}])"
    
    def to_string(self) -> str:
        """Convert event sequence to string format compatible with TraMineR."""
        if len(self.timestamps) == 0:
            return ""
        
        parts = []
        for i, (t, e) in enumerate(zip(self.timestamps, self.events)):
            event_name = self.dictionary[e - 1] if 0 < e <= len(self.dictionary) else f"Event{e}"
            if i == 0:
                parts.append(f"({event_name})")
            else:
                gap = t - self.timestamps[i-1]
                if gap == 0:
                    # Simultaneous event
                    parts[-1] = parts[-1].rstrip(')') + f",{event_name})"
                else:
                    parts.append(f"-({event_name})")
        
        return "".join(parts)


class EventSequenceList:
    """
    Represents a collection of event sequences.
    
    This is the Python equivalent of TraMineR's seqelist class.
    """
    
    def __init__(self, sequences: List[EventSequence], dictionary: List[str],
                 weights: Optional[np.ndarray] = None, lengths: Optional[np.ndarray] = None):
        """
        Initialize an event sequence list.
        
        Args:
            sequences: List of EventSequence objects
            dictionary: Event dictionary (alphabet)
            weights: Optional weights for each sequence
            lengths: Optional sequence lengths
        """
        self.sequences = sequences
        self.dictionary = dictionary
        self.n_sequences = len(sequences)
        
        if weights is None:
            self.weights = np.ones(self.n_sequences, dtype=np.float64)
        else:
            self.weights = np.array(weights, dtype=np.float64)
            if len(self.weights) != self.n_sequences:
                raise ValueError(f"Weights length ({len(self.weights)}) must match number of sequences ({self.n_sequences})")
        
        if lengths is None:
            self.lengths = np.array([len(seq) for seq in sequences], dtype=np.float64)
        else:
            self.lengths = np.array(lengths, dtype=np.float64)
    
    def __len__(self):
        """Return the number of sequences."""
        return self.n_sequences
    
    def __getitem__(self, idx):
        """Get a sequence by index."""
        return self.sequences[idx]
    
    def __iter__(self):
        """Iterate over sequences."""
        return iter(self.sequences)
    
    def get_total_weight(self) -> float:
        """Get the total weight of all sequences."""
        return np.sum(self.weights)
    
    def get_dictionary(self) -> List[str]:
        """Get the event dictionary."""
        return self.dictionary.copy()
    
    def __repr__(self):
        """String representation."""
        return f"EventSequenceList(n={self.n_sequences}, events={len(self.dictionary)})"


class EventSequenceConstraint:
    """
    Time constraints for event sequence analysis.
    
    This is the Python equivalent of TraMineR's seqeconstraint class.
    """
    
    def __init__(self, max_gap: float = -1, window_size: float = -1,
                 age_min: float = -1, age_max: float = -1, age_max_end: float = -1,
                 count_method: Union[int, str] = 1):
        """
        Initialize event sequence constraints.
        
        Args:
            max_gap: Maximum time gap between consecutive events in a subsequence (-1 = no limit)
            window_size: Maximum time window for a subsequence (-1 = no limit)
            age_min: Minimum age/time for events (-1 = no limit)
            age_max: Maximum age/time for events (-1 = no limit)
            age_max_end: Maximum age/time for subsequence end (-1 = no limit)
            count_method: Counting method (1=COBJ, 2=CDIST_O, 3=CWIN, 4=CMINWIN, 5=CDIST)
        """
        # Validate constraints
        if age_max_end != -1 and age_max == -1:
            age_max = age_max_end
        
        if max_gap != -1 and window_size != -1 and max_gap > window_size:
            raise ValueError("max_gap cannot be greater than window_size")
        
        if age_min != -1 and age_max != -1 and age_min > age_max:
            raise ValueError("age_min cannot be greater than age_max")
        
        # Convert count_method string to integer
        if isinstance(count_method, str):
            count_method_map = {
                "COBJ": 1,
                "CDIST_O": 2,
                "CWIN": 3,
                "CMINWIN": 4,
                "CDIST": 5
            }
            if count_method not in count_method_map:
                raise ValueError(f"Unknown count_method: {count_method}")
            count_method = count_method_map[count_method]
        
        if count_method not in [1, 2, 3, 4, 5]:
            raise ValueError(f"count_method must be 1-5, got {count_method}")
        
        self.max_gap = max_gap
        self.window_size = window_size
        self.age_min = age_min
        self.age_max = age_max
        self.age_max_end = age_max_end
        self.count_method = count_method
    
    def __repr__(self):
        """String representation of constraints."""
        parts = []
        if self.max_gap != -1:
            parts.append(f"max_gap={self.max_gap}")
        if self.window_size != -1:
            parts.append(f"window_size={self.window_size}")
        if self.age_min != -1:
            parts.append(f"age_min={self.age_min}")
        if self.age_max != -1:
            parts.append(f"age_max={self.age_max}")
        if self.age_max_end != -1:
            parts.append(f"age_max_end={self.age_max_end}")
        
        count_names = {1: "COBJ", 2: "CDIST_O", 3: "CWIN", 4: "CMINWIN", 5: "CDIST"}
        parts.append(f"count_method={count_names.get(self.count_method, self.count_method)}")
        
        return f"EventSequenceConstraint({', '.join(parts)})"


class SubsequenceList:
    """
    Represents a list of subsequences with metadata.
    
    This is the Python equivalent of TraMineR's subseqelist class.
    """
    
    def __init__(self, eseq: EventSequenceList,
                 subsequences: List[EventSequence],
                 data: pd.DataFrame,
                 constraint: EventSequenceConstraint,
                 type: str = "frequent"):
        """
        Initialize a subsequence list.
        
        Args:
            eseq: Original event sequence list
            subsequences: List of subsequence EventSequence objects
            data: DataFrame with Support, Count, etc.
            constraint: Constraints used
            type: Type of subsequence list ("frequent", "user", "chisq")
        """
        self.eseq = eseq
        self.subsequences = subsequences
        self.data = data
        self.constraint = constraint
        self.type = type
    
    def __len__(self):
        """Return the number of subsequences."""
        return len(self.subsequences)
    
    def __getitem__(self, idx):
        """Get a subsequence by index."""
        if isinstance(idx, slice):
            return SubsequenceList(
                self.eseq,
                self.subsequences[idx],
                self.data.iloc[idx].reset_index(drop=True),
                self.constraint,
                self.type
            )
        return self.subsequences[idx]
    
    def __repr__(self):
        """String representation."""
        return f"SubsequenceList(n={len(self.subsequences)}, type={self.type})"


# ============================================================================
# Main Functions with Intuitive Names
# ============================================================================

def create_event_sequences(data: Optional[pd.DataFrame] = None,
                           id: Optional[Union[np.ndarray, pd.Series, List]] = None,
                           timestamp: Optional[Union[np.ndarray, pd.Series, List]] = None,
                           event: Optional[Union[np.ndarray, pd.Series, List]] = None,
                           end_event: Optional[str] = None,
                           tevent: Union[str, np.ndarray] = "transition",
                           use_labels: bool = True,
                           weighted: bool = True,
                           alphabet: Optional[List[str]] = None,
                           seqdata=None) -> EventSequenceList:
    """
    Create an event sequence object from state sequences or timestamped event data.
    
    TraMineR equivalent: seqecreate()
    
    Args:
        data: Either a DataFrame with 'id', 'timestamp'/'time', and 'event' columns,
              or a SequenceData object (state sequence) to convert
        id: Individual identifiers (required if data is None or not a DataFrame)
        timestamp: Event timestamps (required if data is None or not a DataFrame)
        event: Event names/types (required if data is None or not a DataFrame)
        end_event: Optional event name indicating end of observation
        tevent: Transition method for state-to-event conversion:
                - "transition": One event per transition (default)
                - "state": One event per state entry
                - "period": Pair of start/end events
                - Or a transition matrix (numpy array)
        use_labels: If True, use state labels instead of codes
        weighted: If True, preserve weights from state sequences
        alphabet: Optional list of event labels in a specific order (e.g. TraMineR alphabet
                  for reference comparison). If provided, the event dictionary uses this order.
        seqdata: Deprecated, use data instead
    
    Returns:
        EventSequenceList object containing event sequences
    
    Examples:
        >>> # From TSE format DataFrame
        >>> tse_data = pd.DataFrame({
        ...     'id': [1, 1, 2, 2],
        ...     'timestamp': [18, 22, 17, 20],
        ...     'event': ['EnterUni', 'Graduate', 'EnterUni', 'Graduate']
        ... })
        >>> eseq = create_event_sequences(data=tse_data)
        
        >>> # From state sequence (SequenceData object)
        >>> eseq = create_event_sequences(data=state_seq, tevent="transition")
    """
    # Handle deprecated parameter
    if seqdata is not None:
        warnings.warn("seqdata parameter is deprecated, use data instead", DeprecationWarning)
        data = seqdata
    
    # Handle data parameter
    if data is not None:
        # Check if it's a SequenceData object (state sequence)
        if hasattr(data, 'seqdata') and hasattr(data, 'states'):
            # Convert state sequence to event sequence
            return _state_to_event_sequence(data, tevent, use_labels, weighted, end_event, alphabet)
        
        # Otherwise, assume it's a DataFrame
        elif isinstance(data, pd.DataFrame):
            if 'id' in data.columns and ('timestamp' in data.columns or 'time' in data.columns) and 'event' in data.columns:
                id = data['id'].values
                event = data['event'].values
                if 'timestamp' in data.columns:
                    timestamp = data['timestamp'].values
                else:
                    timestamp = data['time'].values
            else:
                raise ValueError("DataFrame must contain 'id', 'timestamp'/'time', and 'event' columns")
    
    # Validate required parameters
    if id is None:
        raise ValueError("Could not find an id argument")
    if timestamp is None:
        raise ValueError("Could not find a timestamp argument")
    if event is None:
        raise ValueError("Could not find an event argument")
    
    # Convert to numpy arrays
    id = np.array(id)
    timestamp = np.array(timestamp, dtype=np.float64)
    event = np.array(event)
    
    # Check for missing values
    if np.any(np.isnan(id)) or np.any(np.isnan(timestamp)) or pd.isna(event).any():
        raise ValueError("Missing values not supported in event sequences")
    
    # Create event dictionary
    if isinstance(event, pd.Series):
        event = event.values
    
    unique_events = pd.Series(event).unique()
    dictionary = sorted([str(e) for e in unique_events])
    
    # Check for special characters in event names
    special_chars = ['(', ')', ',']
    for evt in dictionary:
        if any(char in evt for char in special_chars):
            warnings.warn(
                f"Event '{evt}' contains special characters '(', ')', or ','. "
                "Searching for specific subsequences may not work properly.",
                UserWarning
            )
    
    # Convert events to integer codes
    event_to_code = {evt: idx + 1 for idx, evt in enumerate(dictionary)}
    event_codes = np.array([event_to_code[str(e)] for e in event])
    
    # Handle end_event
    end_event_code = 0
    if end_event is not None:
        if end_event in event_to_code:
            end_event_code = event_to_code[end_event]
        else:
            raise ValueError(f"end_event '{end_event}' not found in event dictionary")
    
    # Group events by id
    df = pd.DataFrame({
        'id': id,
        'timestamp': timestamp,
        'event': event_codes
    })
    
    # Sort by id and timestamp
    df = df.sort_values(['id', 'timestamp', 'event'])
    
    # Check if events are grouped by id
    unique_ids = df['id'].unique()
    
    # Create sequences
    sequences = []
    for uid in unique_ids:
        seq_data = df[df['id'] == uid]
        sequences.append(EventSequence(
            id=int(uid),
            timestamps=seq_data['timestamp'].values,
            events=seq_data['event'].values,
            dictionary=dictionary
        ))
    
    # Create EventSequenceList
    eseq_list = EventSequenceList(sequences, dictionary)
    
    return eseq_list


def find_frequent_subsequences(eseq: EventSequenceList,
                               str_subseq: Optional[List[str]] = None,
                               min_support: Optional[float] = None,
                               pmin_support: Optional[float] = None,
                               constraint: Optional[EventSequenceConstraint] = None,
                               max_k: int = -1,
                               weighted: bool = True) -> SubsequenceList:
    """
    Find frequent subsequences in event sequences.
    
    TraMineR equivalent: seqefsub()
    
    Args:
        eseq: EventSequenceList object
        str_subseq: Optional list of specific subsequences to search for (as strings)
        min_support: Minimum support in number of sequences
        pmin_support: Minimum support as proportion (0-1)
        constraint: EventSequenceConstraint object
        max_k: Maximum number of events in subsequence (-1 = no limit)
        weighted: If True, use sequence weights
    
    Returns:
        SubsequenceList object with frequent subsequences
    
    Examples:
        >>> # Find subsequences with at least 20 occurrences
        >>> fsubseq = find_frequent_subsequences(eseq, min_support=20)
        
        >>> # Find subsequences with at least 1% support
        >>> fsubseq = find_frequent_subsequences(eseq, pmin_support=0.01)
        
        >>> # Search for specific subsequences
        >>> fsubseq = find_frequent_subsequences(eseq, str_subseq=["(A)-(B)", "(B)-(C)"])
    """
    if not isinstance(eseq, EventSequenceList):
        raise TypeError("eseq must be an EventSequenceList object")
    
    if constraint is None:
        constraint = EventSequenceConstraint()
    
    if not isinstance(constraint, EventSequenceConstraint):
        warnings.warn("constraint should be an EventSequenceConstraint object. Using default.", UserWarning)
        constraint = EventSequenceConstraint()
    
    # Validate constraint
    if constraint.count_method == 3 and constraint.window_size == -1:
        raise ValueError("CWIN method requires window_size constraint")
    
    # Handle weights
    if weighted:
        total_weight = eseq.get_total_weight()
    else:
        total_weight = float(len(eseq))
        # Temporarily set weights to 1
        original_weights = eseq.weights.copy()
        eseq.weights = np.ones(len(eseq), dtype=np.float64)
    
    try:
        # Handle user-specified subsequences
        if str_subseq is not None:
            return _search_specific_subsequences(eseq, str_subseq, constraint, total_weight, weighted)
        
        # Validate support threshold
        if min_support is None:
            if pmin_support is None:
                raise ValueError("You should specify a minimum support through min_support or pmin_support")
            min_support = pmin_support * total_weight
        
        # Find frequent subsequences
        # This is a simplified implementation - full version would use prefix tree algorithm
        subsequences, supports, counts = _find_frequent_subsequences(
            eseq, min_support, constraint, max_k
        )
        
        # Create data frame
        data = pd.DataFrame({
            'Support': supports / total_weight,
            'Count': counts
        })
        
        # Sort by support (descending)
        sort_idx = np.argsort(supports)[::-1]
        subsequences = [subsequences[i] for i in sort_idx]
        data = data.iloc[sort_idx].reset_index(drop=True)
        
        return SubsequenceList(eseq, subsequences, data, constraint, type="frequent")
    
    finally:
        # Restore original weights if we modified them
        if not weighted:
            eseq.weights = original_weights


def count_subsequence_occurrences(subseq: SubsequenceList,
                                  method: Optional[Union[str, int]] = None,
                                  constraint: Optional[EventSequenceConstraint] = None,
                                  rules: bool = False) -> np.ndarray:
    """
    Count occurrences of subsequences in sequences.
    
    TraMineR equivalent: seqeapplysub()
    
    Args:
        subseq: SubsequenceList object (result from find_frequent_subsequences)
        method: Counting method:
                - "presence" or "COBJ" (1): Count per sequence (0/1)
                - "count" or "CDIST_O" (2): Count distinct occurrences
                - "CWIN" (3): Count within time windows
                - "CMINWIN" (4): Count minimum windows
                - "CDIST" (5): Count with distance constraints
                - Or integer 1-5
        constraint: EventSequenceConstraint object (uses subseq.constraint if None)
        rules: If True, count subsequences within subsequences (for rule mining)
    
    Returns:
        numpy array of shape (n_sequences, n_subsequences) with counts
    
    Examples:
        >>> fsubseq = find_frequent_subsequences(eseq, min_support=10)
        >>> counts = count_subsequence_occurrences(fsubseq, method="presence")
        >>> # counts[i, j] = 1 if sequence i contains subsequence j, else 0
    """
    if not isinstance(subseq, SubsequenceList):
        raise TypeError("subseq must be a SubsequenceList object")
    
    if constraint is None:
        constraint = subseq.constraint
    
    if not isinstance(constraint, EventSequenceConstraint):
        warnings.warn("constraint should be an EventSequenceConstraint object. Using default.", UserWarning)
        constraint = EventSequenceConstraint()
    
    # Handle method parameter
    if method is not None:
        if isinstance(method, str):
            method_map = {
                "presence": 1, "COBJ": 1,
                "count": 2, "CDIST_O": 2,
                "CWIN": 3,
                "CMINWIN": 4,
                "CDIST": 5
            }
            if method not in method_map:
                raise ValueError(f"Unknown method: {method}")
            constraint.count_method = method_map[method]
        elif isinstance(method, int):
            if method not in [1, 2, 3, 4, 5]:
                raise ValueError(f"method must be 1-5, got {method}")
            constraint.count_method = method
    
    # Get sequences to search in
    if rules:
        search_sequences = subseq.subsequences
    else:
        search_sequences = subseq.eseq.sequences
    
    n_seqs = len(search_sequences)
    n_subseqs = len(subseq.subsequences)
    
    # Initialize result matrix
    result = np.zeros((n_seqs, n_subseqs), dtype=np.float64)
    
    # Count occurrences for each subsequence in each sequence
    for j, subseq_seq in enumerate(subseq.subsequences):
        for i, seq in enumerate(search_sequences):
            count = _count_subsequence_in_sequence(
                subseq_seq, seq, constraint
            )
            result[i, j] = count
    
    return result


def compare_groups(subseq: SubsequenceList,
                   group: Union[np.ndarray, pd.Series, List],
                   method: str = "chisq",
                   pvalue_limit: Optional[float] = None,
                   weighted: bool = True) -> SubsequenceList:
    """
    Compare groups to find discriminating subsequences.
    
    TraMineR equivalent: seqecmpgroup()
    
    Args:
        subseq: SubsequenceList object (result from find_frequent_subsequences)
        group: Group membership for each sequence (array-like)
        method: Test method ("chisq" for chi-square test, "bonferroni" for Bonferroni correction)
        pvalue_limit: Maximum p-value threshold (default: 2.0 for display)
        weighted: If True, use sequence weights
    
    Returns:
        SubsequenceList object filtered to discriminating subsequences
    
    Examples:
        >>> fsubseq = find_frequent_subsequences(eseq, min_support=10)
        >>> groups = np.array(['Male', 'Female', 'Male', 'Female', 'Male'])
        >>> discr = compare_groups(fsubseq, groups, pvalue_limit=0.05)
    """
    if not isinstance(subseq, SubsequenceList):
        raise TypeError("subseq must be a SubsequenceList object")
    
    group = np.array(group)
    if len(group) != len(subseq.eseq):
        raise ValueError(f"group length ({len(group)}) must match number of sequences ({len(subseq.eseq)})")
    
    if pvalue_limit is None:
        pvalue_limit = 2.0  # TraMineR default
    
    # Handle weights
    if not weighted:
        original_weights = subseq.eseq.weights.copy()
        subseq.eseq.weights = np.ones(len(subseq.eseq), dtype=np.float64)
        total_weight = float(len(subseq.eseq))
    else:
        total_weight = subseq.eseq.get_total_weight()
    
    try:
        if method == "chisq" or method == "bonferroni":
            bonferroni = (method == "bonferroni")
            results = _chi_square_tests(subseq, group, bonferroni, weighted)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Filter by p-value
        significant = results['p.value'] <= pvalue_limit
        significant_idx = np.where(significant)[0]
        
        if len(significant_idx) == 0:
            warnings.warn("No subsequences found with p-value <= pvalue_limit", UserWarning)
            return SubsequenceList(
                subseq.eseq, [], pd.DataFrame(), subseq.constraint, type="chisq"
            )
        
        # Sort by test statistic (descending)
        sort_idx = significant_idx[np.argsort(results.loc[significant_idx, 'statistic'].values)[::-1]]
        
        # Create filtered subsequence list
        filtered_subseqs = [subseq.subsequences[i] for i in sort_idx]
        filtered_data = results.loc[sort_idx].reset_index(drop=True)
        
        result = SubsequenceList(
            subseq.eseq, filtered_subseqs, filtered_data, subseq.constraint, type="chisq"
        )
        result.labels = np.unique(group)
        result.bonferroni = {'used': bonferroni, 'ntest': len(subseq)}
        
        return result
    
    finally:
        if not weighted:
            subseq.eseq.weights = original_weights


# ============================================================================
# Internal Helper Functions
# ============================================================================

def _state_to_event_sequence(seqdata, tevent, use_labels, weighted, end_event, alphabet=None):
    """Convert a state sequence (SequenceData) to an event sequence.

    Aligned with TraMineR seqecreate(tevent=...) via seqformat(STS->TSE) and seqetm():
    - First event: initial state at time 0 (diagonal of tevent matrix).
    - Transition events: at times 1, 2, ... (j in 1:(slength-1) in STS_to_TSE).
    - Event dictionary: observed events only (matches alphabet(eseq) when alphabet=None).
    """
    from sequenzo.define_sequence_data import SequenceData
    
    if not isinstance(seqdata, SequenceData):
        raise TypeError("seqdata must be a SequenceData object")
    
    # Get transition matrix
    if isinstance(tevent, str):
        tevent_matrix = _create_transition_matrix(seqdata, tevent, use_labels)
    elif isinstance(tevent, np.ndarray):
        tevent_matrix = tevent
    else:
        raise TypeError("tevent must be a string or numpy array")
    
    # Convert state sequence to TSE format
    sequences = []
    seq_matrix = seqdata.seqdata.values
    ids = seqdata.ids

    # First pass: collect observed events (transitions + initial state per sequence, to match TraMineR)
    observed_events = set()
    for seq_row in seq_matrix:
        prev_state = None
        for state in seq_row:
            if pd.isna(state):
                continue
            if prev_state is None:
                observed_events.add(str(state))  # TraMineR includes initial state in alphabet
            if prev_state is not None and state != prev_state:
                if tevent == "transition":
                    event_name = f"{prev_state}>{state}"
                elif tevent == "state":
                    event_name = str(state)
                elif tevent == "period":
                    event_name = f"end{prev_state},begin{state}"
                else:
                    event_name = f"{prev_state}>{state}"
                observed_events.add(event_name)
            prev_state = state

    # Build event dictionary: use provided alphabet order (e.g. TraMineR) or sorted observed
    if alphabet is not None:
        dictionary = [x for x in alphabet if x in observed_events]
        if set(dictionary) != observed_events:
            for x in sorted(observed_events):
                if x not in dictionary:
                    dictionary.append(x)
    else:
        dictionary = sorted(list(observed_events))

    # Convert each sequence
    for idx, (seq_id, seq_row) in enumerate(zip(ids, seq_matrix)):
        timestamps = []
        events = []

        # First event: initial state at time 0 (TraMineR STS_to_TSE: "First status=> entrance event", times[myi] <- 0)
        prev_state = None
        for state in seq_row:
            if pd.isna(state):
                continue
            if prev_state is None and str(state) in dictionary:
                timestamps.append(0.0)
                events.append(dictionary.index(str(state)) + 1)
            prev_state = state

        # Then transitions at times 1, 2, ... (TraMineR: times[myi] <- j for j in 1:(slength-1))
        prev_state = None
        for pos, state in enumerate(seq_row):
            if pd.isna(state):
                continue

            if prev_state is not None and state != prev_state:
                if tevent == "transition":
                    event_name = f"{prev_state}>{state}"
                elif tevent == "state":
                    event_name = str(state)
                elif tevent == "period":
                    event_name = f"end{prev_state},begin{state}"
                else:
                    event_name = f"{prev_state}>{state}"

                if event_name in dictionary:
                    # TraMineR: times[myi] <- j for j in 1:(slength-1); j is 1-based index of first state
                    # Our pos is 0-based index of second state, so j = pos + 1
                    timestamps.append(float(pos + 1))
                    events.append(dictionary.index(event_name) + 1)

            prev_state = state

        sequences.append(EventSequence(
            id=int(seq_id),
            timestamps=np.array(timestamps),
            events=np.array(events),
            dictionary=dictionary
        ))
    
    # Get weights and lengths
    weights = None
    lengths = None
    if weighted and seqdata.weights is not None:
        weights = seqdata.weights
    
    if hasattr(seqdata, 'seqdata'):
        lengths = np.array([len(seq) for seq in seqdata.seqdata.values])
    
    return EventSequenceList(sequences, dictionary, weights=weights, lengths=lengths)


def _create_transition_matrix(seqdata, method, use_labels):
    """Create a transition matrix for state-to-event conversion."""
    states = seqdata.states
    n_states = len(states)
    
    if method == "transition":
        # One event per transition: "A>B"
        transitions = []
        for i in range(n_states):
            for j in range(n_states):
                if i == j:
                    transitions.append(states[i])
                else:
                    transitions.append(f"{states[i]}>{states[j]}")
        return transitions
    
    elif method == "state":
        # One event per state entry
        return states.copy()
    
    elif method == "period":
        # Pair of events: "endA,beginB"
        transitions = []
        for i in range(n_states):
            for j in range(n_states):
                if i == j:
                    transitions.append(states[i])
                else:
                    transitions.append(f"end{states[i]},begin{states[j]}")
        return transitions
    
    else:
        raise ValueError(f"Unknown transition method: {method}")


def _search_specific_subsequences(eseq: EventSequenceList,
                                  str_subseq: List[str],
                                  constraint: EventSequenceConstraint,
                                  total_weight: float,
                                  weighted: bool) -> SubsequenceList:
    """Search for specific user-defined subsequences."""
    # Parse string subsequences
    subsequences = []
    for sstr in str_subseq:
        subseq = _parse_subsequence_string(sstr, eseq.dictionary)
        subsequences.append(subseq)
    
    # Count occurrences
    constraint_presence = EventSequenceConstraint(count_method=1)
    counts = count_subsequence_occurrences(
        SubsequenceList(eseq, subsequences, pd.DataFrame(), constraint, "user"),
        method="presence", constraint=constraint
    )
    counts_distinct = count_subsequence_occurrences(
        SubsequenceList(eseq, subsequences, pd.DataFrame(), constraint, "user"),
        method="count", constraint=constraint
    )
    
    # Calculate support
    if weighted:
        support = np.sum(eseq.weights[:, np.newaxis] * counts_distinct, axis=0)
        support_presence = np.sum(eseq.weights[:, np.newaxis] * counts, axis=0)
    else:
        support = np.sum(counts_distinct, axis=0)
        support_presence = np.sum(counts, axis=0)
    
    # Create data frame
    data = pd.DataFrame({
        'Support': support_presence / total_weight,
        'Count': support
    })
    
    return SubsequenceList(eseq, subsequences, data, constraint, type="user")


def _parse_subsequence_string(sstr: str, dictionary: List[str]) -> EventSequence:
    """Parse a subsequence string like "(A)-(B,C)" into an EventSequence."""
    # Remove outer parentheses if present
    sstr = sstr.strip()
    if sstr.startswith('(') and sstr.endswith(')'):
        sstr = sstr[1:-1]
    
    # Split by transitions (-)
    transitions = sstr.split(')-(')
    if len(transitions) == 1:
        transitions = sstr.split('-')
    
    timestamps = []
    events = []
    t_index = 1
    
    for trans in transitions:
        # Remove parentheses
        trans = trans.strip().strip('()')
        
        # Split by comma for simultaneous events
        simultaneous = [e.strip() for e in trans.split(',')]
        
        for event_name in simultaneous:
            if event_name not in dictionary:
                raise ValueError(f"Event '{event_name}' not found in dictionary: {dictionary}")
            event_code = dictionary.index(event_name) + 1
            timestamps.append(float(t_index))
            events.append(event_code)
        
        t_index += 1
    
    return EventSequence(-1, np.array(timestamps), np.array(events), dictionary)


def _find_frequent_subsequences(eseq: EventSequenceList,
                               min_support: float,
                               constraint: EventSequenceConstraint,
                               max_k: int) -> Tuple[List[EventSequence], np.ndarray, np.ndarray]:
    """Find frequent subsequences using a simplified algorithm."""
    # This is a placeholder - full implementation would be much more complex
    subsequences = []
    supports = []
    counts = []
    
    # Find all 1-event subsequences
    for event_code in range(1, len(eseq.dictionary) + 1):
        subseq = EventSequence(-1, np.array([1.0]), np.array([event_code]), eseq.dictionary)
        count_array = count_subsequence_occurrences(
            SubsequenceList(eseq, [subseq], pd.DataFrame(), constraint, "frequent"),
            method="presence"
        )
        support = np.sum(eseq.weights * count_array[:, 0])
        
        if support >= min_support:
            subsequences.append(subseq)
            supports.append(support)
            counts.append(np.sum(count_array[:, 0]))
    
    # Find 2-event subsequences (if max_k allows)
    if max_k == -1 or max_k >= 2:
        for i, event1_code in enumerate(range(1, len(eseq.dictionary) + 1)):
            for event2_code in range(1, len(eseq.dictionary) + 1):
                # Try with gap and without gap
                for gap in [0.0, 1.0]:
                    subseq = EventSequence(
                        -1,
                        np.array([1.0, 1.0 + gap]),
                        np.array([event1_code, event2_code]),
                        eseq.dictionary
                    )
                    count_array = count_subsequence_occurrences(
                        SubsequenceList(eseq, [subseq], pd.DataFrame(), constraint, "frequent"),
                        method="presence"
                    )
                    support = np.sum(eseq.weights * count_array[:, 0])
                    
                    if support >= min_support:
                        subsequences.append(subseq)
                        supports.append(support)
                        counts.append(np.sum(count_array[:, 0]))
    
    return subsequences, np.array(supports), np.array(counts)


def _count_subsequence_in_sequence(subseq_seq: EventSequence, 
                                   seq: EventSequence,
                                   constraint: EventSequenceConstraint) -> float:
    """Count how many times a subsequence appears in a sequence."""
    if len(subseq_seq.timestamps) == 0:
        return 1.0 if len(seq.timestamps) == 0 else 0.0
    
    if len(seq.timestamps) == 0:
        return 0.0
    
    count_method = constraint.count_method
    
    if count_method == 1:  # COBJ: presence/absence
        return 1.0 if _find_subsequence_presence(subseq_seq, seq, constraint) else 0.0
    
    elif count_method == 2:  # CDIST_O: distinct occurrences
        return _count_distinct_occurrences(subseq_seq, seq, constraint)
    
    elif count_method == 3:  # CWIN: within windows
        if constraint.window_size == -1:
            raise ValueError("CWIN method requires window_size constraint")
        return _count_within_windows(subseq_seq, seq, constraint)
    
    elif count_method == 4:  # CMINWIN: minimum windows
        if constraint.window_size == -1:
            raise ValueError("CMINWIN method requires window_size constraint")
        return _count_minimum_windows(subseq_seq, seq, constraint)
    
    elif count_method == 5:  # CDIST: with distance constraints
        return _count_with_distance(subseq_seq, seq, constraint)
    
    else:
        raise ValueError(f"Unknown count method: {count_method}")


def _find_subsequence_presence(subseq_seq: EventSequence, 
                               seq: EventSequence,
                               constraint: EventSequenceConstraint) -> bool:
    """Check if subsequence is present in sequence (method 1: COBJ)."""
    return _find_first_occurrence(subseq_seq, seq, constraint) is not None


def _find_occurrence_starting_at(subseq_seq: EventSequence,
                                 seq: EventSequence,
                                 start_idx: int,
                                 constraint: EventSequenceConstraint) -> bool:
    """Check if subsequence occurs in sequence as ordered subset with first event at start_idx (TraMineR: subsequence = ordered subset)."""
    if len(subseq_seq.events) == 0:
        return True
    if start_idx >= len(seq.events) or seq.events[start_idx] != subseq_seq.events[0]:
        return False
    start_time = seq.timestamps[start_idx]
    if constraint.age_min != -1 and start_time < constraint.age_min:
        return False
    if constraint.age_max != -1 and start_time > constraint.age_max:
        return False
    indices = [start_idx]
    pos = start_idx + 1
    for k in range(1, len(subseq_seq.events)):
        found = False
        for i in range(pos, len(seq.events)):
            if seq.events[i] != subseq_seq.events[k]:
                continue
            if constraint.max_gap != -1 and seq.timestamps[i] - seq.timestamps[indices[-1]] > constraint.max_gap:
                continue
            if constraint.window_size != -1 and seq.timestamps[i] - seq.timestamps[indices[0]] > constraint.window_size:
                continue
            indices.append(i)
            pos = i + 1
            found = True
            break
        if not found:
            return False
    if constraint.age_max_end != -1 and seq.timestamps[indices[-1]] > constraint.age_max_end:
        return False
    return True


def _find_first_occurrence(subseq_seq: EventSequence,
                           seq: EventSequence,
                           constraint: EventSequenceConstraint) -> Optional[Tuple[int, float]]:
    """Find the first occurrence of subsequence in sequence (ordered subset, not necessarily consecutive)."""
    if len(subseq_seq.events) == 0:
        return (0, seq.timestamps[0] if len(seq.timestamps) > 0 else 0.0)
    if len(seq.events) < len(subseq_seq.events):
        return None
    for start_idx in range(len(seq.events)):
        if _find_occurrence_starting_at(subseq_seq, seq, start_idx, constraint):
            return (start_idx, seq.timestamps[start_idx])
    return None


def _matches_at_position(subseq_seq: EventSequence,
                         seq: EventSequence,
                         start_idx: int,
                         constraint: EventSequenceConstraint) -> bool:
    """Check if subsequence matches sequence as ordered subset starting at start_idx (used by CDIST_O etc.)."""
    return _find_occurrence_starting_at(subseq_seq, seq, start_idx, constraint)


def _find_first_occurrence_from(subseq_seq: EventSequence,
                                seq: EventSequence,
                                start_from: int,
                                constraint: EventSequenceConstraint) -> Optional[Tuple[int, float]]:
    """Find first occurrence with first event at or after start_from."""
    if len(subseq_seq.events) == 0:
        if len(seq.events) == 0:
            return (0, 0.0)
        idx = min(start_from, len(seq.events) - 1)
        return (idx, seq.timestamps[idx])
    if len(seq.events) < len(subseq_seq.events):
        return None
    for start_idx in range(start_from, len(seq.events)):
        if _find_occurrence_starting_at(subseq_seq, seq, start_idx, constraint):
            return (start_idx, seq.timestamps[start_idx])
    return None


def _count_distinct_occurrences(subseq_seq: EventSequence,
                                seq: EventSequence,
                                constraint: EventSequenceConstraint) -> float:
    """Count distinct occurrences of subsequence in sequence (method 2: CDIST_O)."""
    count = 0
    start_idx = 0
    
    while True:
        match = _find_first_occurrence_from(subseq_seq, seq, start_idx, constraint)
        if match is None:
            break
        
        count += 1
        start_idx = match[0] + 1  # Move past this occurrence
    
    return float(count)


def _count_within_windows(subseq_seq: EventSequence,
                         seq: EventSequence,
                         constraint: EventSequenceConstraint) -> float:
    """Count occurrences within time windows (method 3: CWIN)."""
    return _count_distinct_occurrences(subseq_seq, seq, constraint)


def _count_minimum_windows(subseq_seq: EventSequence,
                          seq: EventSequence,
                          constraint: EventSequenceConstraint) -> float:
    """Count minimum windows containing subsequence (method 4: CMINWIN)."""
    return _count_distinct_occurrences(subseq_seq, seq, constraint)


def _count_with_distance(subseq_seq: EventSequence,
                        seq: EventSequence,
                        constraint: EventSequenceConstraint) -> float:
    """Count with distance constraints (method 5: CDIST)."""
    return _count_distinct_occurrences(subseq_seq, seq, constraint)


def _chi_square_tests(subseq: SubsequenceList,
                     group: np.ndarray,
                     bonferroni: bool,
                     weighted: bool) -> pd.DataFrame:
    """Perform chi-square tests for each subsequence."""
    if not HAS_SCIPY:
        raise ImportError("scipy is required for compare_groups. Please install scipy.")
    
    # Get presence matrix
    presence_matrix = count_subsequence_occurrences(subseq, method="presence")
    
    group_factor = pd.Categorical(group)
    n_groups = len(group_factor.categories)
    n_subseqs = len(subseq)
    
    results = []
    
    for j in range(n_subseqs):
        # Create contingency table
        contingency = np.zeros((n_groups, 2), dtype=np.float64)
        
        for i, g in enumerate(group_factor.codes):
            weight = subseq.eseq.weights[i] if weighted else 1.0
            if presence_matrix[i, j] > 0:
                contingency[g, 1] += weight  # Present
            else:
                contingency[g, 0] += weight  # Absent
        
        # Check if we can perform test
        total_present = np.sum(contingency[:, 1])
        total_absent = np.sum(contingency[:, 0])
        
        if total_present > 0 and total_present < len(group):
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            if bonferroni:
                # Bonferroni correction
                p_value = 1 - (1 - p_value) ** n_subseqs
            
            # Calculate frequencies and residuals
            row_totals = np.sum(contingency, axis=1)
            freq = contingency[:, 1] / row_totals
            residuals = (contingency[:, 1] - expected[:, 1]) / np.sqrt(expected[:, 1] + 1e-10)
            
            # Create result row
            result_row = {
                'p.value': p_value,
                'statistic': chi2,
                'index': j + 1,
                'Support': subseq.data.iloc[j]['Support']
            }
            
            # Add frequency and residual columns for each group
            for k, cat in enumerate(group_factor.categories):
                result_row[f'Freq.{cat}'] = freq[k]
                result_row[f'Resid.{cat}'] = residuals[k]
            
            results.append(result_row)
        else:
            # All or none have the subsequence
            freq = np.full(n_groups, total_present / len(group))
            residuals = np.zeros(n_groups)
            
            result_row = {
                'p.value': 1.0,
                'statistic': 0.0,
                'index': j + 1,
                'Support': subseq.data.iloc[j]['Support']
            }
            
            for k, cat in enumerate(group_factor.categories):
                result_row[f'Freq.{cat}'] = freq[k]
                result_row[f'Resid.{cat}'] = residuals[k]
            
            results.append(result_row)
    
    return pd.DataFrame(results)
