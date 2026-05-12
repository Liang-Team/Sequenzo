"""
@Author  : Yuqi Liang 梁彧祺
@File    : core.py
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
EventSequenceData.from_tse()         seqecreate()          Create event sequence objects from TSE data
EventSequenceData.from_state_sequences() seqecreate()      Create event sequence objects from state sequences
find_frequent_subsequences()         seqefsub()            Find frequently occurring event patterns above a support threshold
count_subsequence_occurrences()      seqeapplysub()        Count how many times each subsequence appears in each sequence
compare_groups()                     seqecmpgroup()        Compare groups to find discriminating subsequences (chi-square tests)
plot_event_parallel_coordinates()     seqpcplot()           Parallel-coordinate event-sequence visualization
plot_subsequence_frequencies()        plot.subseqelist()    Plot bar chart of subsequence frequencies

Key Classes:
------------
- EventSequence: Represents a single event sequence for one individual
- EventSequenceList: Collection of event sequences (equivalent to TraMineR's seqelist)
- EventSequenceConstraint: Time constraints for subsequence search (age windows, gaps, etc.)
- SubsequenceList: List of frequent subsequences with metadata (equivalent to TraMineR's subseqelist)

Example Usage:
--------------
    >>> import pandas as pd
    >>> from sequenzo.event_sequences import EventSequenceData, find_frequent_subsequences
    >>> 
    >>> # Create event sequences from TSE format data
    >>> tse_data = pd.DataFrame({
    ...     'id': [1, 1, 2, 2],
    ...     'timestamp': [18, 22, 17, 20],
    ...     'event': ['EnterUni', 'Graduate', 'EnterUni', 'Graduate']
    ... })
    >>> eseq = EventSequenceData.from_tse(data=tse_data)
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


class EventSequenceData(EventSequenceList):
    """
    Primary event-sequence container, parallel to ``SequenceData``.

    This class is the recommended public object for event-sequence workflows.
    """

    @classmethod
    def from_tse(cls,
                 data: Optional[pd.DataFrame] = None,
                 id: Optional[Union[np.ndarray, pd.Series, List]] = None,
                 timestamp: Optional[Union[np.ndarray, pd.Series, List]] = None,
                 event: Optional[Union[np.ndarray, pd.Series, List]] = None,
                 end_event: Optional[str] = None,
                 event_labels_order: Optional[List[str]] = None):
        """Create event sequences from TSE-style data.

        TraMineR parameter mapping: ``event_labels_order`` -> ``alphabet``.
        """
        return _create_event_sequences(
            data=data,
            id=id,
            timestamp=timestamp,
            event=event,
            end_event=end_event,
            event_labels_order=event_labels_order,
        )

    @classmethod
    def from_state_sequences(cls,
                             seqdata,
                             event_representation: Union[str, np.ndarray] = "transition",
                             use_labels: bool = True,
                             weighted: bool = True,
                             end_event: Optional[str] = None,
                             event_labels_order: Optional[List[str]] = None):
        """Create event sequences from a ``SequenceData`` object.

        TraMineR parameter mapping: ``event_representation`` -> ``tevent``,
        ``event_labels_order`` -> ``alphabet``, ``weighted`` -> ``weighted``.
        """
        return _create_event_sequences(
            data=seqdata,
            event_representation=event_representation,
            use_labels=use_labels,
            weighted=weighted,
            end_event=end_event,
            event_labels_order=event_labels_order,
        )

    def to_tse(self) -> pd.DataFrame:
        """Convert this event-sequence object to TSE format."""
        return convert_event_sequences_to_tse(self)

    def transition_matrix(self, weighted: bool = True, normalize: bool = True) -> pd.DataFrame:
        """Compute event transition matrix."""
        return compute_event_transition_matrix(self, weighted=weighted, normalize=normalize)

    def contains(self,
                 target_subsequence: Union["EventSequence", str],
                 search_constraint: Optional["EventSequenceConstraint"] = None) -> pd.Series:
        """Check per-sequence containment of a target subsequence."""
        return check_event_subsequence_containment(
            self,
            target_subsequence=target_subsequence,
            search_constraint=search_constraint,
        )


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

def _create_event_sequences(data: Optional[pd.DataFrame] = None,
                            id: Optional[Union[np.ndarray, pd.Series, List]] = None,
                            timestamp: Optional[Union[np.ndarray, pd.Series, List]] = None,
                            event: Optional[Union[np.ndarray, pd.Series, List]] = None,
                            end_event: Optional[str] = None,
                            event_representation: Union[str, np.ndarray] = "transition",
                            use_labels: bool = True,
                            weighted: bool = True,
                            event_labels_order: Optional[List[str]] = None,
                            seqdata=None) -> EventSequenceData:
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
        event_representation: Transition method for state-to-event conversion:
                - "transition": One event per transition (default)
                - "state": One event per state entry
                - "period": Pair of start/end events
                - Or a transition matrix (numpy array)
        use_labels: If True, use state labels instead of codes
        weighted: If True, preserve weights from state sequences
        event_labels_order: Optional list of event labels in a specific order (e.g. TraMineR event_labels_order
                  for reference comparison). If provided, the event dictionary uses this order.
        seqdata: Deprecated, use data instead
    
    Returns:
        EventSequenceData object containing event sequences
    
    Examples:
        >>> # From TSE format DataFrame
        >>> tse_data = pd.DataFrame({
        ...     'id': [1, 1, 2, 2],
        ...     'timestamp': [18, 22, 17, 20],
        ...     'event': ['EnterUni', 'Graduate', 'EnterUni', 'Graduate']
        ... })
        >>> eseq = EventSequenceData.from_tse(data=tse_data)
        
        >>> # From state sequence (SequenceData object)
        >>> eseq = EventSequenceData.from_state_sequences(state_seq, event_representation="transition")
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
            return _state_to_event_sequence(
                data,
                event_representation,
                use_labels,
                weighted,
                end_event,
                event_labels_order,
            )
        
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
    observed_events = [str(e) for e in unique_events]
    if event_labels_order is not None:
        dictionary = [ev for ev in event_labels_order if ev in observed_events]
        for ev in sorted(observed_events):
            if ev not in dictionary:
                dictionary.append(ev)
    else:
        dictionary = sorted(observed_events)
    
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
    eseq_list = EventSequenceData(sequences, dictionary)
    
    return eseq_list


def create_event_sequences(
    data: Optional[pd.DataFrame] = None,
    id: Optional[Union[np.ndarray, pd.Series, List]] = None,
    timestamp: Optional[Union[np.ndarray, pd.Series, List]] = None,
    event: Optional[Union[np.ndarray, pd.Series, List]] = None,
    end_event: Optional[str] = None,
    event_representation: Union[str, np.ndarray] = "transition",
    tevent: Optional[Union[str, np.ndarray]] = None,
    use_labels: bool = True,
    weighted: bool = True,
    event_labels_order: Optional[List[str]] = None,
    alphabet: Optional[List[str]] = None,
    seqdata=None,
) -> EventSequenceData:
    """
    Public TraMineR-friendly wrapper for creating event sequences.

    ``tevent`` and ``alphabet`` are accepted as compatibility aliases for
    TraMineR-style workflows and older Sequenzo tests.
    """
    if tevent is not None:
        event_representation = tevent
    if alphabet is not None:
        event_labels_order = list(alphabet)

    return _create_event_sequences(
        data=data,
        id=id,
        timestamp=timestamp,
        event=event,
        end_event=end_event,
        event_representation=event_representation,
        use_labels=use_labels,
        weighted=weighted,
        event_labels_order=event_labels_order,
        seqdata=seqdata,
    )


def find_frequent_subsequences(event_sequences: EventSequenceList,
                               target_subsequences: Optional[List[str]] = None,
                               min_support: Optional[float] = None,
                               min_support_ratio: Optional[float] = None,
                               search_constraint: Optional[EventSequenceConstraint] = None,
                               max_k: int = -1,
                               weighted: bool = True) -> SubsequenceList:
    """
    Find frequent subsequences in event sequences.
    
    TraMineR equivalent: seqefsub()
    TraMineR parameter mapping: ``event_sequences`` -> ``eseq``,
    ``target_subsequences`` -> ``str.subseq``,
    ``min_support_ratio`` -> ``pmin.support``,
    ``search_constraint`` -> ``constraint``.
    
    Args:
        event_sequences: EventSequenceList object
        target_subsequences: Optional list of specific subsequences to search for (as strings)
        min_support: Minimum support in number of sequences
        min_support_ratio: Minimum support as proportion (0-1)
        search_constraint: EventSequenceConstraint object
        max_k: Maximum number of events in subsequence (-1 = no limit)
        weighted: If True, use sequence weights
    
    Returns:
        SubsequenceList object with frequent subsequences
    
    Examples:
        >>> # Find subsequences with at least 20 occurrences
        >>> fsubseq = find_frequent_subsequences(event_sequences, min_support=20)
        
        >>> # Find subsequences with at least 1% support
        >>> fsubseq = find_frequent_subsequences(event_sequences, min_support_ratio=0.01)
        
        >>> # Search for specific subsequences
        >>> fsubseq = find_frequent_subsequences(event_sequences, target_subsequences=["(A)-(B)", "(B)-(C)"])
    """
    if not isinstance(event_sequences, EventSequenceList):
        raise TypeError("event_sequences must be an EventSequenceList object")
    
    if search_constraint is None:
        search_constraint = EventSequenceConstraint()
    
    if not isinstance(search_constraint, EventSequenceConstraint):
        warnings.warn("search_constraint should be an EventSequenceConstraint object. Using default.", UserWarning)
        search_constraint = EventSequenceConstraint()
    
    # Validate constraint
    if search_constraint.count_method == 3 and search_constraint.window_size == -1:
        raise ValueError("CWIN method requires window_size constraint")
    
    # Handle weights
    if weighted:
        total_weight = event_sequences.get_total_weight()
    else:
        total_weight = float(len(event_sequences))
        # Temporarily set weights to 1
        original_weights = event_sequences.weights.copy()
        event_sequences.weights = np.ones(len(event_sequences), dtype=np.float64)
    
    try:
        # Handle user-specified subsequences
        if target_subsequences is not None:
            return _search_specific_subsequences(event_sequences, target_subsequences, search_constraint, total_weight, weighted)
        
        # Validate support threshold
        if min_support is None:
            if min_support_ratio is None:
                raise ValueError("You should specify a minimum support through min_support or min_support_ratio")
            min_support = min_support_ratio * total_weight
        
        # Find frequent subsequences
        # This is a simplified implementation - full version would use prefix tree algorithm
        subsequences, supports, counts = _find_frequent_subsequences(
            event_sequences, min_support, search_constraint, max_k
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
        
        return SubsequenceList(event_sequences, subsequences, data, search_constraint, type="frequent")
    
    finally:
        # Restore original weights if we modified them
        if not weighted:
            event_sequences.weights = original_weights


def count_subsequence_occurrences(subsequence_results: SubsequenceList,
                                  counting_method: Optional[Union[str, int]] = None,
                                  search_constraint: Optional[EventSequenceConstraint] = None,
                                  include_rules: bool = False) -> np.ndarray:
    """
    Count occurrences of subsequences in sequences.
    
    TraMineR equivalent: seqeapplysub()
    TraMineR parameter mapping: ``subsequence_results`` -> ``fsub``,
    ``counting_method`` -> ``method``, ``search_constraint`` -> ``constraint``.
    
    Args:
        subsequence_results: SubsequenceList object (result from find_frequent_subsequences)
        counting_method: Counting method:
                - "presence" or "COBJ" (1): Count per sequence (0/1)
                - "count" or "CDIST_O" (2): Count distinct occurrences
                - "CWIN" (3): Count within time windows
                - "CMINWIN" (4): Count minimum windows
                - "CDIST" (5): Count with distance constraints
                - Or integer 1-5
        search_constraint: EventSequenceConstraint object (uses subsequence_results.constraint if None)
        include_rules: If True, count subsequences within subsequences (for rule mining)
    
    Returns:
        numpy array of shape (n_sequences, n_subsequences) with counts
    
    Examples:
        >>> fsubseq = find_frequent_subsequences(eseq, min_support=10)
        >>> counts = count_subsequence_occurrences(fsubseq, counting_method="presence")
        >>> # counts[i, j] = 1 if sequence i contains subsequence j, else 0
    """
    if not isinstance(subsequence_results, SubsequenceList):
        raise TypeError("subsequence_results must be a SubsequenceList object")
    
    if search_constraint is None:
        search_constraint = subsequence_results.constraint
    
    if not isinstance(search_constraint, EventSequenceConstraint):
        warnings.warn("search_constraint should be an EventSequenceConstraint object. Using default.", UserWarning)
        search_constraint = EventSequenceConstraint()
    
    # Handle method parameter
    if counting_method is not None:
        if isinstance(counting_method, str):
            method_map = {
                "presence": 1, "COBJ": 1,
                "count": 2, "CDIST_O": 2,
                "CWIN": 3,
                "CMINWIN": 4,
                "CDIST": 5
            }
            if counting_method not in method_map:
                raise ValueError(f"Unknown counting_method: {counting_method}")
            search_constraint.count_method = method_map[counting_method]
        elif isinstance(counting_method, int):
            if counting_method not in [1, 2, 3, 4, 5]:
                raise ValueError(f"counting_method must be 1-5, got {counting_method}")
            search_constraint.count_method = counting_method
    
    # Get sequences to search in
    if include_rules:
        search_sequences = subsequence_results.subsequences
    else:
        search_sequences = subsequence_results.eseq.sequences
    
    n_seqs = len(search_sequences)
    n_subseqs = len(subsequence_results.subsequences)
    
    # Initialize result matrix
    result = np.zeros((n_seqs, n_subseqs), dtype=np.float64)
    
    # Count occurrences for each subsequence in each sequence
    for j, subseq_seq in enumerate(subsequence_results.subsequences):
        for i, seq in enumerate(search_sequences):
            count = _count_subsequence_in_sequence(
                subseq_seq, seq, search_constraint
            )
            result[i, j] = count
    
    return result


def compare_groups(subsequence_results: SubsequenceList,
                   group_labels: Union[np.ndarray, pd.Series, List],
                   test_method: str = "chisq",
                   pvalue_threshold: Optional[float] = None,
                   weighted: bool = True) -> SubsequenceList:
    """
    Compare groups to find discriminating subsequences.
    
    TraMineR equivalent: seqecmpgroup()
    TraMineR parameter mapping: ``subsequence_results`` -> ``fsub``,
    ``group_labels`` -> ``group``, ``test_method`` -> ``method``,
    ``pvalue_threshold`` -> ``pvalue.limit``.
    
    Args:
        subsequence_results: SubsequenceList object (result from find_frequent_subsequences)
        group_labels: Group membership for each sequence (array-like)
        test_method: Test method ("chisq" for chi-square test, "bonferroni" for Bonferroni correction)
        pvalue_threshold: Maximum p-value threshold (default: 2.0 for display)
        weighted: If True, use sequence weights
    
    Returns:
        SubsequenceList object filtered to discriminating subsequences
    
    Examples:
        >>> fsubseq = find_frequent_subsequences(eseq, min_support=10)
        >>> groups = np.array(['Male', 'Female', 'Male', 'Female', 'Male'])
        >>> discr = compare_groups(fsubseq, groups, pvalue_threshold=0.05)
    """
    if not isinstance(subsequence_results, SubsequenceList):
        raise TypeError("subsequence_results must be a SubsequenceList object")
    
    group_labels = np.array(group_labels)
    if len(group_labels) != len(subsequence_results.eseq):
        raise ValueError(f"group_labels length ({len(group_labels)}) must match number of sequences ({len(subsequence_results.eseq)})")
    
    if pvalue_threshold is None:
        pvalue_threshold = 2.0  # TraMineR default
    
    # Handle weights
    if not weighted:
        original_weights = subsequence_results.eseq.weights.copy()
        subsequence_results.eseq.weights = np.ones(len(subsequence_results.eseq), dtype=np.float64)
        total_weight = float(len(subsequence_results.eseq))
    else:
        total_weight = subsequence_results.eseq.get_total_weight()
    
    try:
        if test_method == "chisq" or test_method == "bonferroni":
            bonferroni = (test_method == "bonferroni")
            results = _chi_square_tests(subsequence_results, group_labels, bonferroni, weighted)
        else:
            raise ValueError(f"Unknown test_method: {test_method}")
        
        # Filter by p-value
        significant = results['p.value'] <= pvalue_threshold
        significant_idx = np.where(significant)[0]
        
        if len(significant_idx) == 0:
            warnings.warn("No subsequences found with p-value <= pvalue_threshold", UserWarning)
            return SubsequenceList(
                subsequence_results.eseq, [], pd.DataFrame(), subsequence_results.constraint, type="chisq"
            )
        
        # Sort by test statistic (descending)
        sort_idx = significant_idx[np.argsort(results.loc[significant_idx, 'statistic'].values)[::-1]]
        
        # Create filtered subsequence list
        filtered_subseqs = [subsequence_results.subsequences[i] for i in sort_idx]
        filtered_data = results.loc[sort_idx].reset_index(drop=True)
        
        result = SubsequenceList(
            subsequence_results.eseq, filtered_subseqs, filtered_data, subsequence_results.constraint, type="chisq"
        )
        result.labels = np.unique(group_labels)
        result.bonferroni = {'used': bonferroni, 'ntest': len(subsequence_results)}
        
        return result
    
    finally:
        if not weighted:
            subsequence_results.eseq.weights = original_weights


def convert_event_sequences_to_tse(event_sequences: EventSequenceList) -> pd.DataFrame:
    """
    Convert an EventSequenceList to TraMineR-style TSE (Time-Stamped Event) format.

    TraMineR equivalent: seqe2tse()
    TraMineR parameter mapping: ``event_sequences`` -> ``eseq``.

    The resulting DataFrame has three columns:
        - 'id'      : individual identifier
        - 'timestamp': event time (numeric, already sorted within each id)
        - 'event'   : event label (string, using the event dictionary)

    Notes
    -----
    - This is a pure format conversion helper. It does not modify the original
      event sequences and can be called at any time.
    - Timestamps are taken as-is from the EventSequence objects.
    """
    rows = []
    for seq in event_sequences.sequences:
        # For each event in the sequence, create one TSE row
        for t, e in zip(seq.timestamps, seq.events):
            # Map integer code back to event label using dictionary
            if 1 <= int(e) <= len(seq.dictionary):
                ev_label = seq.dictionary[int(e) - 1]
            else:
                ev_label = f"Event{int(e)}"
            rows.append(
                {
                    "id": seq.id,
                    "timestamp": float(t),
                    "event": ev_label,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["id", "timestamp", "event"])

    tse_df = pd.DataFrame(rows, columns=["id", "timestamp", "event"])
    # Ensure events are sorted by id and timestamp exactly like TraMineR
    tse_df = tse_df.sort_values(["id", "timestamp", "event"]).reset_index(drop=True)
    return tse_df


def compute_event_transition_matrix(
    event_sequences: EventSequenceList,
    weighted: bool = True,
    normalize: bool = True,
    use_weights: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Compute the event transition matrix for an EventSequenceList.

    TraMineR equivalent: seqetm()
    TraMineR parameter mapping: ``event_sequences`` -> ``eseq``,
    ``weighted`` -> ``weighted``.

    This function counts how often each ordered pair of events (i -> j)
    occurs across all sequences and optionally normalizes the counts
    into transition probabilities.

    Parameters
    ----------
    event_sequences : EventSequenceList
        Collection of event sequences.
    weighted : bool, default True
        If True, each sequence contributes according to its weight stored in
        event_sequences.weights. If False, all sequences are equally weighted.
    normalize : bool, default True
        If True, each row of the resulting matrix is normalized so that:
            sum_j P(i -> j) = 1  (when row i has at least one outgoing transition)
        If False, raw weighted counts are returned.

    Returns
    -------
    pandas.DataFrame
        Square matrix with shape (K, K), where K is the number of distinct
        events in the dictionary. Rows and columns are indexed by event labels.
        Cell (i, j) contains either:
            - the weighted count of transitions i -> j       (normalize=False), or
            - the transition probability P(i -> j)           (normalize=True).
    """
    if use_weights is not None:
        weighted = use_weights

    n_events = len(event_sequences.dictionary)
    if n_events == 0:
        return pd.DataFrame()

    # Initialize transition count matrix
    counts = np.zeros((n_events, n_events), dtype=np.float64)

    # Decide which weights to use
    if weighted:
        weights = event_sequences.weights
    else:
        weights = np.ones(event_sequences.n_sequences, dtype=np.float64)

    # Loop over sequences and accumulate transitions
    for idx, seq in enumerate(event_sequences.sequences):
        w = float(weights[idx])
        # Need at least two events to define a transition
        if len(seq.events) < 2 or w == 0.0:
            continue

        for k in range(len(seq.events) - 1):
            src_code = int(seq.events[k])
            dst_code = int(seq.events[k + 1])

            # Codes are 1-based indices into the dictionary. We ignore any
            # unexpected values for safety.
            if 1 <= src_code <= n_events and 1 <= dst_code <= n_events:
                counts[src_code - 1, dst_code - 1] += w

    # Convert to DataFrame with readable labels
    labels = list(event_sequences.dictionary)
    tm = pd.DataFrame(counts, index=labels, columns=labels, dtype=float)

    if normalize:
        # Normalize each row to sum to 1 (if row sum > 0)
        row_sums = tm.sum(axis=1)
        non_zero = row_sums > 0
        tm.loc[non_zero] = tm.loc[non_zero].div(row_sums[non_zero], axis=0)

    return tm


def check_event_subsequence_containment(
    event_sequences: EventSequenceList,
    target_subsequence: Union[EventSequence, str],
    search_constraint: Optional[EventSequenceConstraint] = None,
) -> pd.Series:
    """
    Check whether each event sequence contains a given subsequence.

    TraMineR equivalent: seqecontain()
    TraMineR parameter mapping: ``event_sequences`` -> ``eseq``,
    ``target_subsequence`` -> ``subseq``,
    ``search_constraint`` -> ``constraint``.

    This is a high-level convenience wrapper built on top of the internal
    subsequence search utilities. It returns a boolean indicator for each
    sequence, telling you whether the target subsequence occurs at least once.

    Parameters
    ----------
    event_sequences : EventSequenceList
        Collection of event sequences to be scanned.
    target_subsequence : EventSequence or str
        Target subsequence specification:
        - If EventSequence: used directly, its dictionary must be compatible
          with event_sequences.dictionary.
        - If str: parsed using the same syntax as TraMineR, e.g. "(A)-(B,C)".
    search_constraint : EventSequenceConstraint, optional
        Time and counting constraints controlling what counts as a valid
        occurrence (maximum gap, window size, age limits, etc.). When None,
        a default unconstrained EventSequenceConstraint is used.

    Returns
    -------
    pandas.Series
        Boolean Series of length n_sequences, indexed by implicit 0..n-1:
        - True  : subsequence occurs at least once in the sequence
        - False : subsequence does not occur
    """
    if search_constraint is None:
        search_constraint = EventSequenceConstraint()

    # Ensure we have an EventSequence representation of the target subsequence
    if isinstance(target_subsequence, EventSequence):
        target = target_subsequence
    elif isinstance(target_subsequence, str):
        target = _parse_subsequence_string(target_subsequence, event_sequences.dictionary)
    else:
        raise TypeError(
            "target_subsequence must be either an EventSequence or a subsequence string "
            "such as '(A)-(B,C)'."
        )

    results = []
    for seq in event_sequences.sequences:
        present = _find_subsequence_presence(
            subseq_seq=target,
            seq=seq,
            constraint=search_constraint,
        )
        results.append(bool(present))

    return pd.Series(results, name="contains")


# ============================================================================
# Internal Helper Functions
# ============================================================================

def _tevent_grid_lookup(tevent_matrix, i0: int, j0: int, n_states: int) -> str:
    """Event label at row i0, column j0 (0-based) in a full n×n event_representation grid."""
    if isinstance(tevent_matrix, np.ndarray):
        if tevent_matrix.ndim != 2:
            raise ValueError("event_representation ndarray must be 2-D")
        return str(tevent_matrix[i0, j0])
    return str(tevent_matrix[i0 * n_states + j0])


def _is_full_nxn_tevent(tevent_matrix, n_states: int) -> bool:
    """True if event_representation is an n×n grid (ndarray or row-major flat list of length n²)."""
    if isinstance(tevent_matrix, np.ndarray):
        return tevent_matrix.ndim == 2 and tevent_matrix.shape == (n_states, n_states)
    if isinstance(tevent_matrix, (list, tuple)):
        return len(tevent_matrix) == n_states * n_states
    return False


def _initial_event_label(seqdata, event_representation, event_representation_matrix, state_code, n_states: int) -> str:
    """Label for entering the first observed state (time 0); diagonal of event_representation grid."""
    j = int(state_code) - 1
    if isinstance(event_representation, np.ndarray) or _is_full_nxn_tevent(event_representation_matrix, n_states):
        return _tevent_grid_lookup(event_representation_matrix, j, j, n_states)
    return str(seqdata.states[j])


def _transition_event_label(seqdata, event_representation, event_representation_matrix, prev_code, state_code, n_states: int) -> str:
    """Label for a change from prev_code to state_code."""
    i = int(prev_code) - 1
    j = int(state_code) - 1
    if isinstance(event_representation, np.ndarray) or _is_full_nxn_tevent(event_representation_matrix, n_states):
        return _tevent_grid_lookup(event_representation_matrix, i, j, n_states)
    if event_representation == "state":
        return str(seqdata.states[j])
    if event_representation == "transition":
        return f"{seqdata.states[i]}>{seqdata.states[j]}"
    if event_representation == "period":
        return f"end{seqdata.states[i]},begin{seqdata.states[j]}"
    return f"{seqdata.states[i]}>{seqdata.states[j]}"


def _state_to_event_sequence(
    seqdata,
    event_representation,
    use_labels,
    weighted,
    end_event,
    event_labels_order=None,
):
    """Convert a state sequence (SequenceData) to an event sequence.

    Aligned with TraMineR seqecreate(event_representation=...) via seqformat(STS->TSE) and seqetm():
    - First event: initial state at time 0 (diagonal of event_representation matrix).
    - Transition events: at times 1, 2, ... (j in 1:(slength-1) in STS_to_TSE).
    - Event dictionary: observed events only (matches event_labels_order(eseq) when event_labels_order=None).
    """
    from sequenzo.define_sequence_data import SequenceData
    
    if not isinstance(seqdata, SequenceData):
        raise TypeError("seqdata must be a SequenceData object")
    
    # Get transition matrix
    if isinstance(event_representation, str):
        event_representation_matrix = _create_transition_matrix(seqdata, event_representation, use_labels)
    elif isinstance(event_representation, np.ndarray):
        event_representation_matrix = event_representation
    else:
        raise TypeError("event_representation must be a string or numpy array")
    
    # Convert state sequence to TSE format
    sequences = []
    seq_matrix = seqdata.seqdata.values
    ids = seqdata.ids
    n_states = len(seqdata.states)

    # First pass: collect observed events (transitions + initial state per sequence, to match TraMineR)
    observed_events = set()
    for seq_row in seq_matrix:
        prev_state = None
        for state in seq_row:
            if pd.isna(state):
                continue
            if prev_state is None:
                observed_events.add(
                    _initial_event_label(seqdata, event_representation, event_representation_matrix, state, n_states)
                )
            if prev_state is not None and state != prev_state:
                observed_events.add(
                    _transition_event_label(seqdata, event_representation, event_representation_matrix, prev_state, state, n_states)
                )
            prev_state = state

    # Build event dictionary:
    # - If user provided event_labels_order, keep that order (then append any observed extras)
    # - Otherwise, prefer transition-matrix order (TraMineR-like), not lexical sorting.
    if event_labels_order is not None:
        dictionary = [x for x in event_labels_order if x in observed_events]
        if set(dictionary) != observed_events:
            for x in sorted(observed_events):
                if x not in dictionary:
                    dictionary.append(x)
    else:
        ordered = []
        # event_representation_matrix can be a list/array of candidate event names
        try:
            if isinstance(event_representation_matrix, np.ndarray):
                flat = event_representation_matrix.ravel().tolist()
            else:
                flat = list(event_representation_matrix)
        except Exception:
            flat = []

        for ev in flat:
            evs = str(ev)
            if evs in observed_events and evs not in ordered:
                ordered.append(evs)

        # Keep deterministic completion for any event not present in event_representation ordering
        for ev in sorted(observed_events):
            if ev not in ordered:
                ordered.append(ev)

        dictionary = ordered

    # Convert each sequence
    for idx, (seq_id, seq_row) in enumerate(zip(ids, seq_matrix)):
        timestamps = []
        events = []

        # First event: initial state at time 0 (TraMineR STS_to_TSE: diagonal event_representation cell)
        prev_state = None
        for state in seq_row:
            if pd.isna(state):
                continue
            if prev_state is None:
                init_label = _initial_event_label(seqdata, event_representation, event_representation_matrix, state, n_states)
                if init_label in dictionary:
                    timestamps.append(0.0)
                    events.append(dictionary.index(init_label) + 1)
            prev_state = state

        # Then transitions at times 1, 2, ... (TraMineR: times[myi] <- j for j in 1:(slength-1))
        prev_state = None
        for pos, state in enumerate(seq_row):
            if pd.isna(state):
                continue

            if prev_state is not None and state != prev_state:
                event_name = _transition_event_label(
                    seqdata, event_representation, event_representation_matrix, prev_state, state, n_states
                )

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

    eseq_list = EventSequenceData(sequences, dictionary, weights=weights, lengths=lengths)

    # ------------------------------------------------------------------
    # Attach state-consistent event colors when source is SequenceData.
    # This ensures event-sequence plots can reuse the same color semantics
    # as state-sequence plots (e.g., A/B/C/D keep identical colors).
    # ------------------------------------------------------------------
    event_color_map = {}
    try:
        state_color_map = getattr(seqdata, "color_map", None)  # keys are 1..K
        if isinstance(state_color_map, dict) and state_color_map:
            for ev in dictionary:
                s_code = None
                ev_str = str(ev)

                # Event is a raw state (e.g., "1", "A")
                if ev_str.isdigit():
                    s_code = int(ev_str)
                # Transition event (e.g., "1>3"): use destination state's color
                elif ">" in ev_str:
                    right = ev_str.split(">")[-1].strip()
                    if right.isdigit():
                        s_code = int(right)
                    elif hasattr(seqdata, "state_mapping") and right in seqdata.state_mapping:
                        s_code = seqdata.state_mapping[right]
                # Period event (e.g., "end1,begin3"): use begin state's color
                elif "begin" in ev_str:
                    marker = "begin"
                    idx = ev_str.rfind(marker)
                    tail = ev_str[idx + len(marker):].strip() if idx >= 0 else ""
                    # Keep only trailing digits if mixed punctuation exists
                    digits = "".join(ch for ch in tail if ch.isdigit())
                    if digits:
                        s_code = int(digits)
                    elif tail and hasattr(seqdata, "state_mapping") and tail in seqdata.state_mapping:
                        s_code = seqdata.state_mapping[tail]

                if s_code is not None and s_code in state_color_map:
                    event_color_map[ev] = state_color_map[s_code]
    except Exception:
        # If any mapping step fails, plotting falls back to default palettes.
        event_color_map = {}

    eseq_list.event_color_map = event_color_map
    return eseq_list


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
        counting_method="presence", search_constraint=constraint
    )
    counts_distinct = count_subsequence_occurrences(
        SubsequenceList(eseq, subsequences, pd.DataFrame(), constraint, "user"),
        counting_method="count", search_constraint=constraint
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
            counting_method="presence"
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
                        counting_method="presence"
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
    presence_matrix = count_subsequence_occurrences(subseq, counting_method="presence")
    
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
