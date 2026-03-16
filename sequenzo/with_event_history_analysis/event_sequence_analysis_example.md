# Event Sequence Analysis Examples

This document provides examples of how to use the event sequence analysis functions in Sequenzo.

## Basic Usage

### Creating Event Sequences from TSE Format

```python
import pandas as pd
import numpy as np
from sequenzo.with_event_history_analysis import create_event_sequences

# Create TSE format data
tse_data = pd.DataFrame({
    'id': [1, 1, 1, 2, 2, 2, 3, 3],
    'timestamp': [18, 22, 26, 17, 20, 21, 19, 23],
    'event': ['EnterUni', 'Graduate', 'StartJob', 'EnterUni', 'Graduate', 'StartJob', 'EnterUni', 'Graduate']
})

# Create event sequence object
eseq = create_event_sequences(data=tse_data)

# Or specify columns explicitly
eseq = create_event_sequences(id=tse_data['id'], 
                              timestamp=tse_data['timestamp'],
                              event=tse_data['event'])
```

### Creating Event Sequences from State Sequences

```python
from sequenzo import SequenceData
from sequenzo.with_event_history_analysis import create_event_sequences

# Create state sequence
state_seq = SequenceData(
    data=df,
    time=['Year1', 'Year2', 'Year3', 'Year4'],
    states=['Student', 'Full-time', 'Part-time'],
    id_col='PersonID'
)

# Convert to event sequence using transition method
eseq = create_event_sequences(data=state_seq, tevent="transition")

# Or use state method (one event per state entry)
eseq = create_event_sequences(data=state_seq, tevent="state")
```

### Finding Frequent Subsequences

```python
from sequenzo.with_event_history_analysis import find_frequent_subsequences, EventSequenceConstraint

# Find subsequences with at least 20 occurrences
fsubseq = find_frequent_subsequences(eseq, min_support=20)

# Find subsequences with at least 1% support
fsubseq = find_frequent_subsequences(eseq, pmin_support=0.01)

# With time constraints
constraint = EventSequenceConstraint(
    max_gap=2.0,      # Maximum 2 time units between events
    window_size=10.0, # Maximum 10 time units for subsequence
    age_min=18,       # Events must start at age 18+
    age_max=30        # Events must end by age 30
)
fsubseq = find_frequent_subsequences(eseq, min_support=10, constraint=constraint)

# View results
print(fsubseq.data.head())
print(fsubseq.subsequences[0])  # First subsequence
```

### Searching for Specific Subsequences

```python
# Search for specific subsequence patterns
str_subseq = ["(EnterUni)-(Graduate)", "(Graduate)-(StartJob)"]
fsubseq = find_frequent_subsequences(eseq, str_subseq=str_subseq)

# Format: transitions separated by "-", simultaneous events in parentheses
# Example: "(A,B)-(C)" means events A and B occur simultaneously, then C
```

### Counting Subsequence Occurrences

```python
from sequenzo.with_event_history_analysis import count_subsequence_occurrences

# Count presence/absence (method 1: COBJ)
presence_matrix = count_subsequence_occurrences(fsubseq, method="presence")
# presence_matrix[i, j] = 1 if sequence i contains subsequence j, else 0

# Count distinct occurrences (method 2: CDIST_O)
count_matrix = count_subsequence_occurrences(fsubseq, method="count")
# count_matrix[i, j] = number of times sequence i contains subsequence j

# Use as features for machine learning
features = pd.DataFrame(
    count_matrix,
    index=[seq.id for seq in eseq.sequences],
    columns=[f"Subseq_{i}" for i in range(len(fsubseq))]
)
```

### Comparing Groups

```python
from sequenzo.with_event_history_analysis import compare_groups

# Define groups
groups = np.array(['Male', 'Female', 'Male', 'Female', 'Male'])

# Find discriminating subsequences
discr = compare_groups(fsubseq, groups, method="chisq", pvalue_limit=0.05)

# View results
print(discr.data[['Support', 'p.value', 'statistic', 'Freq.Male', 'Freq.Female']])

# With Bonferroni correction
discr = compare_groups(fsubseq, groups, method="bonferroni", pvalue_limit=0.05)
```

### Visualizing Event Sequences

```python
from sequenzo.with_event_history_analysis import plot_event_sequences, plot_subsequence_frequencies

# Plot event sequences as index plot
plot_event_sequences(eseq, type="index", top_n=20)

# Plot event sequences as parallel coordinates
plot_event_sequences(eseq, type="parallel")

# Plot subsequence frequencies
plot_subsequence_frequencies(fsubseq, top_n=10)

# Save plots
plot_event_sequences(eseq, type="index", save_as="event_sequences.png")
plot_subsequence_frequencies(fsubseq, top_n=10, save_as="subseq_freq.png")
```

## Complete Workflow Example

```python
import pandas as pd
import numpy as np
from sequenzo.with_event_history_analysis import (
    create_event_sequences, find_frequent_subsequences, compare_groups, 
    count_subsequence_occurrences, plot_event_sequences, plot_subsequence_frequencies,
    EventSequenceConstraint
)

# Step 1: Load or create event data
tse_data = pd.DataFrame({
    'id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    'timestamp': [18, 22, 26, 17, 20, 21, 19, 23, 27, 18, 22],
    'event': ['EnterUni', 'Graduate', 'StartJob', 'EnterUni', 'Graduate', 
              'StartJob', 'EnterUni', 'Graduate', 'StartJob', 'EnterUni', 'Graduate']
})

# Step 2: Create event sequence object
eseq = create_event_sequences(data=tse_data)

# Step 3: Visualize event sequences
plot_event_sequences(eseq, type="index", top_n=10)

# Step 4: Find frequent subsequences
fsubseq = find_frequent_subsequences(eseq, min_support=2)  # At least 2 sequences
print(f"Found {len(fsubseq)} frequent subsequences")
print(fsubseq.data)

# Step 5: Visualize subsequence frequencies
plot_subsequence_frequencies(fsubseq, top_n=10)

# Step 6: Count occurrences in each sequence
counts = count_subsequence_occurrences(fsubseq, method="presence")
print(f"Count matrix shape: {counts.shape}")

# Step 7: Compare groups (if you have group labels)
groups = np.array(['GroupA', 'GroupA', 'GroupB', 'GroupB'])
if len(groups) == len(eseq):
    discr = compare_groups(fsubseq, groups, pvalue_limit=0.1)
    print(f"Found {len(discr)} discriminating subsequences")
    print(discr.data[['Support', 'p.value']])
```

## Notes

1. **Event Sequence Format**: Events are stored with timestamps. The same event can occur multiple times for the same individual.

2. **Subsequence Format**: When searching for specific subsequences, use the format:
   - `"(A)-(B)"` for A followed by B
   - `"(A,B)-(C)"` for A and B simultaneously, then C
   - Transitions are separated by `-`, simultaneous events by `,`

3. **Counting Methods**: Different counting methods serve different purposes:
   - `presence` (COBJ): Binary indicator (0/1) - useful for classification
   - `count` (CDIST_O): Number of occurrences - useful for frequency analysis
   - `CWIN`, `CMINWIN`, `CDIST`: Advanced methods for time-constrained counting

4. **Performance**: For large datasets, consider:
   - Using `min_support` to filter rare subsequences early
   - Limiting `max_k` to restrict subsequence length
   - Using time constraints to focus on relevant periods

5. **Compatibility**: These functions are designed to be compatible with TraMineR's event sequence analysis functions, so results should be similar (though not identical due to implementation differences).
