# Sequence Comparison and Difference Analysis

This module provides functions for comparing groups of sequences and analyzing how differences between groups evolve across positions.

## Overview

The `compare_differences` module implements two main types of analysis:

1. **Position-wise Discrepancy Analysis** (`compare_groups_across_positions`): Analyzes how differences between groups of sequences evolve along the positions using sliding windows.

2. **Sequence Comparison Tests** (`compare_groups_overall`, `compute_likelihood_ratio_test`, `compute_bayesian_information_criterion_test`): Performs likelihood ratio tests and Bayesian information criterion tests to compare two groups of sequences.

## Functions

### compare_groups_across_positions()

**Position-wise discrepancy analysis between groups of sequences**

Analyzes how the part of discrepancy explained by the group variable evolves along the position axis. At each position t, it computes distances over a time-window (t + cmprange[0], t + cmprange[1]) and derives the explained discrepancy on that window.

**Corresponds to TraMineR function:** `seqdiff()`

```python
from sequenzo.compare_differences import compare_groups_across_positions

result = compare_groups_across_positions(
    seqdata,           # Sequence data (DataFrame)
    group,             # Grouping variable
    cmprange=(0, 1),   # Sliding window range
    seqdist_args={'method': 'LCS', 'norm': 'auto'},
    with_missing=False,
    weighted=True,
    squared=False
)
```

**Returns:**
- `stat`: DataFrame with Pseudo F, Pseudo R2, Bartlett, and Levene statistics
- `discrepancy`: DataFrame with discrepancy values per group and total

**Visualization:**
```python
from sequenzo.compare_differences import plot_group_differences_across_positions

# Plot Pseudo R2 over positions
plot_group_differences_across_positions(result, stat='Pseudo R2')

# Plot discrepancy per group
plot_group_differences_across_positions(result, stat='discrepancy')

# Plot two statistics on dual y-axes
plot_group_differences_across_positions(result, stat=['Pseudo R2', 'Levene'])
```

### compare_groups_overall()

**Compare sets of sequences using LRT and BIC**

Compares two groups of sequences by computing likelihood ratio test (LRT) statistics and Bayesian Information Criterion (BIC). The comparison can be done either between two separate sequence datasets or between groups within a single dataset.

**Corresponds to TraMineRextras function:** `seqCompare()`

```python
from sequenzo.compare_differences import compare_groups_overall

results = compare_groups_overall(
    seqdata,              # First sequence dataset
    seqdata2=None,        # Second dataset (or use group)
    group=None,           # Grouping variable (if seqdata2 is None)
    set_var=None,         # Stratification variable
    s=100,                # Bootstrap sample size
    seed=36963,           # Random seed
    stat="all",           # "LRT", "BIC", or "all"
    squared="LRTonly",    # Distance squaring method
    weighted=True,        # Use sequence weights
    method="OM"           # Distance method
)
```

**Returns:**
- NumPy array with columns depending on `stat` parameter:
  - `stat="LRT"`: LRT statistic and p-value
  - `stat="BIC"`: Delta BIC and Bayes Factor
  - `stat="all"`: All four statistics

### compute_likelihood_ratio_test()

**Convenience wrapper for likelihood ratio test**

```python
from sequenzo.compare_differences import compute_likelihood_ratio_test

results = compute_likelihood_ratio_test(seqdata, group=group, method="LCS", s=100)
# Returns: [LRT, p-value]
```

### compute_bayesian_information_criterion_test()

**Convenience wrapper for Bayesian information criterion**

```python
from sequenzo.compare_differences import compute_bayesian_information_criterion_test

results = compute_bayesian_information_criterion_test(seqdata, group=group, method="LCS", s=100)
# Returns: [Delta BIC, Bayes Factor]
```

## Examples

### Example 1: Position-wise Analysis

```python
import pandas as pd
import numpy as np
from sequenzo.compare_differences import (
    compare_groups_across_positions,
    plot_group_differences_across_positions
)

# Create sequence data
seqdata = pd.DataFrame({
    'pos1': ['A', 'A', 'B', 'B'],
    'pos2': ['A', 'B', 'B', 'C'],
    'pos3': ['B', 'B', 'C', 'C'],
    'pos4': ['B', 'C', 'C', 'C']
})

# Define groups
group = np.array([1, 1, 2, 2])

# Run analysis with centered sliding windows of length 5
result = compare_groups_across_positions(
    seqdata, 
    group=group, 
    cmprange=(-2, 2),
    seqdist_args={'method': 'LCS', 'norm': 'auto'}
)

# Print results
print(result['stat'])
print(result['discrepancy'])

# Plot Pseudo R2
plot_group_differences_across_positions(result, stat='Pseudo R2')
```

### Example 2: Compare Two Groups

```python
import pandas as pd
import numpy as np
from sequenzo.compare_differences import (
    compute_likelihood_ratio_test,
    compute_bayesian_information_criterion_test
)

# Create sequence data
seqdata = pd.DataFrame({
    'pos1': ['A', 'A', 'B', 'B', 'A', 'A'],
    'pos2': ['A', 'B', 'B', 'C', 'A', 'B'],
    'pos3': ['B', 'B', 'C', 'C', 'B', 'B']
})

# Define groups
group = np.array([1, 1, 1, 2, 2, 2])

# Likelihood ratio test
lrt_result = compute_likelihood_ratio_test(seqdata, group=group, method="LCS", s=50)
print(f"LRT: {lrt_result[0, 0]:.4f}")
print(f"p-value: {lrt_result[0, 1]:.4f}")

# Bayesian information criterion
bic_result = compute_bayesian_information_criterion_test(seqdata, group=group, method="LCS", s=50)
print(f"Delta BIC: {bic_result[0, 0]:.4f}")
print(f"Bayes Factor: {bic_result[0, 1]:.4f}")
```

### Example 3: Stratified Comparison

```python
import pandas as pd
import numpy as np
from sequenzo.compare_differences import compare_groups_overall

# Create sequence data
seqdata = pd.DataFrame({
    'pos1': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'pos2': ['A', 'B', 'B', 'C', 'A', 'B', 'B', 'C'],
    'pos3': ['B', 'B', 'C', 'C', 'B', 'B', 'C', 'C']
})

# Define groups and sets
group = np.array([1, 1, 2, 2, 1, 1, 2, 2])
set_var = np.array([1, 1, 1, 1, 2, 2, 2, 2])

# Compare within each set
results = compare_groups_overall(
    seqdata, 
    group=group, 
    set_var=set_var,
    method="LCS", 
    s=50
)

print("Comparison for each set:")
print(results)
```

## Key Parameters

### Distance Methods

Both functions use `get_distance_matrix()` for computing distances. Common methods include:
- `"OM"`: Optimal Matching
- `"LCS"`: Longest Common Subsequence
- `"HAM"`: Hamming distance
- `"DHD"`: Dynamic Hamming Distance

### Normalization

When using `compare_groups_across_positions`, you can specify normalization in `seqdist_args`:
```python
seqdist_args = {
    'method': 'LCS',
    'norm': 'auto'  # or 'maxlength', 'gmean', 'maxdist', etc.
}
```

### Sliding Windows (compare_groups_across_positions)

The `cmprange` parameter defines the sliding window:
- `cmprange=(0, 1)`: Compare position t with t+1
- `cmprange=(-2, 2)`: Use centered window of length 5
- `cmprange=(-1, 1)`: Use centered window of length 3

### Sampling Strategy (compare_groups_overall)

- `s=0`: No sampling, use all sequences
- `s>0`: Bootstrap sampling with specified sample size
- `opt=1`: Compute distances per sample (less memory)
- `opt=2`: Compute full distance matrix once (faster)

## Interpretation

### compare_groups_across_positions Statistics

- **Pseudo F**: F-statistic for testing group differences
- **Pseudo R2**: Proportion of discrepancy explained by groups (0-1)
- **Bartlett**: Test for homogeneity of variances
- **Levene**: Robust test for homogeneity of variances

Higher Pseudo R2 indicates stronger group differences at that position.

### compare_groups_overall Statistics

- **LRT**: Likelihood ratio test statistic (higher = more different)
- **p-value**: Probability under null hypothesis (< 0.05 = significant)
- **Delta BIC**: BIC difference (positive = evidence for differences)
- **Bayes Factor**: Strength of evidence (>1 = support for differences)
  - 1-3: Weak evidence
  - 3-10: Moderate evidence
  - >10: Strong evidence

## References

Studer, M., Ritschard, G., Gabadinho, A., & Müller, N. S. (2011). Discrepancy analysis of state sequences. Sociological methods & research, 40(3), 471-510.

Liao, T. F., & Fasang, A. E. (2021). Comparing groups of life-course sequences using the Bayesian information criterion and the likelihood-ratio test. Sociological Methodology, 51(1), 44-85.

## TraMineR Correspondence

| Sequenzo Function | TraMineR Function | TraMineRextras Function |
|------------------|-------------------|-------------------------|
| `compare_groups_across_positions()` | `seqdiff()` | - |
| `plot_group_differences_across_positions()` | `plot.seqdiff()` | - |
| `print_group_differences_across_positions()` | `print.seqdiff()` | - |
| `compare_groups_overall()` | - | `seqCompare()` |
| `compute_likelihood_ratio_test()` | - | `seqLRT()` |
| `compute_bayesian_information_criterion_test()` | - | `seqBIC()` |
