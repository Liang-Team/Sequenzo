# Sequenzo Utils Module

This module provides utility functions for statistical calculations used throughout Sequenzo. The main focus is on **weighted statistics functions** that match TraMineR's implementation.

## Overview

The `utils` module contains reusable statistical functions that were previously implemented inline throughout the codebase. By centralizing these functions, we ensure consistency, maintainability, and alignment with TraMineR's behavior.

## Weight Support Throughout Sequenzo

**Before diving into the utility functions, it's important to understand how weights are used throughout Sequenzo.** Weighted calculations are integrated across the entire package in four main areas:

### 1. SequenceData Objects

**How weights are stored:**

The `SequenceData` class stores weights as an attribute that can be accessed and used throughout the analysis:

```python
from sequenzo import SequenceData
import numpy as np
import pandas as pd

# Create sequence data with weights
data = pd.DataFrame({
    'ID': [1, 2, 3],
    'T1': ['A', 'B', 'A'],
    'T2': ['B', 'B', 'A']
})
weights = np.array([1.5, 2.0, 0.5])  # Different weights for each sequence

seqdata = SequenceData(
    data=data,
    time=['T1', 'T2'],
    states=['A', 'B'],
    weights=weights  # Pass weights during initialization
)

# Access weights
print(seqdata.weights)  # [1.5, 2.0, 0.5]

# Flatten weights for time-point level analysis
flattened = seqdata.flatten_weights()  # Repeats each weight for each time point
```

**Where weights are used in SequenceData:**

1. **Initialization**: Weights can be provided when creating a `SequenceData` object
2. **Property access**: `seqdata.weights` returns the weight array
3. **Flattening**: `flatten_weights()` method repeats weights across sequence length for time-point level analysis
4. **Uniqueness statistics**: `check_uniqueness_rate(weighted=True)` uses weights to calculate weighted uniqueness rates
5. **Cross-tabulation**: `get_xtabs()` method uses weights when `weighted=True`

**Example:**
```python
# Create weighted sequence data
seqdata = SequenceData(
    data=df,
    time=['Year1', 'Year2', 'Year3'],
    states=['Employed', 'Unemployed'],
    weights=np.array([1.2, 0.8, 1.5, 1.0])  # 4 sequences with different weights
)

# Check weighted uniqueness
stats = seqdata.check_uniqueness_rate(weighted=True)
print(f"Weighted uniqueness rate: {stats['weighted_uniqueness_rate']}")
```

### 2. Visualization Functions

**How weights are used in visualization:**

Most visualization functions in Sequenzo support weights through the `weights="auto"` parameter, which automatically uses `seqdata.weights` if available:

**Functions with weight support:**

1. **`plot_mean_time()`** - Weighted mean time spent in each state
2. **`plot_state_distribution()`** - Weighted state distributions over time
3. **`plot_modal_state()`** - Weighted modal state at each time point
4. **`plot_most_frequent_sequences()`** - Weighted sequence frequencies
5. **`plot_relative_frequency()`** - Weighted representative sequences
6. **`plot_sequence_index()`** - Sorting and grouping by weights
7. **`plot_transition_matrix()`** - Weighted transition rates

**Usage pattern:**

```python
from sequenzo import plot_mean_time, plot_state_distribution

# Automatic weight detection (recommended)
plot_mean_time(seqdata, weights="auto")  # Uses seqdata.weights if available

# Explicit weights
custom_weights = np.array([1.0, 2.0, 1.5])
plot_mean_time(seqdata, weights=custom_weights)

# No weights (unweighted)
plot_mean_time(seqdata, weights=None)
```

**How weights work in visualization:**

1. **Frequency plots**: Sequences are counted with their weights:
   ```python
   # In plot_most_frequent_sequences()
   for seq, w in zip(sequences, weights):
       agg[tuple(seq)] = agg.get(tuple(seq), 0.0) + float(w)
   ```

2. **State distributions**: Weighted proportions at each time point:
   ```python
   # In plot_state_distribution()
   W = np.repeat(weights[:, None], n_time_points, axis=1)  # Broadcast weights
   weighted_counts = (seq_df == state) * W  # Weighted counting
   ```

3. **Mean time**: Weighted average time spent in each state:
   ```python
   # Uses weighted_mean() internally or np.average() with weights
   mean_time = np.average(time_in_state, weights=weights)
   ```

**Example:**
```python
# Create sequence data with weights
seqdata = SequenceData(
    data=df,
    time=['T1', 'T2', 'T3'],
    states=['A', 'B', 'C'],
    weights=np.array([1.0, 2.0, 1.5, 0.5])  # 4 sequences
)

# Visualizations automatically use weights
plot_state_distribution(seqdata, weights="auto")  # Weighted distributions
plot_mean_time(seqdata, weights="auto")  # Weighted mean times
plot_most_frequent_sequences(seqdata, weights="auto")  # Weighted frequencies
```

### 3. Distance Matrix Computation

**How weights are used in distance calculations:**

The `get_distance_matrix()` function uses weights in two main ways:

1. **Substitution cost matrix calculation**: When `weighted=True` (default), state distributions used to compute substitution costs account for sequence weights
2. **CHI2 and EUCLID methods**: These methods use weighted state distributions

**Usage:**

```python
from sequenzo import get_distance_matrix

# Create sequence data with weights
seqdata = SequenceData(
    data=df,
    time=['T1', 'T2', 'T3'],
    states=['A', 'B', 'C'],
    weights=np.array([1.0, 2.0, 1.5])
)

# Compute distance matrix with weighted substitution costs
distance_matrix = get_distance_matrix(
    seqdata=seqdata,
    method="OM",
    sm="TRATE",  # Transition rates computed with weights
    weighted=True  # Use weights in state distribution calculations
)

# For CHI2 method, weights affect state distributions
chi2_distances = get_distance_matrix(
    seqdata=seqdata,
    method="CHI2",
    weighted=True  # Weighted chi-square distances
)
```

**How weights affect distance calculations:**

1. **Substitution cost matrices** (`get_substitution_cost_matrix()`):
   - When `weighted=True`, transition rates and state frequencies account for sequence weights
   - Example: `TRATE` method computes weighted transition probabilities
   ```python
   # Weighted transition rate calculation
   # P(state_j | state_i) = sum(weights * transitions) / sum(weights)
   ```

2. **CHI2 distance**: Uses weighted state distributions to compute chi-square statistics

3. **EUCLID distance**: Uses weighted state distributions for Euclidean distance calculation

**Example:**
```python
# Weighted substitution cost matrix
sm = get_substitution_cost_matrix(
    seqdata=seqdata,
    method="TRATE",
    weighted=True  # Transition rates account for weights
)

# Weighted distance matrix
dist = get_distance_matrix(
    seqdata=seqdata,
    method="OM",
    sm=sm,
    weighted=True
)
```

### 4. Tree Analysis and Clustering

**How weights are used in tree analysis:**

1. **Sequence trees** (`build_sequence_tree()`):
   - Extracts weights from `seqdata.weights` when `weighted=True`
   - Uses weights in distance association tests and tree splitting

2. **Distance trees** (`build_distance_tree()`):
   - Accepts weights as parameter
   - Uses weights in pseudo-variance calculations and permutation tests

3. **Distance association** (`compute_distance_association()`):
   - Computes weighted inertia (sum of weighted squared distances)
   - Formula: `SCtot = sum(w_i * w_j * d_ij)` for all pairs (i,j)

**Example:**
```python
from sequenzo.tree_analysis import build_sequence_tree

# Build tree with weighted analysis
tree_result = build_sequence_tree(
    seqdata=seqdata,
    predictors=predictors_df,
    weighted=True  # Uses seqdata.weights automatically
)
```

**How weights work in clustering:**

1. **Cluster quality**: Weighted cluster quality indices (ASWw, etc.)
2. **Medoid selection**: Weighted medoid computation considers sequence weights
3. **Pseudo-variance**: Weighted pseudo-variance for distance matrices:
   ```python
   # Formula: sum(w_i * w_j * d_ij) / (sum(weights))^2
   ```

### Summary: Weight Usage Patterns

**Pattern 1: Automatic weight detection (`weights="auto"`)**
```python
# Most visualization and analysis functions support this
function(seqdata, weights="auto")  # Uses seqdata.weights if available
```

**Pattern 2: Explicit weights**
```python
# Pass weights directly
function(seqdata, weights=np.array([1.0, 2.0, 1.5]))
```

**Pattern 3: Weighted parameter**
```python
# Boolean flag to enable/disable weighting
function(seqdata, weighted=True)  # Uses seqdata.weights when True
```

**Pattern 4: No weights (unweighted)**
```python
# Explicitly disable weights
function(seqdata, weights=None)
# or
function(seqdata, weighted=False)
```

### Best Practices

1. **Store weights in SequenceData**: Always provide weights when creating `SequenceData` objects, then use `weights="auto"` in functions
2. **Consistent weighting**: Use the same weights throughout your analysis pipeline
3. **Weight normalization**: Weights don't need to sum to 1, but be aware of how this affects interpretations
4. **Missing weights**: If weights are not provided, functions default to equal weights (all sequences weighted equally)

---

## Weighted Statistics Functions

**Now that we understand how weights are used throughout Sequenzo, let's examine the utility functions that standardize weighted statistical calculations.**

### Background: Previous Implementation

Before the creation of this module, weighted statistical calculations were implemented **inline** within various functions throughout Sequenzo. For example:

**In `sequenzo/sequence_characteristics/cross_sectional_indicators.py`:**
- The `get_mean_time_in_states()` function calculated weighted means directly:
  ```python
  # Line 82: Inline weighted mean calculation
  mtime = np.sum(istatd_values * weights[:, np.newaxis], axis=0) / wtot
  
  # Lines 94-95: Inline weighted variance calculation
  var = np.sum(weights[:, np.newaxis] * (vcent ** 2), axis=0) * wtot / (wtot ** 2 - w2tot)
  ```

**In `sequenzo/visualization/plot_mean_time.py`:**
- The `_compute_mean_time()` function used `np.average()` with weights for weighted calculations

**In other modules:**
- Various functions implemented weighted calculations using NumPy operations directly
- Each implementation had slight variations and potential inconsistencies
- No centralized place to update or maintain weighted calculation logic

### Current Implementation: Unified Utility Functions

The `weighted_stats.py` module provides **standardized, reusable functions** that match TraMineR's implementation exactly. These functions can be used throughout Sequenzo to ensure consistency and maintainability.

**Why this matters:**

Given that weights are used extensively across SequenceData, visualization, distance matrices, and tree analysis (as described above), having standardized weighted statistics functions ensures:

1. **Consistency**: All weighted calculations use the same implementation
2. **Maintainability**: Changes to weighted calculation logic only need to be made in one place
3. **TraMineR Compatibility**: These functions match TraMineR's implementation exactly
4. **Code Reusability**: Avoid duplicating weighted calculation code throughout the codebase

### Available Functions

#### 1. `weighted_mean()`

Computes the weighted mean of a vector.

**Corresponds to R function:** `wtd.mean()` in TraMineR-wtd-stats.R

**Usage:**
```python
from sequenzo import weighted_mean
import numpy as np

x = np.array([1, 2, 3, 4, 5])
weights = np.array([1, 2, 1, 2, 1])
mean = weighted_mean(x, weights=weights)
```

**Parameters:**
- `x`: Input vector of values
- `weights`: Optional weights for each observation. If None, computes unweighted mean.
- `normwt`: Normalization flag (kept for API compatibility, but ignored)
- `na_rm`: If True, remove NA/NaN values before computation (default: True)

**Returns:** Weighted mean as a float

#### 2. `weighted_variance()`

Computes the weighted variance of a vector.

**Corresponds to R function:** `wtd.var()` in TraMineR-wtd-stats.R

**Usage:**
```python
from sequenzo import weighted_variance
import numpy as np

x = np.array([1, 2, 3, 4, 5])
weights = np.array([1, 2, 1, 2, 1])
variance = weighted_variance(x, weights=weights, method='unbiased')
```

**Parameters:**
- `x`: Input vector of values
- `weights`: Optional weights for each observation. If None, computes unweighted variance.
- `normwt`: If True, normalize weights so they sum to length(x) (default: False)
- `na_rm`: If True, remove NA/NaN values before computation (default: True)
- `method`: Method for variance calculation:
  - `'unbiased'`: Unbiased frequency weights (uses n-1 denominator) - **default**
  - `'ML'`: Maximum likelihood (uses n denominator)

**Returns:** Weighted variance as a float

#### 3. `weighted_five_number_summary()`

Computes the weighted five-number summary (minimum, Q1, median, Q3, maximum).

**Corresponds to R function:** `wtd.fivenum.tmr()` in TraMineR-wtd-stats.R

**Usage:**
```python
from sequenzo import weighted_five_number_summary
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
weights = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1])
fivenum = weighted_five_number_summary(x, weights=weights)
# Returns: [min, Q1, median, Q3, max]
```

**Parameters:**
- `x`: Input vector of values
- `weights`: Optional weights for each observation. If None, uses equal weights.
- `na_rm`: If True, remove NA/NaN values before computation (default: True)

**Returns:** NumPy array of length 5 containing [min, Q1, median, Q3, max]

### Why Use These Functions?

#### Benefits

1. **Consistency**: All weighted calculations use the same implementation, ensuring consistent results across Sequenzo
2. **Maintainability**: Changes to weighted calculation logic only need to be made in one place
3. **TraMineR Compatibility**: These functions match TraMineR's implementation exactly, ensuring compatibility with R-based workflows
4. **Code Reusability**: Avoid duplicating weighted calculation code throughout the codebase
5. **Documentation**: Centralized documentation makes it easier for users to understand weighted calculations

#### Migration from Inline Calculations

If you're working with code that uses inline weighted calculations, consider refactoring to use these utility functions:

**Before (inline):**
```python
# Inline weighted mean
wtot = np.sum(weights)
mtime = np.sum(values * weights[:, np.newaxis], axis=0) / wtot
```

**After (using utility function):**
```python
from sequenzo.utils import weighted_mean
mtime = weighted_mean(values, weights=weights)
```

## TraMineR Reference

These functions are based on TraMineR's weighted statistics implementation:

- **Source File**: `TraMineR-wtd-stats.R`
- **Original Package**: Based on Hmisc package functions (included in TraMineR to avoid dependencies)
- **GitHub**: https://github.com/cran/TraMineR/blob/master/R/TraMineR-wtd-stats.R

The Python implementations match the R functions' behavior exactly, including:
- Handling of missing values
- Weight normalization options
- Variance calculation methods
- Five-number summary interpolation for unequal weights

## Usage Examples

### Example 1: Basic Weighted Mean

```python
from sequenzo import weighted_mean
import numpy as np

# Sample data
values = np.array([10, 20, 30, 40, 50])
weights = np.array([1, 2, 3, 2, 1])

# Calculate weighted mean
result = weighted_mean(values, weights=weights)
print(f"Weighted mean: {result}")
# Output: Weighted mean: 30.0
```

### Example 2: Weighted Variance with Different Methods

```python
from sequenzo import weighted_variance
import numpy as np

values = np.array([1, 2, 3, 4, 5])
weights = np.array([1, 2, 1, 2, 1])

# Unbiased variance (default)
var_unbiased = weighted_variance(values, weights=weights, method='unbiased')

# Maximum likelihood variance
var_ml = weighted_variance(values, weights=weights, method='ML')

print(f"Unbiased variance: {var_unbiased}")
print(f"ML variance: {var_ml}")
```

### Example 3: Weighted Five-Number Summary

```python
from sequenzo import weighted_five_number_summary
import numpy as np

# Sample data with unequal weights
values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
weights = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1])

# Calculate five-number summary
summary = weighted_five_number_summary(values, weights=weights)
print(f"Five-number summary: {summary}")
print(f"Min: {summary[0]}, Q1: {summary[1]}, Median: {summary[2]}, Q3: {summary[3]}, Max: {summary[4]}")
```

### Example 4: Handling Missing Values

```python
from sequenzo import weighted_mean
import numpy as np

# Data with missing values
values = np.array([1, 2, np.nan, 4, 5])
weights = np.array([1, 2, 1, 2, 1])

# Automatically handles NaN (na_rm=True by default)
result = weighted_mean(values, weights=weights, na_rm=True)
print(f"Weighted mean (NaN removed): {result}")

# Or keep NaN
result_with_nan = weighted_mean(values, weights=weights, na_rm=False)
print(f"Weighted mean (with NaN): {result_with_nan}")
```

## Import Options

You can import these functions in several ways:

```python
# Option 1: Direct import from sequenzo (recommended)
from sequenzo import weighted_mean, weighted_variance, weighted_five_number_summary

# Option 2: Import from utils submodule
from sequenzo.utils import weighted_mean, weighted_variance, weighted_five_number_summary

# Option 3: Wildcard import
from sequenzo import *
```

## Implementation Details

### Weighted Mean Formula

The weighted mean is calculated as:
```
weighted_mean = sum(weights * x) / sum(weights)
```

### Weighted Variance Formula

For unbiased frequency weights:
```
xbar = sum(weights * x) / sum(weights)
variance = sum(weights * (x - xbar)^2) / (sum(weights) - 1)
```

For maximum likelihood:
```
variance = sum(weights * (x - xbar)^2) / sum(weights)
```

### Weighted Five-Number Summary

The five-number summary uses weighted quantile interpolation for unequal weights:
- For equal weights: Uses standard fivenum positions
- For unequal weights: Uses interpolated index calculation based on cumulative weights

## Notes

- All functions handle edge cases such as:
  - Zero weights
  - All weights equal to zero
  - Single observation
  - Missing values (when `na_rm=True`)
  
- The functions are designed to match TraMineR's behavior exactly, ensuring compatibility with R-based sequence analysis workflows.

- These functions are used internally throughout Sequenzo, but are also available for direct use by users who need weighted statistical calculations.

## See Also

- `sequenzo/sequence_characteristics/cross_sectional_indicators.py` - Example of functions that could use these utilities
- `sequenzo/visualization/plot_mean_time.py` - Example of weighted calculations in visualization
- `sequenzo/define_sequence_data.py` - SequenceData class with weights support
- `sequenzo/dissimilarity_measures/get_distance_matrix.py` - Weighted distance calculations
- `sequenzo/dissimilarity_measures/get_substitution_cost_matrix.py` - Weighted substitution costs
- TraMineR documentation: https://cran.r-project.org/package=TraMineR
