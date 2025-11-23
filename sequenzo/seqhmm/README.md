# Sequenzo HMM Module

This module provides Hidden Markov Model (HMM) functionality for sequence analysis in Sequenzo, inspired by the seqHMM R package but implemented natively in Python.

## What are Hidden Markov Models?

**Hidden Markov Models (HMMs)** are statistical models that help us understand sequences of observations (like employment states over time) by assuming there are hidden underlying states that generate these observations.

### Basic Concepts (Beginner-Friendly Explanation)

Think of HMMs like this:
- **Hidden States**: The underlying states you can't directly observe (e.g., "stable career", "transitioning", "exploring")
- **Observations**: What you actually see (e.g., "employed", "unemployed", "student")
- **Transitions**: How likely you are to move from one hidden state to another
- **Emissions**: How likely each hidden state is to produce each observation

For example, if someone is in the "stable career" hidden state, they're more likely to be observed as "employed". But sometimes people transition to "exploring" states where they might try different things.

## Features

This module provides a comprehensive set of HMM tools:

### Basic HMM (Hidden Markov Model)
- Standard hidden Markov models for single-channel sequence data
- **What it does**: Finds hidden patterns in sequences (e.g., career trajectories, life courses)
- **When to use**: When you have sequences and want to discover underlying states

### Mixture HMM (MHMM)
- Multiple HMM submodels representing different clusters/groups
- **What it does**: Identifies different groups (clusters) of sequences, each with their own HMM pattern
- **When to use**: When you suspect there are different types of sequences (e.g., different career patterns for different people)

### Non-homogeneous HMM (NHMM)
- HMM with time-varying or covariate-dependent probabilities
- **What it does**: Allows transition probabilities to change over time or depend on covariates (e.g., age, education)
- **When to use**: When you want to model how sequences evolve differently based on characteristics or time

### Multichannel Support
- Handle multiple parallel sequences per subject (e.g., marriage, children, residence simultaneously)
- **What it does**: Models multiple dimensions of life simultaneously
- **When to use**: When you have multiple types of sequences for the same people

### Formula-Based Covariates
- Specify covariates using R-style formulas (e.g., "~ age + education")
- **What it does**: Makes it easy to include predictor variables in your model
- **When to use**: When you want to model how characteristics affect sequence patterns

### Model Simulation
- Generate synthetic sequences from HMM models
- **What it does**: Create artificial data that follows your model patterns
- **When to use**: For testing, validation, or understanding what your model predicts

### Model Comparison
- Compare models using AIC, BIC, and other criteria
- **What it does**: Helps you choose the best model (e.g., how many hidden states?)
- **When to use**: When you need to select between different model specifications

### Bootstrap Confidence Intervals
- Estimate uncertainty in model parameters
- **What it does**: Provides confidence intervals for your estimates
- **When to use**: When you need to know how reliable your estimates are

### Advanced Optimization
- Multiple optimization strategies (EM + global + local optimization)
- **What it does**: Finds better parameter estimates using sophisticated algorithms
- **When to use**: When standard EM algorithm doesn't converge well

## Quick Start

### Example 1: Basic HMM

```python
from sequenzo import SequenceData, load_dataset
from sequenzo.seqhmm import build_hmm, fit_model, predict, posterior_probs, plot_hmm

# Load example data
# This dataset tracks employment states from age 15 to 86
df = load_dataset('mvad')
seq = SequenceData(
    df, 
    time=range(15, 86),  # Time points (ages 15-85)
    states=['EM', 'FE', 'HE', 'JL', 'SC', 'TR']  # States: Employment, Further Education, etc.
)

# Build HMM with 4 hidden states
# Think of these as 4 different "career pattern types"
hmm = build_hmm(seq, n_states=4, random_state=42)

# Fit the model (learn the parameters)
# This finds the best transition and emission probabilities
hmm = fit_model(hmm, n_iter=100, tol=1e-2, verbose=True)

# Predict hidden states for each sequence
# This tells us which "career pattern type" each person follows
predicted_states = predict(hmm)

# Get posterior probabilities (how confident are we about each state at each time?)
posteriors = posterior_probs(hmm)
# Returns DataFrame with: id, time, state, probability

# Visualize the model
plot_hmm(hmm, which='all')  # Shows transition and emission probabilities
```

### Example 2: Mixture HMM (Clustering Sequences)

```python
from sequenzo.seqhmm import build_mhmm, fit_mhmm, predict_mhmm

# Build a Mixture HMM with 3 clusters
# This will find 3 different groups of sequences
mhmm = build_mhmm(
    seq, 
    n_clusters=3,      # 3 different groups
    n_states=4,        # Each group has 4 hidden states
    random_state=42
)

# Fit the model
mhmm = fit_mhmm(mhmm, n_iter=100, verbose=True)

# Predict which cluster each sequence belongs to
clusters = predict_mhmm(mhmm)
# Returns: ['Cluster 1', 'Cluster 2', ...] for each sequence

# Visualize
from sequenzo.seqhmm import plot_mhmm
plot_mhmm(mhmm, which='all')
```

### Example 3: HMM with Covariates (NHMM)

```python
from sequenzo.seqhmm import build_nhmm, fit_nhmm
import numpy as np
import pandas as pd

# Create covariate data
# Each sequence has characteristics (e.g., age at start, education level)
n_sequences = len(seq.sequences)
n_timepoints = max(len(s) for s in seq.sequences)

# Example: Time-varying covariate (just time itself)
X = np.zeros((n_sequences, n_timepoints, 1))
for i in range(n_sequences):
    for t in range(len(seq.sequences[i])):
        X[i, t, 0] = t  # Time covariate

# Build NHMM where transition probabilities depend on time
nhmm = build_nhmm(
    seq, 
    n_states=4,
    X=X,  # Covariate matrix
    random_state=42
)

# Fit the model
nhmm = fit_nhmm(nhmm, n_iter=100, verbose=True)

# Or use formula-based approach (easier!)
covariate_df = pd.DataFrame({
    'id': range(n_sequences),
    'age': np.random.randint(20, 50, n_sequences),  # Age at start
    'education': np.random.choice(['low', 'high'], n_sequences)
})

nhmm = build_nhmm(
    seq,
    n_states=4,
    emission_formula="~ age + education",  # Formula syntax
    data=covariate_df,
    id_var='id',
    time_var=None,  # Time-invariant covariates
    random_state=42
)
```

### Example 4: Model Simulation

```python
from sequenzo.seqhmm import simulate_hmm, simulate_mhmm
import numpy as np
import pandas as pd

# Simulate sequences from an HMM
# This creates artificial data that follows your model

# Define model parameters
initial_probs = np.array([0.5, 0.5])  # Start in state 0 or 1 with equal probability
transition_probs = np.array([
    [0.7, 0.3],  # From state 0: 70% stay, 30% move to state 1
    [0.3, 0.7]   # From state 1: 30% move to state 0, 70% stay
])
emission_probs = np.array([
    [0.9, 0.1],  # State 0 mostly emits observation 'A'
    [0.1, 0.9]   # State 1 mostly emits observation 'B'
])

# Simulate 10 sequences of length 20
sim = simulate_hmm(
    n_sequences=10,
    initial_probs=initial_probs,
    transition_probs=transition_probs,
    emission_probs=emission_probs,
    sequence_length=20,
    alphabet=['A', 'B'],
    random_state=42
)

# Simulate MHMM with formula-based covariates
data = pd.DataFrame({
    'covariate_1': np.random.rand(30),
    'covariate_2': np.random.choice(['A', 'B'], size=30)
})

# Coefficients determine how covariates affect cluster probabilities
coefs = np.array([
    [0, -1.5],        # Intercepts (first cluster is reference, set to 0)
    [0, 3.0],         # Effect of covariate_1 on cluster 2
    [0, -0.7]         # Effect of covariate_2_B on cluster 2
])

sim_mhmm = simulate_mhmm(
    n_sequences=30,
    n_clusters=2,
    initial_probs=[np.array([0.5, 0.5]), np.array([0.3, 0.7])],
    transition_probs=[
        np.array([[0.7, 0.3], [0.3, 0.7]]),
        np.array([[0.8, 0.2], [0.2, 0.8]])
    ],
    emission_probs=[
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.7, 0.3], [0.3, 0.7]])
    ],
    sequence_length=20,
    formula="~ covariate_1 + covariate_2",
    data=data,
    coefficients=coefs,
    alphabet=['A', 'B'],
    random_state=42
)
```

### Example 5: Model Comparison

```python
from sequenzo.seqhmm import build_hmm, fit_model, aic, bic, compare_models

# Try different numbers of hidden states
models = {}
for n_states in [3, 4, 5, 6]:
    hmm = build_hmm(seq, n_states=n_states, random_state=42)
    hmm = fit_model(hmm, verbose=False)
    models[f'{n_states}_states'] = hmm

# Compare using BIC (lower is better)
comparison = compare_models(list(models.values()), criterion='BIC')
print(f"Best model: {comparison['best_model']}")

# Or get AIC/BIC for individual models
for name, model in models.items():
    print(f"{name}: AIC={aic(model):.2f}, BIC={bic(model):.2f}")
```

## API Reference

### Core Functions

#### `build_hmm()` - Build a Basic HMM

Creates a Hidden Markov Model object from sequence data.

**Parameters:**
- `observations`: SequenceData object or list of sequences
- `n_states`: Number of hidden states (e.g., 3-6 for typical analyses)
- `initial_probs`: Optional custom initial state probabilities
- `transition_probs`: Optional custom transition probability matrix
- `emission_probs`: Optional custom emission probability matrix
- `state_names`: Optional names for hidden states (e.g., ['Stable', 'Transition', 'Exploration'])
- `random_state`: Random seed for reproducibility

**Returns:** HMM object

```python
hmm = build_hmm(seq, n_states=4, random_state=42)
```

#### `fit_model()` - Fit an HMM

Estimates model parameters using the EM (Expectation-Maximization) algorithm.

**Parameters:**
- `model`: HMM object to fit
- `n_iter`: Maximum number of iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-2)
- `verbose`: Whether to print progress (default: False)

**Returns:** Fitted HMM object

```python
hmm = fit_model(hmm, n_iter=100, tol=1e-2, verbose=True)
```

#### `predict()` - Predict Hidden States

Finds the most likely sequence of hidden states using the Viterbi algorithm.

**Parameters:**
- `model`: Fitted HMM object
- `newdata`: Optional new sequence data (uses training data if None)

**Returns:** List of predicted state sequences

```python
predicted_states = predict(hmm)
```

#### `posterior_probs()` - Compute Posterior Probabilities

Computes the probability of each hidden state at each time point.

**Parameters:**
- `model`: Fitted HMM object
- `newdata`: Optional new sequence data

**Returns:** DataFrame with columns: `id`, `time`, `state`, `probability`

```python
probs_df = posterior_probs(hmm)
# Example output:
#    id  time      state  probability
# 0   1     1     State 1         0.85
# 1   1     2     State 1         0.92
```

#### `plot_hmm()` - Visualize HMM Parameters

Creates visualizations of transition and emission probabilities.

**Parameters:**
- `model`: HMM object
- `which`: What to plot ('transition', 'emission', 'initial', or 'all')
- `figsize`: Figure size tuple
- `ax`: Optional matplotlib axes

```python
plot_hmm(hmm, which='all')
```

### Mixture HMM Functions

#### `build_mhmm()` - Build a Mixture HMM

Creates a Mixture HMM with multiple clusters.

```python
mhmm = build_mhmm(
    seq,
    n_clusters=3,      # Number of groups/clusters
    n_states=4,        # States per cluster
    random_state=42
)
```

#### `fit_mhmm()` - Fit a Mixture HMM

Fits the mixture model using EM algorithm.

```python
mhmm = fit_mhmm(mhmm, n_iter=100, verbose=True)
```

#### `predict_mhmm()` - Predict Cluster Assignments

Predicts which cluster each sequence belongs to.

```python
clusters = predict_mhmm(mhmm)
```

### Non-homogeneous HMM Functions

#### `build_nhmm()` - Build an NHMM with Covariates

Creates an NHMM where probabilities depend on covariates.

**Two ways to specify covariates:**

1. **Direct covariate matrix:**
```python
X = np.zeros((n_sequences, n_timepoints, n_covariates))
nhmm = build_nhmm(seq, n_states=4, X=X)
```

2. **Formula-based (easier!):**
```python
nhmm = build_nhmm(
    seq,
    n_states=4,
    emission_formula="~ age + education",
    data=covariate_df,
    id_var='id',
    random_state=42
)
```

#### `fit_nhmm()` - Fit an NHMM

Fits the NHMM using numerical optimization.

```python
nhmm = fit_nhmm(nhmm, n_iter=100, verbose=True)
```

### Model Comparison Functions

#### `aic()` - Akaike Information Criterion

```python
from sequenzo.seqhmm import aic, bic
print(f"AIC: {aic(hmm):.2f}")
print(f"BIC: {bic(hmm):.2f}")
```

#### `compare_models()` - Compare Multiple Models

```python
from sequenzo.seqhmm import compare_models
comparison = compare_models([hmm1, hmm2, hmm3], criterion='BIC')
```

### Simulation Functions

#### `simulate_hmm()` - Simulate from HMM

```python
sim = simulate_hmm(
    n_sequences=10,
    initial_probs=...,
    transition_probs=...,
    emission_probs=...,
    sequence_length=20,
    alphabet=['A', 'B'],
    random_state=42
)
```

#### `simulate_mhmm()` - Simulate from MHMM

Supports both fixed cluster probabilities and formula-based covariates:

```python
# With fixed cluster probabilities
sim = simulate_mhmm(
    n_sequences=10,
    n_clusters=2,
    initial_probs=[...],
    transition_probs=[...],
    emission_probs=[...],
    cluster_probs=np.array([0.6, 0.4]),
    sequence_length=20,
    random_state=42
)

# With formula-based covariates
sim = simulate_mhmm(
    n_sequences=30,
    n_clusters=2,
    initial_probs=[...],
    transition_probs=[...],
    emission_probs=[...],
    sequence_length=20,
    formula="~ covariate_1 + covariate_2",
    data=data,
    coefficients=coefs,
    random_state=42
)
```

### Bootstrap Functions

#### `bootstrap_model()` - Bootstrap Confidence Intervals

```python
from sequenzo.seqhmm import bootstrap_model

boot_results = bootstrap_model(
    hmm,
    n_sim=100,      # Number of bootstrap samples
    verbose=True
)

# Get confidence intervals
ci = boot_results['summary']['initial_probs']['ci_95']
```

## Dependencies

- `hmmlearn`: Python library for HMMs (foundation for basic HMM)
- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `scipy`: Optimization (for NHMM)
- `matplotlib`: Visualization

## Implementation Status

### Fully Implemented Features

- **Basic HMM**: Complete implementation with EM algorithm
- **Mixture HMM (MHMM)**: Full support with cluster estimation
- **Non-homogeneous HMM (NHMM)**: Complete with covariate support
- **Multichannel Support**: Handle multiple parallel sequences
- **Formula-Based Covariates**: R-style formula interface for NHMM and MHMM simulation
- **Model Simulation**: Simulate sequences from HMM and MHMM models
- **Model Comparison**: AIC, BIC, and model comparison tools
- **Bootstrap Confidence Intervals**: Nonparametric bootstrap for uncertainty estimation
- **Advanced Optimization**: EM + global + local optimization with multiple restarts
- **Visualization**: Comprehensive plotting functions for all model types

### ⚠️ Known Limitations

1. **Formula Interface**: Currently supports basic additive formulas (e.g., "~ x1 + x2"). Advanced features not yet supported:
   - Interactions (e.g., "~ x1 * x2")
   - Lag terms (e.g., "~ lag(x1)")
   - Transformations (e.g., "~ log(x1)")

3. **Multichannel NHMM**: Multichannel support is available for HMM and MHMM, but not yet for NHMM

## Notes

- This implementation is inspired by the seqHMM R package but designed with Python conventions
- Basic HMM is built on `hmmlearn`'s `CategoricalHMM`
- NHMM and advanced features use custom implementations
- The API is designed to be intuitive and beginner-friendly while remaining powerful for advanced users

## Getting Help

For more detailed documentation, see:
- `IMPLEMENTATION_STATUS.md`: Detailed implementation status and feature comparison
- Tutorial notebooks: Examples and case studies
- seqHMM R package documentation: For understanding the underlying methodology

## References

- **seqHMM R package**: https://github.com/helske/seqHMM
  - Helske & Helske (2019). Mixture Hidden Markov Models for Sequence Data: The seqHMM Package in R. *Journal of Statistical Software, 88*(3).
- **hmmlearn Python package**: https://github.com/hmmlearn/hmmlearn
