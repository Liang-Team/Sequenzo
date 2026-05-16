# SAMM (Sequence Analysis Multi-state Model) Usage Guide

## Overview

This module provides Python tools for analyzing sequences using a multi-state perspective, creating person-period datasets that can be used for event history analysis. It implements the **SAMM** workflow from the R package **[TraMineRextras](https://cran.r-project.org/package=TraMineRextras)** (not TraMineR core).

- **TraMineRextras**: `seqsamm()`, `seqsammseq()`, `seqsammeha()`, S3 `plot.SAMM()`
- **TraMineR** (input/plotting only): `seqdef()` for sequence data; `seqplot()` inside `plot.SAMM()`

In Sequenzo, build sequences with `SequenceData` (same role as TraMineR `seqdef()`).

### Python ↔ R function map

| Python (Sequenzo) | R (TraMineRextras) |
|-------------------|--------------------|
| `sequence_analysis_multi_state_model()` | `seqsamm()` |
| `seqsamm` (alias) | `seqsamm()` |
| `SAMM` | object of class `"SAMM"` |
| `plot_samm()` | `plot.SAMM()` → TraMineR `seqplot()` |
| `seqsammseq()` | `seqsammseq()` |
| `set_typology()` | typology set inside `seqsammeha()` in R; Sequenzo adds a separate helper |
| `seqsammeha()` | `seqsammeha()` |

Column naming: primary column is `spell.time` (R); `spell_time` is an alias. Subsequence columns `s.1`, `s.2`, ... hold numeric state codes.

## Void, missing, and R parity

TraMineR distinguishes two notions on a `seqdef` object:

| Concept | R attribute | Default | Role in `seqsamm()` |
|---------|-------------|---------|---------------------|
| **Void** | `attr(seqdata, "void")` | `"%"` | Subsequence windows containing void are **dropped** (`maxmiss = 0`) |
| **Missing** | `attr(seqdata, "nr")` | `NA` | **Not** filtered by `seqsamm()` unless it equals the void symbol |

Sequenzo `SequenceData` mirrors TraMineR via **`void=`** (default `"%"`, like `seqdef()`):

```python
seq = SequenceData(
    df,
    time=cols,
    states=["就业", "失业", "%"],   # void symbol must be in states if it appears in data
    labels=["Employed", "Unemployed", "Out of window"],
    void="%",                        # default; TraMineR attr(seqdata, "void")
)
# seq.void_code is the integer code used internally by seqsamm
```

- **`void="%"`** (default) + **`"%"` in `states`** → `seq.void_code` is set → `seqsamm` drops subsequences containing void (R parity).
- **`void=None`** → no void metadata; no void-based row dropping (typical for complete tables with no `%` padding).
- **Missing** (auto-added from `NaN`) ≠ void; `seqsamm` does not drop rows just because of missing unless you misuse `void=`.

If your data contain `%` but you omit it from `states`, `SequenceData` raises a clear error at construction time.

### Recommended R ↔ Python check (same toy data)

Before trusting downstream EHA results, run the same object in R and Python and compare three quantities:

| Check | R | Python |
|-------|---|--------|
| Row count | `nrow(samm)` | `len(samm.data)` |
| Time range | `range(samm$time)` | `samm.data["time"].min(), samm.data["time"].max()` |
| Spell duration | `summary(samm$spell.time)` | `samm.data["spell.time"].describe()` |

Minimal R script (adjust paths and `seqdef` to match your Python `SequenceData`):

```r
library(TraMineR)
library(TraMineRextras)
# seqdata <- seqdef(...)  # same alphabet / void as Python
samm <- seqsamm(seqdata, sublength = 3)
nrow(samm)
range(samm$time)
summary(samm$spell.time)
```

```python
from sequenzo.with_event_history_analysis import sequence_analysis_multi_state_model

samm = sequence_analysis_multi_state_model(seq, sublength=3)
len(samm.data)
int(samm.data["time"].min()), int(samm.data["time"].max())
samm.data["spell.time"].describe()
```

Mismatches in row count are often due to **void** configuration (Python not dropping `%` windows) or different **censoring** (`time` max should be `L - sublength` for sequence length `L`). Mismatches in `spell.time` usually mean spell-reset logic or void filtering differs—re-check `seqdata.void` and `states`.

Unit tests for loop bound, void filter, and `spell.time` live in `tests/with_event_history_analysis/test_seqsamm_r_parity.py`.

## Basic Workflow

### Step 1: Create a SAMM object from your sequence data

R: `mvad.samm <- seqsamm(mvad.seq, sublength = 6, covar = ...)`

```python
from sequenzo import SequenceData
from sequenzo.with_event_history_analysis import sequence_analysis_multi_state_model

# First, create your SequenceData object
# Assuming you have a DataFrame with sequence columns
seq = SequenceData(
    data=df,
    id_col='person_id',
    time=['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10'],
    states=['employed', 'unemployed', 'education', 'retired'],
    labels=['Employed', 'Unemployed', 'In Education', 'Retired'],
    void='%',  # default; include '%' in states if your data use TraMineR void padding
)

# Create a SAMM object with subsequence length of 3
# This means we look 3 time steps ahead from each position
samm_obj = sequence_analysis_multi_state_model(seq, sublength=3)

# View the person-period data
print(samm_obj.data.head(10))
```

### Step 2: Explore transitions and subsequences

R: `jlseq <- seqsammseq(mvad.samm, "joblessness")`; `plot(mvad.samm)`

```python
from sequenzo.with_event_history_analysis import seqsammseq, plot_samm

# Get all subsequences following unemployment
unemployed_subseq = seqsammseq(samm_obj, spell='unemployed')
print(unemployed_subseq.head())

# Visualize transition patterns (Sequenzo: matplotlib index plots;
# R: plot.SAMM() -> seqplot() with groups by exit state)
plot_samm(samm_obj, title="Transition Patterns by State")
```

### Step 3: Set up typologies for analysis

In R, typology labels are usually passed directly to `seqsammeha()`. Sequenzo also provides `set_typology()` for row-aligned labels, cluster ids, or id / (id, begin) mappings.

```python
from sequenzo.with_event_history_analysis import set_typology

# First, identify transitions from unemployment
unemployed_transitions = (samm_obj.data['s.1'] == 'unemployed') & (samm_obj.data['transition'] == True)

# Get the subsequences for these transitions
transition_data = samm_obj.data[unemployed_transitions]

# Create your typology classification
# For example, classify based on what happens next:
# - If they get employed: "reemployment"
# - If they stay unemployed: "long_term_unemployment"  
# - If they go to education: "education"
typology = []
for _, row in transition_data.iterrows():
    next_state = row['s.2']  # The state they transition to
    if next_state == 'employed':
        typology.append('reemployment')
    elif next_state == 'unemployed':
        typology.append('long_term_unemployment')
    elif next_state == 'education':
        typology.append('education')
    else:
        typology.append('other')

# Apply the typology
samm_obj = set_typology(samm_obj, spell='unemployed', typology=typology)
```

### Step 4: Create event history analysis dataset

R: `jleha <- seqsammeha(mvad.samm, "joblessness", jltype, persper = TRUE)`

```python
from sequenzo.with_event_history_analysis import seqsammeha

# Create person-period dataset for event history analysis
eha_data = seqsammeha(
    samm_obj, 
    spell='unemployed',
    typology=typology,
    persper=True  # Use True for person-period data, False for spell-level data
)

# View the dataset
print(eha_data.head())

# The dataset now has binary outcome variables:
# - SAMMreemployment: 1 if this observation resulted in reemployment
# - SAMMlong_term_unemployment: 1 if stayed unemployed
# - SAMMeducation: 1 if went to education
# - SAMMother: 1 if other outcome

# You can now use this data with statistical models:
# - Logistic regression
# - Cox proportional hazards models
# - Discrete-time event history models
# - etc.
```

## Example: Full Analysis

```python
import pandas as pd
import numpy as np
from sequenzo import SequenceData
from sequenzo.with_event_history_analysis import sequence_analysis_multi_state_model, seqsammeha, plot_samm

# 1. Prepare your data
df = pd.read_csv('employment_sequences.csv')

# 2. Create SequenceData
seq = SequenceData(
    data=df,
    id_col='person_id',
    time=[f'year_{i}' for i in range(2010, 2020)],
    states=['employed', 'unemployed', 'education'],
    labels=['Employed', 'Unemployed', 'Education']
)

# 3. Create SAMM object (R: seqsamm())
samm_obj = sequence_analysis_multi_state_model(seq, sublength=3)

# 4. Visualize patterns (R: plot.SAMM())
plot_samm(samm_obj, title="Employment Transition Patterns")

# 5. Define typologies (example: what happens after unemployment)
unemployed_transitions = (samm_obj.data['s.1'] == 'unemployed') & (samm_obj.data['transition'] == True)
transition_data = samm_obj.data[unemployed_transitions]

# Simple typology: quick vs slow reemployment
typology = []
for _, row in transition_data.iterrows():
    if row['s.2'] == 'employed':
        typology.append('quick_reemployment')
    elif row['s.2'] == 'unemployed' and row['s.3'] == 'employed':
        typology.append('slow_reemployment')
    else:
        typology.append('no_reemployment')

# 6. Create EHA dataset (R: seqsammeha())
eha_data = seqsammeha(samm_obj, spell='unemployed', typology=typology)

# 7. Run statistical analysis (example with sklearn)
from sklearn.linear_model import LogisticRegression

# Prepare features (X) and outcomes (y)
# Add your covariates here (age, education, etc.)
X = eha_data[['spell_time', 'age', 'education_level']]  # spell_time = R spell.time
y = eha_data['SAMMquick_reemployment']  # Example outcome

# Fit model
model = LogisticRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
```

## Key Concepts Explained

### Person-Period Data
- **Traditional format**: One row per person, columns for each time point
- **Person-period format**: One row for each person-time combination
- **Example**: 100 people × 10 time points = 1,000 rows

### Subsequences
- At each time point, we extract the next `sublength` states
- **Example**: If `sublength=3` at time 5, we get states at times 5, 6, 7

### Spells
- A continuous period in the same state
- **Example**: Employed from time 1-5, then unemployed from time 6-8

### Typologies
- Classification of transition outcomes
- **Example**: After unemployment → "quick reemployment", "long-term", "exit labor force"

## Functions Reference

| Python (Sequenzo) | R (TraMineRextras) | Purpose |
|-------------------|--------------------|---------|
| `sequence_analysis_multi_state_model()` / `seqsamm` | `seqsamm()` | Create SAMM person-period dataset |
| `plot_samm()` | `plot.SAMM()` | Visualize transition patterns |
| `seqsammseq()` | `seqsammseq()` | Extract subsequences after a state |
| `set_typology()` | (via `seqsammeha()` in R) | Assign typology classifications |
| `seqsammeha()` | `seqsammeha()` | Generate event history analysis dataset |

## Tips

1. **Choose sublength carefully**: It determines how far ahead you look
   - Too short: Miss important patterns
   - Too long: Lose observations at the end of sequences

2. **Typologies should be meaningful**: Base them on theory or research questions

3. **Person-period vs. Spell-level**:
   - Use person-period (`persper=True`) for duration dependence
   - Use spell-level (`persper=False`) for simpler models

4. **Add covariates**: Use the `covar` parameter in `sequence_analysis_multi_state_model()` for time-invariant variables (R: `covar` in `seqsamm()`)

## For More Information

- **TraMineRextras** SAMM documentation: `?seqsamm` in R; [package manual](https://cran.r-project.org/web/packages/TraMineRextras/refman/TraMineRextras.html)
- **Source**: [seqsamm.R](https://rdrr.io/cran/TraMineRextras/src/R/seqsamm.R)
- **TraMineR** (general sequence tools): https://traminer.unige.ch/
