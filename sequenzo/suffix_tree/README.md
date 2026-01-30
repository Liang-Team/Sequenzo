# Suffix Tree: Sequence Convergence Analysis

The suffix tree framework analyzes how sequences **converge** over time when read **backward** from the end. It supports two modes: **position-based** (level = time index from end) and **spell-based** (level = spell index from end), aligned with `RLCP` and `RLCPspell` distances.

## Quick Start

```python
from sequenzo import SequenceData, build_suffix_tree
from sequenzo import get_depth_stats, compute_suffix_count, compute_merging_factor
from sequenzo import compute_js_convergence_spell

# Prepare data
seqdata = SequenceData(df, time=time_cols, id_col="id", states=states)

# Position mode (default) — level = time index from end
tree = build_suffix_tree(seqs)
# Or: tree = build_suffix_tree(seqdata, mode="position")

# Spell mode — level = spell index from end (aligned with RLCPspell distance)
tree = build_suffix_tree(seqdata, mode="spell", expcost=0)    # expcost=0: ignore duration
tree = build_suffix_tree(seqdata, mode="spell", expcost=0.5)  # expcost>0: duration-weighted

# Indicators (shared: get_depth_stats, compute_suffix_count work for both tree types)
depth_stats = get_depth_stats(tree)
suffix_counts = compute_suffix_count(tree, max_depth)
merging_factors = compute_merging_factor(tree, max_depth)

# Spell-specific JS convergence (when tree is SpellSuffixTree)
js = compute_js_convergence_spell(
    tree._spell_states, tree._spell_durations, states, tree._expcost
)
```

## Position vs Spell Mode

| Mode      | Level meaning            | Input type            | Aligns with     |
|----------|--------------------------|------------------------|-----------------|
| `position` | Time index from end (last t, last t-1, ...) | `List[List]` or `SequenceData` | RLCP distance   |
| `spell`    | Spell index from end (last spell, last 2 spells, ...) | `SequenceData` only | RLCPspell distance |

- **Position**: Compares sequences from the *last time point backward*. E.g. "given the last 3 years, how many distinct ending patterns exist?"
- **Spell**: Compares sequences spell-by-spell from the end. E.g. "given the last 2 spells, how many distinct suffix patterns exist?" — useful when sequence lengths differ but you care about ending spell patterns.

## API Reference

### Central Hub: `build_suffix_tree`

```python
tree = build_suffix_tree(data, mode="position", expcost=0.0)
```

**Parameters**

- **data**: `List[List]` (sequence of states per time point) or `SequenceData`
- **mode**: `"position"` (default) or `"spell"`
- **expcost**: `float`, default `0.0`. Only used when `mode="spell"`:
  - `0`: Structure and indicators ignore duration (state-only merge).
  - `>0`: Duration influences derived indicators (e.g. JS convergence uses spell-length weighting).

**Returns**

- `SuffixTree` when `mode="position"`
- `SpellSuffixTree` when `mode="spell"`

### Shared Indicators (work for both tree types)

| Function              | Description                                        |
|-----------------------|----------------------------------------------------|
| `get_depth_stats(tree)` | Depth counts and suffix lists per level            |
| `compute_suffix_count(tree, max_depth)` | Number of distinct suffixes per level           |
| `compute_merging_factor(tree, max_depth)` | Mean merging at each level               |

### Spell-Specific Indicators

When the tree is a `SpellSuffixTree`, use:

```python
# Jensen-Shannon convergence between consecutive spell-level distributions
js_scores = compute_js_convergence_spell(
    spell_states,      # tree._spell_states
    spell_durations,   # tree._spell_durations
    state_set,         # list of state labels
    expcost            # tree._expcost
)
```

## Individual-Level Indicators

Individual-level indicators measure how typical or unique each sequence’s *ending* is (convergence: low rarity = typical).

### Position-based (time index from end): `IndividualConvergence`

Use when you have a list of sequences (same length) or a position-based suffix tree. Level = time index from end; suffix = states from time t to end.

```python
from sequenzo import IndividualConvergence, build_suffix_tree

tree = build_suffix_tree(seqs, mode="position")
sequences = tree.sequences  # or your list of lists
ind = IndividualConvergence(sequences)

# Per-year suffix rarity (each individual × each time point from end)
rarity_df = ind.compute_suffix_rarity_per_year(zscore=False)
scores = ind.compute_suffix_rarity_score()
std_scores = ind.compute_standardized_rarity_score(min_t=1, window=1)

# Binary convergence (0/1) and first convergence year
converged = ind.compute_converged(method="zscore", z_threshold=1.5, min_t=1)
first_year = ind.compute_first_convergence_year(method="zscore", z_threshold=1.5, min_t=1)

# Path uniqueness (how many time steps from end have a unique suffix)
uniqueness = ind.compute_path_uniqueness()
```

Methods for `compute_converged` and `compute_first_convergence_year`: `"zscore"` (window of *low* rarity z-scores, i.e. z < -z_threshold), `"top_proportion"` (bottom p% most typical), `"quantile"` (below a quantile threshold). See docstrings in `individual_level_indicators.py`.

### Spell-based: `SpellIndividualConvergence`

Use when the unit of analysis is the spell from the end. Level = spell index from end (last spell, last two spells, ...). Requires a `SpellSuffixTree` built with `build_spell_suffix_tree(seqdata)`.

```python
from sequenzo import build_spell_suffix_tree, SpellIndividualConvergence

tree = build_spell_suffix_tree(seqdata, expcost=0)
ind = SpellIndividualConvergence(tree)

# Per-spell suffix rarity (from end)
rarity_df = ind.compute_suffix_rarity_per_spell(zscore=False)
scores = ind.compute_suffix_rarity_score()
std_scores = ind.compute_standardized_rarity_score(min_k=1, window=1)

# Binary convergence and first convergence spell level (from end)
converged = ind.compute_converged(method="zscore", z_threshold=1.5, min_k=1)
first_spell = ind.compute_first_convergence_spell(method="zscore", z_threshold=1.5, min_k=1)

# Path uniqueness (how many spell levels from end have a unique suffix)
uniqueness = ind.compute_path_uniqueness()
```

Variable-length sequences are supported.

### Plotting

- `plot_suffix_rarity_distribution(data, ...)`: distribution of suffix rarity (or standardized) scores, optionally by group, with threshold line.
- `plot_individual_indicators_correlation(df, ...)`: correlation heatmap of individual-level indicators (e.g. converged, suffix_rarity_score, path_uniqueness).

## Helper: `convert_seqdata_to_spells`

Spell representation is shared with prefix tree:

```python
from sequenzo.prefix_tree import convert_seqdata_to_spells

spell_states, spell_durations, state_list = convert_seqdata_to_spells(seqdata)
```

## Design Notes

- Spell mode uses the same DSS + duration pipeline as `RLCPspell` and `OMspell` in `get_distance_matrix`.
- `expcost=0` in spell mode corresponds to "state-only" comparison; `expcost>0` adds duration awareness similar to `RLCPspell` with `expcost>0`.
