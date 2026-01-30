# Prefix Tree: Sequence Divergence Analysis

The prefix tree framework analyzes how sequences **diverge** over time when read **forward** from the start. It supports two modes: **position-based** (level = time index) and **spell-based** (level = spell index), aligned with `LCP` and `LCPspell` distances.

## Quick Start

```python
from sequenzo import SequenceData, build_prefix_tree
from sequenzo import get_depth_stats, compute_prefix_count, compute_js_divergence_spell

# Prepare data
seqdata = SequenceData(df, time=time_cols, id_col="id", states=states)

# Position mode (default) — level = time index
tree = build_prefix_tree(seqs)
# Or: tree = build_prefix_tree(seqdata, mode="position")

# Spell mode — level = spell index (aligned with LCPspell distance)
tree = build_prefix_tree(seqdata, mode="spell", expcost=0)    # expcost=0: ignore duration
tree = build_prefix_tree(seqdata, mode="spell", expcost=0.5)  # expcost>0: duration-weighted

# Indicators (shared: get_depth_stats, compute_prefix_count work for both tree types)
depth_stats = get_depth_stats(tree)
prefix_counts = compute_prefix_count(tree, max_depth)

# Spell-specific JS divergence (when tree is SpellPrefixTree)
js = compute_js_divergence_spell(
    tree._spell_states, tree._spell_durations, states, tree._expcost
)
```

## Position vs Spell Mode

| Mode      | Level meaning        | Input type            | Aligns with     |
|----------|----------------------|------------------------|-----------------|
| `position` | Time index (t=1,2,...) | `List[List]` or `SequenceData` | LCP distance    |
| `spell`    | Spell index (1st spell, 2nd spell, ...) | `SequenceData` only | LCPspell distance |

- **Position**: Compares sequences at the *same time point*. E.g. "at year 2010, what fraction of sequences share this prefix?"
- **Spell**: Compares sequences spell-by-spell. A "spell" is a maximal run of consecutive same states. E.g. "after 2 spells, how many distinct state sequences exist?" — sequences with different numbers of time points but the same spell pattern can be compared.

## API Reference

### Central Hub: `build_prefix_tree`

```python
tree = build_prefix_tree(data, mode="position", expcost=0.0)
```

**Parameters**

- **data**: `List[List]` (sequence of states per time point) or `SequenceData`
- **mode**: `"position"` (default) or `"spell"`
- **expcost**: `float`, default `0.0`. Only used when `mode="spell"`:
  - `0`: Structure and indicators ignore duration (state-only merge).
  - `>0`: Duration influences derived indicators (e.g. JS divergence uses spell-length weighting). Larger `expcost` gives more weight to longer spells.

**Returns**

- `PrefixTree` when `mode="position"`
- `SpellPrefixTree` when `mode="spell"`

### Shared Indicators (work for both tree types)

| Function              | Description                                        |
|-----------------------|----------------------------------------------------|
| `get_depth_stats(tree)` | Depth counts and prefix lists per level            |
| `compute_prefix_count(tree, max_depth)` | Number of distinct prefixes per level           |
| `compute_branching_factor(tree, max_depth)` | Mean branching at each level              |

### Spell-Specific Indicators

When the tree is a `SpellPrefixTree`, use:

```python
# Jensen-Shannon divergence between consecutive spell-level distributions
js_scores = compute_js_divergence_spell(
    spell_states,      # tree._spell_states
    spell_durations,   # tree._spell_durations
    state_set,         # list of state labels
    expcost            # tree._expcost
)
```

## Helper: `convert_seqdata_to_spells`

Extract spell representation from `SequenceData`:

```python
from sequenzo.prefix_tree import convert_seqdata_to_spells

spell_states, spell_durations, state_list = convert_seqdata_to_spells(seqdata)
# spell_states[i] = [s1, s2, ...] state labels for spells of sequence i
# spell_durations[i] = [d1, d2, ...] duration (time points) per spell
```

## Design Notes

- Spell mode uses the same DSS (Distinct SubSequence) + duration pipeline as `LCPspell` and `OMspell` in `get_distance_matrix`.
- `expcost=0` in spell mode corresponds to "state-only" comparison; `expcost>0` adds duration awareness similar to `LCPspell` with `expcost>0`.
