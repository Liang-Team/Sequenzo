# Event Sequences Module

This module provides a dedicated, easy-to-find entrypoint for event-sequence analysis in Sequenzo.

It is designed to be explicit and TraMineR-friendly:
- focused namespace: `sequenzo.event_sequences`
- clear mapping from TraMineR names to Sequenzo names
- lightweight helper utilities (`is_event_sequence`, `is_event_sequence_collection`, `get_event_sequence_lengths`, `get_event_sequence_weights`)

## Why This Module Exists

Event-sequence features were previously implemented under broader analysis namespaces.
That worked technically, but users could miss the capability or confuse it with SAMM/event-history workflows.

`sequenzo.event_sequences` makes the feature discoverable and self-contained.

## Module Boundary (Important)

Use `sequenzo.event_sequences` for:
- Event-sequence objects and constraints
- Frequent subsequence mining and counting
- Group comparison on event subsequences
- Event-sequence visualization

Use `sequenzo.with_event_history_analysis` for:
- SAMM and sequence-history modeling workflows
- Event-history analysis utilities (`seqsha`, person-period tools)

If your task is "patterns of events", use `event_sequences`.
If your task is "history/modeling of transitions", use `with_event_history_analysis`.

## Quick Start

```python
from sequenzo.event_sequences import (
    EventSequenceData,
    find_frequent_subsequences,
    compare_groups,
    is_event_sequence,
    is_event_sequence_collection,
    get_event_sequence_lengths,
    get_event_sequence_weights,
)
```

## TraMineR Mapping

- `seqecreate()` -> `EventSequenceData.from_tse()` / `EventSequenceData.from_state_sequences()`
- `seqefsub()` -> `find_frequent_subsequences()`
- `seqeapplysub()` -> `count_subsequence_occurrences()`
- `seqecmpgroup()` -> `compare_groups()`
- `seqe2tse()` -> `convert_event_sequences_to_tse()`
- `seqetm()` -> `compute_event_transition_matrix()`
- `seqecontain()` -> `check_event_subsequence_containment()`
- `seqelist` -> `EventSequenceList`
- `eseq` -> `EventSequence`

## Recommended Constructor Pattern

- Preferred object-oriented API:
  - `EventSequenceData.from_tse(...)`
  - `EventSequenceData.from_state_sequences(seqdata, ...)`

## Visualization API (Unified Common Parameters)

Event-sequence plotting functions now share a common set of practical plotting arguments:

- `save_as`: output file path (auto-appends `.png` if extension is missing)
- `dpi`: save resolution (default `200`)
- `show`: whether to call `plt.show()` inside the function (default `False`)
- `x_label` / `y_label`: axis label overrides (where applicable)

Supported plotting functions:

- `plot_event_parallel_coordinates(...)`
- `plot_subsequence_frequencies(...)`
- `plot_subsequence_group_contrasts(...)`
- `plot_event_dynamics(...)`

Example:

```python
from sequenzo.event_sequences import (
    plot_event_parallel_coordinates,
    plot_subsequence_frequencies,
    plot_subsequence_group_contrasts,
    plot_event_dynamics,
)

# Parallel coordinates
fig = plot_event_parallel_coordinates(
    eseq,
    group_labels=group,
    x_label="Position",
    y_label="Event",
    save_as="outputs/event_parallel_by_group",
    dpi=300,
    show=True,
)

# Frequent subsequences
fig = plot_subsequence_frequencies(
    fsub[:12],
    x_label="Support",
    y_label="Subsequence",
    save_as="outputs/subsequence_support_top12",
    dpi=300,
    show=True,
)

# Group contrasts
fig = plot_subsequence_group_contrasts(
    discr[:10],
    plot_type="resid",
    x_label="Pearson residual",
    y_label="Subsequence",
    save_as="outputs/subsequence_group_contrasts_resid",
    dpi=300,
    show=True,
)

# Event dynamics (survival/hazard)
fig = plot_event_dynamics(
    eseq,
    group_labels=group,
    curve_type="hazard",
    x_label="Time",
    y_label="Mean number of events",
    save_as="outputs/event_dynamics_hazard",
    dpi=300,
    show=True,
)
```

## About Type Checks

In Python, idiomatic checking is `isinstance(...)`.
This module provides explicit helper names:

- `is_event_sequence(x)`
- `is_event_sequence_collection(x)`

## Length and Weight Helpers

For convenience and TraMineR-style readability:

- `get_event_sequence_lengths(obj)`
  - `EventSequence` -> scalar length
  - `EventSequenceList` -> vector of lengths
- `get_event_sequence_weights(seqelist)` -> vector of sequence weights

## Notes

- This module is the canonical entrypoint for event-sequence analysis in Sequenzo.
- No algorithmic behavior is changed by reorganizing imports.
