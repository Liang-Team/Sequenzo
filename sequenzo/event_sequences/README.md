# Event Sequences Module

This module provides a dedicated, easy-to-find entrypoint for event-sequence analysis in Sequenzo.

It is designed to be explicit and TraMineR-friendly:
- focused namespace: `sequenzo.event_sequences`
- clear mapping from TraMineR names to Sequenzo names
- lightweight compatibility helpers (`is_eseq`, `is_seqelist`, `seqelength`, `seqeweight`)

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
    create_event_sequences,
    find_frequent_subsequences,
    compare_groups,
    is_eseq,
    is_seqelist,
    seqelength,
    seqeweight,
)
```

## TraMineR Mapping

- `seqecreate()` -> `create_event_sequences()`
- `seqefsub()` -> `find_frequent_subsequences()`
- `seqeapplysub()` -> `count_subsequence_occurrences()`
- `seqecmpgroup()` -> `compare_groups()`
- `seqe2tse()` -> `convert_event_sequences_to_tse()`
- `seqetm()` -> `compute_event_transition_matrix()`
- `seqecontain()` -> `check_event_subsequence_containment()`
- `seqelist` -> `EventSequenceList`
- `eseq` -> `EventSequence`

## About `is.eseq()` / `is.seqelist()`

TraMineR uses R-style type check helpers:
- `is.eseq()`
- `is.seqelist()`

In Python, idiomatic checking is `isinstance(...)`.
This module supports both styles:

- `isinstance(x, EventSequence)` is equivalent to `is_eseq(x)`
- `isinstance(x, EventSequenceList)` is equivalent to `is_seqelist(x)`

## Length and Weight Helpers

For convenience and TraMineR-style readability:

- `seqelength(obj)`
  - `EventSequence` -> scalar length
  - `EventSequenceList` -> vector of lengths
- `seqeweight(seqelist)` -> vector of sequence weights

## Notes

- This module is the canonical entrypoint for event-sequence analysis in Sequenzo.
- No algorithmic behavior is changed by reorganizing imports.
