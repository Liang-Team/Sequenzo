# Statistics Module

This module is the user-friendly entry point for descriptive sequence statistics.

## Quick Rule (Beginner Friendly)

- Use `sequenzo.statistics` when you want summary numbers and tables.
- Use `sequenzo.sequence_characteristics` when you want theory-driven indicators (entropy, turbulence, complexity, etc.).

## What Is Included Here

### Weighted Statistics

- `get_weighted_mean`
- `get_weighted_variance`
- `get_weighted_five_number_summary`

### Sequence Summary Statistics

- `get_distinct_state_sequences` (TraMineR `seqdss`)
- `get_state_spell_durations` (TraMineR `seqdur`)
- `get_individual_state_distribution` (TraMineR `seqistatd`)
- `get_mean_time_by_state` (TraMineR `seqmeant`)
- `get_modal_state_sequence` (TraMineR `seqmodst`)
- `get_sequence_length_summary`
- `get_transition_count_summary`

## Why This Is Separate

`statistics` focuses on descriptive summaries for quick understanding and reporting.
`sequence_characteristics` focuses on methodological indicators used for deeper analytical modeling.
