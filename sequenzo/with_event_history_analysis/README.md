# Event History Analysis in Sequenzo

## Overview

This module is dedicated to **event history analysis methods** (for example, SAMM and sequence history workflows).

Event-sequence mining is intentionally separated and now lives in:
- `sequenzo.event_sequences`

This split avoids confusion between:
- **event-sequence pattern analysis** (what event patterns occur), and
- **event-history modeling** (how transitions/history are modeled).

## Scope Boundary

### This module includes
- `SAMM`
- `sequence_analysis_multi_state_model` (`seqsamm`; TraMineRextras)
- `seqsammseq`
- `seqsammeha`
- `get_sequence_history_data`
- `person_level_to_person_period`

**SAMM / void:** TraMineR `seqsamm()` drops subsequence windows that contain **void**
(out-of-window padding; default symbol `"%"`). Use `SequenceData(..., void="%")` (default)
and list the void symbol in `states` when it appears in the data; `seqsamm` reads
`seqdata.void_code`. Pass `void=None` only if you have no void padding. Compare row count,
`time` range, and `spell.time` with R on the same toy data — see `samm_examples.md`.

### This module does NOT include
- `create_event_sequences` / `seqecreate`
- `find_frequent_subsequences` / `seqefsub`
- `compare_groups` / `seqecmpgroup`
- `count_subsequence_occurrences` / `seqeapplysub`
- event-sequence plotting helpers

For these, use `sequenzo.event_sequences`.

## Void design (SAMM / `SequenceData`)

TraMineR separates **void** (calendar padding outside the observation window, default `%`)
from **missing** (unknown state inside the window). This matters mainly for
**event history analysis**: `seqsamm()` removes person-period rows whose subsequence
contains any void, because those windows are not valid “next *k* states” for modelling.

| | Void | Missing |
|---|------|---------|
| Meaning | Not in the observation window yet / anymore | In window, value unknown |
| Sequenzo | `SequenceData(void="%")`, symbol in `states` → `void_code` | `missing_values`, auto `NaN` → Missing state |
| SAMM (`seqsamm`) | Drops row if subsequence contains `void_code` | Does not drop by default |

Typical state-sequence workflows (clustering, index plots) may never show `%` in the data;
you can ignore void until you call `sequence_analysis_multi_state_model()`. For R parity,
use the same void symbol and alphabet as `seqdef()`, then compare `nrow`, `time`, and
`spell.time` (see `samm_examples.md`).

## Why This Split Helps

Separating event-sequence and event-history concerns makes API discovery simpler:
- users do not need to guess where to import from
- docs map cleanly to conceptual tasks
- each module has one primary purpose

## References

- Ritschard, G., Bürgin, R., and Studer, M. (2014). "Exploratory Mining of Life Event Histories", In McArdle, J.J. & Ritschard, G. (eds) *Contemporary Issues in Exploratory Data Mining in the Behavioral Sciences*. Series: Quantitative Methodology, pp. 221-253. New York: Routledge.

- TraMineR Documentation: http://traminer.unige.ch
