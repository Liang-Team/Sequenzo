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
- `sequence_analysis_multi_state_model`
- `seqsammseq`
- `seqsammeha`
- `seqsha`
- `person_level_to_person_period`

### This module does NOT include
- `create_event_sequences` / `seqecreate`
- `find_frequent_subsequences` / `seqefsub`
- `compare_groups` / `seqecmpgroup`
- `count_subsequence_occurrences` / `seqeapplysub`
- event-sequence plotting helpers

For these, use `sequenzo.event_sequences`.

## Why This Split Helps

Separating event-sequence and event-history concerns makes API discovery simpler:
- users do not need to guess where to import from
- docs map cleanly to conceptual tasks
- each module has one primary purpose

## References

- Ritschard, G., Bürgin, R., and Studer, M. (2014). "Exploratory Mining of Life Event Histories", In McArdle, J.J. & Ritschard, G. (eds) *Contemporary Issues in Exploratory Data Mining in the Behavioral Sciences*. Series: Quantitative Methodology, pp. 221-253. New York: Routledge.

- TraMineR Documentation: http://traminer.unige.ch
