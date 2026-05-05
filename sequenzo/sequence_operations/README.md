# Sequence Operations

Utilities aligned with TraMineR sequence operation helpers:

- `concatenate_sequences` - equivalent to TraMineR `seqconc()`
- `decompose_concatenated_sequences` - equivalent to TraMineR `seqdecomp()`
- `split_fixed_width_sequences` - equivalent to TraMineR `seqsep()`
- `recode_sequence_states` - equivalent to TraMineR `seqrecode()`
- `shift_sequence_with_missing_padding` - equivalent to TraMineR `seqshift()`
- `convert_sequences_to_numeric_matrix` - equivalent to TraMineR `seqasnum()`
- `pairwise_sequence_alignment` - equivalent to TraMineR `seqalign()`
- `find_sequence_occurrences` - equivalent to TraMineR `seqfind()`
- `longest_common_prefix_length` - equivalent to TraMineR `seqLLCP()`
- `longest_common_subsequence_length` - equivalent to TraMineR `seqLLCS()`

## Notes on Compatibility

- The implementation follows TraMineR function intent and default behavior.
- For `SequenceData`, `recode_sequence_states` returns a new `SequenceData` object.
- `convert_sequences_to_numeric_matrix(with_missing=False)` keeps missing states as `NaN`, like TraMineR.
- `find_sequence_occurrences` returns 1-based positions to match TraMineR `which(...)` semantics.
- `pairwise_sequence_alignment` uses TraMineR tie-breaking (substitution/match before insertion/deletion when costs are equal).

## Example

```python
from sequenzo.sequence_operations import (
    concatenate_sequences,
    decompose_concatenated_sequences,
    convert_sequences_to_numeric_matrix,
    find_sequence_occurrences,
    longest_common_prefix_length,
    longest_common_subsequence_length,
    pairwise_sequence_alignment,
)

conc = concatenate_sequences(sequence_data, sep="-")
decomp = decompose_concatenated_sequences(conc, sep="-")
num = convert_sequences_to_numeric_matrix(sequence_data, with_missing=False)
occ = find_sequence_occurrences(query_seqdata, reference_seqdata)
lcp = longest_common_prefix_length(sequence_data, sequence_data, index1=0, index2=1)
lcs = longest_common_subsequence_length(sequence_data, sequence_data, index1=0, index2=1)
align = pairwise_sequence_alignment(sequence_data, indices=[0, 1], indel=1.0, sm=substitution_matrix)
```
