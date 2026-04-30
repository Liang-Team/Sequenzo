# Sequence Operations

Utilities aligned with TraMineR sequence operation helpers:

- `seqconc` - concatenate row-wise sequence values into strings
- `seqdecomp` - decompose concatenated sequence strings back to columns
- `seqsep` - split fixed-width sequence strings with separators
- `seqrecode` - recode sequence states using user-defined mapping
- `seqshift` - shift a sequence with NA padding
- `seqasnum` - convert `SequenceData` to TraMineR-style numeric matrix (0-based)

## Notes on Compatibility

- The implementation follows TraMineR function intent and default behavior.
- For `SequenceData`, `seqrecode` returns a new `SequenceData` object.
- `seqasnum(with_missing=False)` keeps missing states as `NaN`, like TraMineR.

## Example

```python
from sequenzo.sequence_operations import seqconc, seqdecomp, seqasnum

conc = seqconc(sequence_data, sep="-")
decomp = seqdecomp(conc, sep="-")
num = seqasnum(sequence_data, with_missing=False)
```
