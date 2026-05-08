# Feature Extraction and Selection

This module exposes two complementary entrypoints that serve different goals:

- `extract_sequence_features()`: **feature extraction only**.
  - Returns duration/timing/sequencing feature matrices.
  - Does **not** run Boruta.
  - Does **not** fit a final predictive/explanatory model.

- `run_feature_extraction_and_selection_pipeline()`: **full workflow**.
  - Extracts features.
  - Runs feature selection (Boruta).
  - Fits a final model (regression/classification).
  - Supports residualization with controls.

## Reproducible Preset

For paper-style reproducibility, you can use:

- `preset="unterlerchner2023"`

This preset locks key defaults (e.g., 12-month timing bins, `max_k=3`,
`min_support=0.05`, Boruta setup, and residualization behavior).

Example:

```python
from sequenzo.feature_extraction_and_selection import (
    run_feature_extraction_and_selection_pipeline,
)

result = run_feature_extraction_and_selection_pipeline(
    seqdata=seqdata,
    outcome=outcome,
    controls=controls,
    preset="unterlerchner2023",
)
```
