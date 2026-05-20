# EMLT tests (`seqemlt`)

Parity tests for `sequenzo.emlt.compute_emlt` against TraMineRextras `seqemlt()`.

## Generate TraMineR reference files

From the repository root, with R, TraMineR, and TraMineRextras installed:

```bash
Rscript tests/emlt/traminer_reference_seqemlt.R
```

This writes `ref_seqemlt_*.csv` into `tests/emlt/`.

## Run tests

```bash
pytest tests/emlt/test_seqemlt_traminer_consistency.py -v
```

Reference comparisons use positional alignment (CSV row names such as `1.10` are not reliable when read by pandas).
