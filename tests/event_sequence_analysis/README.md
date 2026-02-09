# Event sequence analysis tests (LSOG)

Tests for `sequenzo.with_event_history_analysis` event sequence API, using the **dyadic_children (LSOG)** dataset. Results are compared to **TraMineR** when reference files are present.

## Why 3 tests are skipped

These three tests are **skipped** when TraMineR reference files are missing:

- `test_event_sequences_match_traminer_meta`
- `test_event_sequences_match_traminer_fsub`
- `test_event_sequences_match_traminer_applysub`

They compare Sequenzo output to TraMineR (`seqecreate`, `seqefsub`, `seqeapplysub`). The reference files are **not** shipped in the repo; you must generate them with R.

## Generating TraMineR reference files

From the **repository root**, with R and TraMineR installed:

```bash
Rscript tests/event_sequence_analysis/traminer_reference_event_sequence.R \
  sequenzo/datasets/dyadic_children.csv \
  20 \
  tests/event_sequence_analysis
```

This writes into `tests/event_sequence_analysis/`:

- `ref_eseq_meta.csv` — number of sequences and event alphabet size
- `ref_eseq_alphabet.csv` — event alphabet order (required for fsub/applysub comparison)
- `ref_eseq_fsub_support.csv` — Support and Count for frequent subsequences
- `ref_eseq_applysub.csv` — presence matrix (seqeapplysub, method="presence")

After that, run:

```bash
pytest tests/event_sequence_analysis/test_event_sequence_lsog.py -v
```

All 11 tests (including the 3 TraMineR comparison tests) will run, and results must match TraMineR.

## Parameters (aligned with TraMineR)

The R script and Sequenzo use the same semantics:

- **seqecreate**: `tevent = "transition"` (one event per transition; first state at time 0, transitions at 1, 2, …).
- **seqefsub**: `min.support = 2`, default `seqeconstraint()` (count.method = 1, i.e. COBJ / per-sequence presence).
- **seqeapplysub**: `method = "presence"` (0/1 per sequence).

See TraMineR source: `seqecreate.R`, `seqformat-STS_to_TSE.R`, `seqetm.R`, `seqefsub.R`, `seqeapplysub.R`, `seqeconstraint.R`.

## Requirements

- **R** with **TraMineR**: `install.packages("TraMineR")`
- Dataset path: `sequenzo/datasets/dyadic_children.csv` (same data as `load_dataset("dyadic_children")`)
