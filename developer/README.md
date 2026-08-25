# Sequenzo developer resources

This directory contains build diagnostics and implementation notes for contributors.
User-facing installation and usage instructions belong in the project `README.md` and
`Tutorials/`.

## Build and OpenMP diagnostics

- `test_openmp.py` checks macOS and Linux extension modules from the current source
  checkout in separate processes.
- `ARCHITECTURE_GUIDE.md` documents macOS architecture selection during local builds.
- `WINDOWS_OPENMP_GUIDE.md` documents Windows-specific OpenMP checks.

The current implementation is defined by:

- `setup.py` for compile and link configuration;
- `sequenzo/openmp_setup.py` for runtime-library selection;
- `tests/openmp/test_openmp_setup.py` for regression coverage.

Run the diagnostic on macOS or Linux with:

```bash
python developer/test_openmp.py
```

## Distance and clustering notes

- `NORM_GUIDE.md` describes normalization options for sequence distances.
- `SEQUENZO_VS_TRAMINER_COMPARISON.md` records implementation comparisons with TraMineR.
- `CLUSTERING_WEIGHTEDCLUSTER_VS_SEQUENZO.md` records clustering API comparisons.
- `SEQUENZO_HELSKE_IMPLEMENTATION_REQUIREMENTS.md` records requirements derived from
  the sequences-to-variables workflow.
