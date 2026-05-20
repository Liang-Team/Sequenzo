# EMLT module

TraMineRextras mapping: `seqemlt()` → `compute_emlt()` / `seqemlt()`.

## Usage

```python
from sequenzo import SequenceData, compute_emlt

result = compute_emlt(seqdata, a=1, b=1, weighted=True)
# result.sit_cor   — correlations between time-stamped situations
# result.coord     — sequence coordinates on PCA axes (for clustering)
# result.pca       — PCA scores, loadings, sdev (R princomp, cor=TRUE)
```

## Pipeline (same steps as R)

1. Situation frequencies (`sit.freq`)
2. Disjunctive coding (`a`)
3. Transition rates (`sit.transrate`)
4. Time-discounted profiles (`sit.profil`)
5. Profile distances (`c`)
6. Benzécri covariance (`d`)
7. PCA on active situations (`pca`)
8. Sequence coordinates (`coord`)
9. Situation correlations (`sit.cor`)
