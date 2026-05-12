"""
@Author  : Yuqi Liang 梁彧祺
@File    : weightedcluster_compat.py
@Time    : 11/05/2025 16:31
@Desc    : 
Compatibility helpers for matching WeightedCluster (R) behaviour.

These utilities are used when strict numerical alignment with the R package
is required, for example ``sample.int`` and ``diana`` / ``agnes`` trees.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree

_RSCRIPT = shutil.which("Rscript")


class RSampleIntStream:
    """Draw ``sample.int`` values from R in the same order as WeightedCluster."""

    def __init__(self, seed: int, n: int, kvals: Sequence[int]) -> None:
        if _RSCRIPT is None:
            raise RuntimeError(
                "Rscript is required for WeightedCluster-compatible random "
                "medoid initialisation. Install R or pass explicit initialclust."
            )
        self._seed = int(seed)
        self._n = int(n)
        self._kvals = [int(k) for k in kvals]
        self._draws = self._draw_all()

    def _draw_all(self) -> dict[int, np.ndarray]:
        k_text = ", ".join(str(k) for k in self._kvals)
        script = f"""
args <- commandArgs(trailingOnly = TRUE)
seed <- as.integer(args[1])
n <- as.integer(args[2])
kvals <- c({k_text})
set.seed(seed)
draws <- lapply(kvals, function(k) as.integer(sample.int(n, k) - 1))
cat(paste(unlist(lapply(draws, function(x) paste(x, collapse = ","))), collapse = ";"))
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "sample_int_sequence.R"
            script_path.write_text(script, encoding="utf-8")
            completed = subprocess.run(
                [_RSCRIPT, str(script_path), str(self._seed), str(self._n)],
                check=True,
                capture_output=True,
                text=True,
            )
        draws: dict[int, np.ndarray] = {}
        for k, chunk in zip(self._kvals, completed.stdout.strip().split(";")):
            values = np.asarray([int(part) for part in chunk.split(",") if part], dtype=int)
            if values.shape[0] != k:
                raise RuntimeError("R sample.int returned an unexpected number of indices.")
            draws[k] = values
        return draws

    def sample(self, k: int) -> np.ndarray:
        """Return 0-based medoid indices for one ``k``."""
        return self._draws[int(k)].copy()


def _require_rscript() -> str:
    if _RSCRIPT is None:
        raise RuntimeError(
            "Rscript is required for this WeightedCluster method. "
            "Install R or choose a supported hierarchical method."
        )
    return _RSCRIPT


def divisive_hclust_linkage(diss: np.ndarray, method: str) -> np.ndarray:
    """
    Build a linkage matrix with R ``diana`` or ``agnes`` (beta.flexible).

    Parameters
    ----------
    diss
        Square symmetric distance matrix.
    method
        ``"diana"`` or ``"beta.flexible"``.
    """
    diss = np.asarray(diss, dtype=np.float64, order="C")
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square distance matrix.")

    method = method.lower()
    if method not in {"diana", "beta.flexible"}:
        raise ValueError("method must be 'diana' or 'beta.flexible'.")

    rscript = _require_rscript()
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        diss_path = root / "diss.csv"
        out_path = root / "linkage.csv"
        np.savetxt(diss_path, diss, delimiter=",")
        if method == "diana":
            r_body = "hc <- cluster::diana(diss, diss = TRUE)"
        else:
            r_body = (
                "hc <- cluster::agnes(diss, diss = TRUE, method = 'flexible', "
                "par.method = 0.625)"
            )
        script = f"""
suppressPackageStartupMessages({{ library(cluster) }})
diss <- as.matrix(read.csv("{diss_path}", header = FALSE, check.names = FALSE))
{r_body}
merge <- hc$merge
height <- hc$height
out <- data.frame(left = merge[, 1], right = merge[, 2], height = height)
write.csv(out, "{out_path}", row.names = FALSE)
"""
        script_path = root / "twins_linkage.R"
        script_path.write_text(script, encoding="utf-8")
        subprocess.run([rscript, str(script_path)], check=True, capture_output=True, text=True)

        table = np.loadtxt(out_path, delimiter=",", skiprows=1)
        n = diss.shape[0]
        linkage = np.zeros((n - 1, 4), dtype=np.float64)
        for row_idx in range(n - 1):
            left = int(table[row_idx, 0])
            right = int(table[row_idx, 1])
            linkage[row_idx, 0] = -left - 1 if left < 0 else left + n - 1
            linkage[row_idx, 1] = -right - 1 if right < 0 else right + n - 1
            linkage[row_idx, 2] = table[row_idx, 2]
            linkage[row_idx, 3] = 0.0
    return linkage


def cutree_labels(linkage_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Return 1-based cluster labels from a linkage matrix."""
    return cut_tree(linkage_matrix, n_clusters=n_clusters).ravel() + 1


def r_k_medoids_range(
    diss: np.ndarray,
    kvals: Sequence[int],
    weights: Optional[np.ndarray],
    *,
    method: str = "PAMonce",
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run WeightedCluster ``wcKMedRange`` and return clustering and stats tables."""
    diss = np.asarray(diss, dtype=np.float64, order="C")
    if diss.ndim != 2 or diss.shape[0] != diss.shape[1]:
        raise ValueError("diss must be a square distance matrix.")

    rscript = _require_rscript()
    k_text = ", ".join(str(int(k)) for k in kvals)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        diss_path = root / "diss.csv"
        weights_path = root / "weights.csv"
        clustering_path = root / "clustering.csv"
        stats_path = root / "stats.csv"
        np.savetxt(diss_path, diss, delimiter=",")
        if weights is None:
            weights = np.ones(diss.shape[0], dtype=np.float64)
        np.savetxt(weights_path, np.asarray(weights, dtype=np.float64), delimiter=",")
        script = f"""
suppressPackageStartupMessages(library(WeightedCluster))
diss <- as.matrix(read.csv("{diss_path}", header = FALSE, check.names = FALSE))
weights <- as.numeric(read.csv("{weights_path}", header = FALSE, check.names = FALSE)[[1]])
set.seed({int(seed)})
result <- wcKMedRange(diss, kvals = c({k_text}), weights = weights, method = "{method}")
write.csv(as.data.frame(result$clustering), "{clustering_path}", row.names = FALSE)
write.csv(as.data.frame(result$stats), "{stats_path}")
"""
        script_path = root / "k_medoids_range.R"
        script_path.write_text(script, encoding="utf-8")
        subprocess.run([rscript, str(script_path)], check=True, capture_output=True, text=True)

        clustering = pd.read_csv(clustering_path)
        stats = pd.read_csv(stats_path, index_col=0)
    return clustering, stats
