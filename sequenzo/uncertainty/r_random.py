"""
@Author  : Yuqi Liang 梁彧祺
@File    : r_random.py
@Time    : 17/05/2026 22:30
@Desc    : R-compatible random stream for parity tests.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _r_eval(script: str) -> str:
    out = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


class RRandomStream:
    """
    Advance R's RNG exactly as in an interactive R session (``set.seed`` + ``runif`` / ``sample``).
    """

    def __init__(self, seed: int):
        self._seed = int(seed)
        self._tmpdir = tempfile.mkdtemp(prefix="sequenzo_mc_rng_")
        self._state_path = Path(self._tmpdir) / "Random.seed.rds"
        _r_eval(f'set.seed({self._seed}); saveRDS(.Random.seed, "{self._state_path}")')

    def random(self, n: int = 1) -> np.ndarray:
        n = int(n)
        p = str(self._state_path)
        script = (
            f'.Random.seed <- readRDS("{p}"); '
            f"u <- runif({n}); "
            f'saveRDS(.Random.seed, "{p}"); '
            f"cat(paste(u, collapse=' '))"
        )
        raw = _r_eval(script)
        if not raw:
            return np.zeros(n, dtype=float)
        return np.array([float(x) for x in raw.split()], dtype=float)

    def sample_int(self, n_pop: int, k: int) -> np.ndarray:
        p = str(self._state_path)
        script = (
            f'.Random.seed <- readRDS("{p}"); '
            f"s <- sample({int(n_pop)}, {int(k)}); "
            f'saveRDS(.Random.seed, "{p}"); '
            f"cat(paste(s, collapse=' '))"
        )
        raw = _r_eval(script)
        return np.array([int(x) for x in raw.split()], dtype=int)


def r_random_state(seed: int) -> RRandomStream:
    return RRandomStream(seed)
