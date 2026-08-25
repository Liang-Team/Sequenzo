"""PAMonce clustering for state sequences."""

from __future__ import annotations

import numpy as np

import sequenzo.clustering.clustering_c_code as clustering_c_code
from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures import get_distance_matrix


_CODEBOOK_CHUNK_SIZE = 131_072
_CODEBOOK_MAX_VALUES = np.iinfo(np.uint16).max + 1


def _hamming_counts_condensed(sequences: np.ndarray, dtype) -> np.ndarray:
    n = sequences.shape[0]
    result = np.empty(n * (n - 1) // 2, dtype=dtype)
    offset = 0
    for left in range(n - 1):
        width = n - left - 1
        result[offset : offset + width] = np.count_nonzero(
            sequences[left + 1 :] != sequences[left], axis=1
        )
        offset += width
    return result


def _integer_hamming_condensed(sequences: np.ndarray) -> np.ndarray:
    length = int(sequences.shape[1])
    if length <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif length <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    else:
        dtype = np.uint32

    unique_sequences, inverse = np.unique(
        sequences, axis=0, return_inverse=True
    )
    if unique_sequences.shape[0] == sequences.shape[0]:
        return _hamming_counts_condensed(sequences, dtype)

    unique_distance = _hamming_counts_condensed(unique_sequences, dtype)
    unique_n = unique_sequences.shape[0]
    n = sequences.shape[0]
    result = np.empty(n * (n - 1) // 2, dtype=dtype)
    offset = 0
    for left in range(n - 1):
        right_ids = inverse[left + 1 :]
        width = right_ids.size
        left_id = inverse[left]
        same = right_ids == left_id
        row = result[offset : offset + width]
        row[same] = 0
        if np.any(~same):
            lower = np.minimum(left_id, right_ids[~same])
            upper = np.maximum(left_id, right_ids[~same])
            source = (
                lower * (2 * unique_n - lower - 1) // 2
                + upper
                - lower
                - 1
            )
            row[~same] = unique_distance[source]
        offset += width
    return result


def _can_use_integer_hamming(method: str, kwargs: dict) -> bool:
    if method.upper() != "HAM" or kwargs.get("norm", "none") != "none":
        return False
    substitution = kwargs.get("sm")
    return substitution is None or (
        isinstance(substitution, str) and substitution.upper() == "CONSTANT"
    )


def _lossless_codebook(distance: np.ndarray):
    values = set()
    for start in range(0, distance.size, _CODEBOOK_CHUNK_SIZE):
        values.update(
            np.unique(distance[start : start + _CODEBOOK_CHUNK_SIZE]).tolist()
        )
        if len(values) > _CODEBOOK_MAX_VALUES:
            return distance, None

    codebook = np.asarray(sorted(values), dtype=np.float64)
    if codebook.size <= np.iinfo(np.uint8).max + 1:
        dtype = np.uint8
    else:
        dtype = np.uint16
    encoded_bytes = distance.size * np.dtype(dtype).itemsize + codebook.nbytes
    if encoded_bytes >= distance.nbytes:
        return distance, None
    codes = np.empty(distance.size, dtype=dtype)
    for start in range(0, distance.size, _CODEBOOK_CHUNK_SIZE):
        stop = min(start + _CODEBOOK_CHUNK_SIZE, distance.size)
        codes[start:stop] = np.searchsorted(codebook, distance[start:stop])
    return codes, codebook


def cluster_sequences_pamonce(
    seqdata: SequenceData,
    k: int,
    *,
    method: str,
    distance_kwargs: dict | None = None,
    weights=None,
    threads: int | None = None,
    memory_budget_mb: float | None = None,
    return_diagnostics: bool = False,
):
    """Cluster sequences and return each row's 1-based medoid row position."""
    if not isinstance(seqdata, SequenceData):
        raise ValueError("seqdata must be a SequenceData object.")
    if threads is not None and (
        not isinstance(threads, (int, np.integer)) or threads < 1
    ):
        raise ValueError("threads must be a positive integer or None.")
    if memory_budget_mb is not None and memory_budget_mb <= 0:
        raise ValueError("memory_budget_mb must be positive or None.")
    requested_threads = 0 if threads is None else int(threads)
    memory_budget_bytes = (
        0
        if memory_budget_mb is None
        else int(float(memory_budget_mb) * 1024 * 1024)
    )

    n = seqdata.seqdata.shape[0]
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, {n}].")

    source_weights = weights
    if source_weights is None and getattr(seqdata, "_weights_provided", False):
        source_weights = seqdata.weights
    if source_weights is None:
        weights_cpp = np.empty(0, dtype=np.float64)
    else:
        source_weights = np.asarray(source_weights, dtype=np.float64)
        if source_weights.shape != (n,):
            raise ValueError(f"weights must contain exactly {n} values.")
        weights_cpp = np.ascontiguousarray(source_weights)

    kwargs = dict(distance_kwargs or {})
    if "opts" in kwargs:
        raise ValueError("Pass distance options directly, not through 'opts'.")
    if kwargs.get("refseq") is not None:
        raise ValueError("refseq does not produce an all-pairs distance matrix.")
    kwargs["full_matrix"] = False
    if _can_use_integer_hamming(method, kwargs):
        sequences = np.ascontiguousarray(seqdata.values)
        distance = _integer_hamming_condensed(sequences)
        distance_codebook = None
    else:
        distance = np.ascontiguousarray(
            get_distance_matrix(seqdata=seqdata, method=method, **kwargs),
            dtype=np.float64,
        )
        if method.upper() == "HAM":
            distance, distance_codebook = _lossless_codebook(distance)
        else:
            distance_codebook = None

    expected_length = n * (n - 1) // 2
    if distance.ndim != 1 or distance.size != expected_length:
        raise ValueError(
            f"distance method must return {expected_length} condensed distances."
        )

    engine_args = [
        int(n),
        distance,
        np.arange(k, dtype=np.int32),
        1,
        weights_cpp,
        requested_threads,
        memory_budget_bytes,
    ]
    if distance_codebook is not None:
        engine_args.append(np.empty(0, dtype=np.int32))
        engine_args.append(distance_codebook)
    engine = clustering_c_code.PAMonce(*engine_args)
    engine.set_collect_diagnostics(bool(return_diagnostics))
    membership = np.asarray(
        engine.runclusterloop_one_based(), dtype=np.int32
    )

    if return_diagnostics:
        if distance_codebook is not None:
            distance_representation = "codebook_condensed"
        elif np.issubdtype(distance.dtype, np.unsignedinteger):
            distance_representation = "integer_condensed"
        else:
            distance_representation = "float64_condensed"
        diagnostics = dict(engine.diagnostics())
        diagnostics.update(
            {
                "original_n": int(n),
                "objective": float(engine.objective()),
                "distance_representation": distance_representation,
                "distance_dtype": str(distance.dtype),
                "distance_storage_bytes": int(
                    distance.nbytes
                    + (0 if distance_codebook is None else distance_codebook.nbytes)
                ),
                "distance_codebook_size": int(
                    0 if distance_codebook is None else distance_codebook.size
                ),
            }
        )
        return membership, diagnostics
    return membership
