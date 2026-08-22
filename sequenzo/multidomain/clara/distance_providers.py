"""
@Author  : Yuqi Liang 梁彧祺
@File    : distance_providers.py
@Time    : 18/05/2026 17:50
@Desc    : 
Distance providers for scalable multidomain CLARA.

CLARA only needs (1) distances within each sample and (2) distances from all
sequences to the current medoids. Providers implement those two operations
without materializing a full N x N distance matrix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sequenzo.define_sequence_data import SequenceData
from sequenzo.multidomain.cat import build_cat_sequence_and_costs
from sequenzo.multidomain.idcd import create_idcd_sequence_from_domains, validate_multidomain_domains

from ._utils import (
    assert_condensed_distance_shape,
    assert_distance_to_medoids_shape,
    assert_sample_distance_shape,
    compute_distance_matrix,
    distance_params_are_data_dependent,
    freeze_seqdist_costs,
    guard_on_demand_distance_params,
    one_based_to_zero_based,
    pairwise_distances_on_indices,
    parallel_map,
    subset_sequence_data,
    warn_large_combined_state_space,
    warm_distance_query_caches,
)

# Picklable workers for DAT domain-level parallelism (joblib processes).
_DATSampleWork = Tuple[SequenceData, Dict[str, Any], Sequence[int], bool]
_DATMedoidWork = Tuple[SequenceData, Dict[str, Any], int, Tuple[int, ...]]


def _dat_sample_domain_distances(work: _DATSampleWork) -> np.ndarray:
    domain, params, sample_indices, condensed = work
    return pairwise_distances_on_indices(
        domain,
        dict(params),
        sample_indices,
        condensed=condensed,
    )


def _dat_medoid_domain_matrix(work: _DATMedoidWork) -> np.ndarray:
    domain, params, n_sequences, medoids = work
    refseq = [np.arange(n_sequences, dtype=np.intp), np.asarray(medoids, dtype=np.intp)]
    dist_args = dict(params)
    return np.asarray(
        compute_distance_matrix(domain, dist_args, refseq=refseq, full_matrix=True),
        dtype=float,
    )

_CAT_METHODS = frozenset({"OM", "HAM"})


def _validate_nonnegative_weights(
    weights: Sequence[float],
    *,
    label: str,
) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values.")
    if np.any(arr < 0):
        raise ValueError(f"{label} must be non-negative.")
    if float(np.sum(arr)) <= 0:
        raise ValueError(f"At least one {label} entry must be positive.")
    return arr


class DistanceProvider(ABC):
    """Interface for on-demand distance computation used by CLARA."""

    @abstractmethod
    def sample_distances(
        self,
        sample_indices: Sequence[int],
        *,
        condensed: bool = False,
    ) -> np.ndarray:
        """Return within-sample distances in square or condensed form."""

    @abstractmethod
    def distance_to_medoids(self, medoid_indices: Sequence[int]) -> np.ndarray:
        """Return an N* x K all-to-medoid matrix."""

    @abstractmethod
    def n_sequences(self) -> int:
        """Number of sequences represented by this provider."""

    def sample_distance_matrix(self, sample_indices: Sequence[int]) -> np.ndarray:
        """Backward-compatible alias for square within-sample distances."""
        return self.sample_distances(sample_indices, condensed=False)


class IDCDDistanceProvider(DistanceProvider):
    """
    Independence from domain costs and distances (IDCD).

    Builds observed combined-state multidomain sequences, then computes
    standard sequence distances on that object.
    """

    def __init__(
        self,
        domains: List[SequenceData],
        *,
        ch_sep: str = "+",
        quiet: bool = True,
        seqdata: Optional[SequenceData] = None,
        **seqdist_args: Any,
    ) -> None:
        if seqdata is None:
            validate_multidomain_domains(domains)
            self._domains = domains
            self._md_seqdata = create_idcd_sequence_from_domains(
                domains,
                ch_sep=ch_sep,
                quiet=quiet,
            )
        else:
            self._domains = domains
            self._md_seqdata = seqdata
        warn_large_combined_state_space(self._md_seqdata)
        if not seqdist_args:
            raise ValueError(
                "distance_params must include seqdist settings, e.g. "
                "{'method': 'OM', 'sm': 'CONSTANT', 'indel': 1, 'norm': 'none'}."
            )
        self._dist_args = freeze_seqdist_costs(
            self._md_seqdata,
            dict(seqdist_args),
        )
        warm_distance_query_caches(self._md_seqdata)

    def n_sequences(self) -> int:
        return int(self._md_seqdata.seqdata.shape[0])

    def sample_distances(
        self,
        sample_indices: Sequence[int],
        *,
        condensed: bool = False,
    ) -> np.ndarray:
        return pairwise_distances_on_indices(
            self._md_seqdata,
            self._dist_args,
            sample_indices,
            condensed=condensed,
        )

    def distance_to_medoids(self, medoid_indices: Sequence[int]) -> np.ndarray:
        medoids = list(map(int, medoid_indices))
        n = self.n_sequences()
        refseq = [np.arange(n, dtype=np.intp), np.asarray(medoids, dtype=np.intp)]
        matrix = compute_distance_matrix(
            self._md_seqdata,
            self._dist_args,
            refseq=refseq,
            full_matrix=True,
        )
        return assert_distance_to_medoids_shape(matrix, n, medoids)

    @property
    def md_seqdata(self) -> SequenceData:
        """Underlying multidomain SequenceData (read-only access)."""
        return self._md_seqdata


class CATDistanceProvider(DistanceProvider):
    """
    Cost additive trick (CAT): multidomain substitution and indel costs are
    derived additively from domain-level costs.
    """

    def __init__(
        self,
        domains: List[SequenceData],
        *,
        method: str = "OM",
        norm: str = "none",
        indel: Any = "auto",
        sm: Optional[Any] = None,
        with_missing: Optional[Any] = None,
        link: str = "sum",
        cval: float = 2,
        miss_cost: float = 2,
        cweight: Optional[List[float]] = None,
        ch_sep: str = "+",
        seqdata: Optional[SequenceData] = None,
        combined_sm: Optional[Any] = None,
        combined_indel: Optional[Any] = None,
        **extra_seqdist_args: Any,
    ) -> None:
        method = str(method).upper()
        if method not in _CAT_METHODS:
            raise ValueError(
                f"CATDistanceProvider currently supports methods in "
                f"{sorted(_CAT_METHODS)}; got {method!r}."
            )

        if seqdata is not None:
            if combined_sm is None:
                raise ValueError(
                    "CATDistanceProvider(seqdata=...) requires combined_sm "
                    "(the already-frozen combined substitution matrix)."
                )
            sm_matrix = combined_sm
            if isinstance(sm_matrix, pd.DataFrame):
                sm_matrix = sm_matrix.to_numpy(dtype=float)
            else:
                sm_matrix = np.asarray(sm_matrix, dtype=float)
            self._md_seqdata = seqdata
            warn_large_combined_state_space(self._md_seqdata)
            self._dist_args = {
                "method": method,
                "sm": sm_matrix,
                "indel": indel if combined_indel is None else combined_indel,
                "norm": norm,
                **extra_seqdist_args,
            }
            guard_on_demand_distance_params(self._dist_args)
            warm_distance_query_caches(self._md_seqdata)
            return

        validate_multidomain_domains(domains)
        if sm is None:
            raise ValueError("CAT requires 'sm': substitution-cost specs per domain.")
        if isinstance(sm, str):
            sm = [sm] * len(domains)
        elif isinstance(sm, (list, tuple)):
            sm = list(sm)
        else:
            raise TypeError(
                "CAT requires sm to be a list or tuple (one entry per domain) or a "
                "single string that is replicated across domains."
            )
        if len(sm) != len(domains):
            raise ValueError(
                f"CAT sm length ({len(sm)}) must match number of domains "
                f"({len(domains)})."
            )

        if cweight is not None:
            cweight_arr = _validate_nonnegative_weights(
                cweight,
                label="cweight",
            )
            cweight = cweight_arr.tolist()
            if len(cweight) != len(domains):
                raise ValueError(
                    f"CAT cweight length ({len(cweight)}) must match number of "
                    f"domains ({len(domains)})."
                )

        bundle = build_cat_sequence_and_costs(
            channels=domains,
            method=method,
            norm=norm,
            indel=indel,
            sm=sm,
            with_missing=with_missing,
            link=link,
            cval=cval,
            miss_cost=miss_cost,
            cweight=cweight,
            ch_sep=ch_sep,
        )
        self._md_seqdata: SequenceData = bundle["seqdata"]
        warn_large_combined_state_space(self._md_seqdata)

        sm_matrix = bundle["sm"]
        if not isinstance(sm_matrix, pd.DataFrame):
            alphabet = bundle["alphabet"]
            sm_matrix = pd.DataFrame(sm_matrix, index=alphabet, columns=alphabet)

        self._dist_args: Dict[str, Any] = {
            "method": method,
            "sm": sm_matrix.to_numpy(dtype=float),
            "indel": bundle["indel"],
            "norm": norm,
            **extra_seqdist_args,
        }
        guard_on_demand_distance_params(self._dist_args)
        warm_distance_query_caches(self._md_seqdata)

    def n_sequences(self) -> int:
        return int(self._md_seqdata.seqdata.shape[0])

    def sample_distances(
        self,
        sample_indices: Sequence[int],
        *,
        condensed: bool = False,
    ) -> np.ndarray:
        return pairwise_distances_on_indices(
            self._md_seqdata,
            self._dist_args,
            sample_indices,
            condensed=condensed,
        )

    def distance_to_medoids(self, medoid_indices: Sequence[int]) -> np.ndarray:
        medoids = list(map(int, medoid_indices))
        n = self.n_sequences()
        refseq = [np.arange(n, dtype=np.intp), np.asarray(medoids, dtype=np.intp)]
        matrix = compute_distance_matrix(
            self._md_seqdata,
            self._dist_args,
            refseq=refseq,
            full_matrix=True,
        )
        return assert_distance_to_medoids_shape(matrix, n, medoids)

    @property
    def md_seqdata(self) -> SequenceData:
        return self._md_seqdata


class DATDistanceProvider(DistanceProvider):
    """
    Distance additive trick (DAT): sum (or mean) of domain-level distances.

    Domain-level distance matrices are independent and can be computed in
    parallel via ``n_jobs_domains``. Defaults to ``1`` so nested parallelism
    does not oversubscribe CPUs when CLARA runs ``R`` iterations with
    ``md_clara(..., n_jobs=-1)``. Set ``n_jobs_domains=-1`` (or the number of
    domains) when CLARA iterations are serial (``n_jobs=1``).
    """

    def __init__(
        self,
        domains: List[SequenceData],
        *,
        method_params: List[Dict[str, Any]],
        domain_weights: Optional[List[float]] = None,
        link: str = "sum",
        n_jobs_domains: int = 1,
    ) -> None:
        validate_multidomain_domains(domains)
        if len(method_params) != len(domains):
            raise ValueError(
                f"method_params length ({len(method_params)}) must match "
                f"number of domains ({len(domains)})."
            )

        link = link.lower()
        if link not in {"sum", "mean"}:
            raise ValueError("link must be 'sum' or 'mean'.")

        self._domains = domains
        self._method_params = [
            freeze_seqdist_costs(domain, dict(params))
            for domain, params in zip(domains, method_params)
        ]
        weights_arr = _validate_nonnegative_weights(
            domain_weights if domain_weights is not None else [1.0] * len(domains),
            label="domain_weights",
        )
        self._weights = weights_arr.tolist()
        if len(self._weights) != len(domains):
            raise ValueError("domain_weights length must match number of domains.")

        self._link = link
        self._weight_sum = float(np.sum(weights_arr))
        self._n_jobs_domains = int(n_jobs_domains)
        if self._n_jobs_domains == 0:
            raise ValueError("n_jobs_domains must not be 0.")
        for domain in self._domains:
            warm_distance_query_caches(domain)

    def n_sequences(self) -> int:
        return int(self._domains[0].seqdata.shape[0])

    @property
    def domain_names(self) -> List[str]:
        """Human-readable domain labels (domain_0, domain_1, ...)."""
        return [f"domain_{index}" for index in range(len(self._domains))]

    @property
    def domain_weights(self) -> List[float]:
        """Weights used to combine domain-level DAT distances."""
        return list(self._weights)

    @property
    def link(self) -> str:
        """DAT combination rule: ``'sum'`` or ``'mean'``."""
        return self._link

    def _combine(self, arrays: List[np.ndarray]) -> np.ndarray:
        combined = np.asarray(arrays[0], dtype=float) * self._weights[0]
        for array, weight in zip(arrays[1:], self._weights[1:]):
            combined += weight * array
        if self._link == "mean":
            combined /= self._weight_sum
        return combined

    def _accumulate_domain_distances(self, iterator) -> np.ndarray:
        """Scale-and-add domain distances without retaining every matrix."""
        combined = None
        for array, weight in iterator:
            if combined is None:
                combined = np.asarray(array, dtype=float) * weight
            else:
                combined += weight * array
        if combined is None:
            raise ValueError("DAT requires at least one domain.")
        if self._link == "mean":
            combined /= self._weight_sum
        return combined

    def _per_domain_sample_distances(
        self,
        sample_indices: Sequence[int],
        *,
        condensed: bool,
    ) -> List[np.ndarray]:
        work = [
            (domain, params, sample_indices, condensed)
            for domain, params in zip(self._domains, self._method_params)
        ]
        return parallel_map(
            _dat_sample_domain_distances,
            work,
            n_jobs=self._n_jobs_domains,
        )

    def _per_domain_medoid_matrices(
        self,
        medoids: Sequence[int],
        *,
        n_sequences: int,
    ) -> List[np.ndarray]:
        medoid_tuple = tuple(int(m) for m in medoids)
        work = [
            (domain, params, n_sequences, medoid_tuple)
            for domain, params in zip(self._domains, self._method_params)
        ]
        return parallel_map(
            _dat_medoid_domain_matrix,
            work,
            n_jobs=self._n_jobs_domains,
        )

    def sample_distances(
        self,
        sample_indices: Sequence[int],
        *,
        condensed: bool = False,
    ) -> np.ndarray:
        if self._n_jobs_domains == 1:
            combined = self._accumulate_domain_distances(
                (
                    _dat_sample_domain_distances(
                        (domain, params, sample_indices, condensed)
                    ),
                    weight,
                )
                for domain, params, weight in zip(
                    self._domains, self._method_params, self._weights
                )
            )
        else:
            per_domain = self._per_domain_sample_distances(
                sample_indices,
                condensed=condensed,
            )
            combined = self._combine(per_domain)
        if condensed:
            return assert_condensed_distance_shape(combined, sample_indices)
        return assert_sample_distance_shape(combined, sample_indices)

    def distance_to_medoids(self, medoid_indices: Sequence[int]) -> np.ndarray:
        medoids = list(map(int, medoid_indices))
        n = self.n_sequences()
        if self._n_jobs_domains == 1:
            combined = self._accumulate_domain_distances(
                (
                    _dat_medoid_domain_matrix((domain, params, n, tuple(medoids))),
                    weight,
                )
                for domain, params, weight in zip(
                    self._domains, self._method_params, self._weights
                )
            )
        else:
            per_domain = self._per_domain_medoid_matrices(medoids, n_sequences=n)
            combined = self._combine(per_domain)
        return assert_distance_to_medoids_shape(combined, n, medoids)

    def per_domain_distance_to_medoids(
        self,
        medoid_indices: Sequence[int],
    ) -> List[np.ndarray]:
        """Return one N* x K raw domain-distance matrix per domain."""
        medoids = list(map(int, medoid_indices))
        n = self.n_sequences()
        per_domain = self._per_domain_medoid_matrices(medoids, n_sequences=n)
        return [
            assert_distance_to_medoids_shape(matrix, n, medoids)
            for matrix in per_domain
        ]

    def weighted_per_domain_distance_to_medoids(
        self,
        medoid_indices: Sequence[int],
    ) -> List[np.ndarray]:
        """
        Return domain-level N* x K matrices after applying DAT combination weights.

        Summing the returned matrices reproduces :meth:`distance_to_medoids`.
        """
        per_domain = self.per_domain_distance_to_medoids(medoid_indices)
        if self._link == "sum":
            scales = self._weights
        else:
            scales = [weight / self._weight_sum for weight in self._weights]
        return [
            scale * matrix
            for scale, matrix in zip(scales, per_domain)
        ]


class SequenceDistanceProvider(DistanceProvider):
    """On-demand distances for a single ``SequenceData`` object.

    Used by leave-one-domain-out when only one domain remains, so the reduced
    fit shares MD-CLARA's ``random_state`` and query path.
    """

    def __init__(
        self,
        seqdata: SequenceData,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        if dist_args is None:
            raise ValueError("SequenceDistanceProvider requires dist_args.")
        self._seqdata = seqdata
        self._dist_args = freeze_seqdist_costs(seqdata, dict(dist_args))
        warm_distance_query_caches(seqdata)

    def n_sequences(self) -> int:
        return int(self._seqdata.seqdata.shape[0])

    def sample_distances(
        self,
        sample_indices: Sequence[int],
        *,
        condensed: bool = False,
    ) -> np.ndarray:
        return pairwise_distances_on_indices(
            self._seqdata,
            self._dist_args,
            sample_indices,
            condensed=condensed,
        )

    def distance_to_medoids(self, medoid_indices: Sequence[int]) -> np.ndarray:
        medoids = list(map(int, medoid_indices))
        n = self.n_sequences()
        refseq = [np.arange(n, dtype=np.intp), np.asarray(medoids, dtype=np.intp)]
        matrix = compute_distance_matrix(
            self._seqdata,
            self._dist_args,
            refseq=refseq,
            full_matrix=True,
        )
        return assert_distance_to_medoids_shape(matrix, n, medoids)


def _cat_init_kwargs(params: Dict[str, Any], n_domains: int) -> Dict[str, Any]:
    """Keyword arguments for :class:`CATDistanceProvider` / CAT cost construction."""
    out = dict(params)
    sm = out.get("sm")
    if isinstance(sm, str):
        out["sm"] = [sm] * n_domains
    return out


def make_aggregated_distance_provider(
    domains: List[SequenceData],
    strategy: str,
    distance_params: Optional[Dict[str, Any]],
    aggregation: Dict[str, Any],
) -> DistanceProvider:
    """
    Build an MD-CLARA provider on unique profiles with costs frozen correctly.

    Data-independent costs (OM CONSTANT with a numeric indel) are estimated on
    the aggregated unique-profile objects, which already carry ``aggWeights``.

    Data-dependent costs (TRATE, INDELS, INDELSLOG) are estimated on the
    original weighted cases, then sequences are subset to unique profiles.
    ``indel='auto'`` with CONSTANT or an explicit substitution matrix is
    resolved to an explicit number from those substitution costs; it is not
    treated as a frequency-dependent cost.
    """
    strategy = strategy.lower()
    params = dict(distance_params or {})
    agg_idx = one_based_to_zero_based(aggregation["aggIndex"], name="aggIndex")
    agg_weights = np.asarray(aggregation["aggWeights"], dtype=float)
    agg_domains = [
        subset_sequence_data(domain, agg_idx, weights=agg_weights)
        for domain in domains
    ]
    freeze_on_original = distance_params_are_data_dependent(strategy, params)

    if strategy == "dat":
        method_params = list(params.get("method_params", []))
        if not method_params:
            raise ValueError("DAT strategy requires distance_params['method_params'].")
        source_domains = domains if freeze_on_original else agg_domains
        frozen = dict(params)
        frozen["method_params"] = [
            freeze_seqdist_costs(domain, dict(block))
            for domain, block in zip(source_domains, method_params)
        ]
        return DATDistanceProvider(agg_domains, **frozen)

    if strategy == "idcd":
        ch_sep = params.get("ch_sep", "+")
        seqdist = {key: value for key, value in params.items() if key != "ch_sep"}
        if freeze_on_original:
            original_md = create_idcd_sequence_from_domains(
                domains, ch_sep=ch_sep, quiet=True
            )
            frozen = freeze_seqdist_costs(original_md, seqdist)
            subset = subset_sequence_data(
                original_md, agg_idx, weights=agg_weights
            )
            return IDCDDistanceProvider(
                agg_domains,
                seqdata=subset,
                ch_sep=ch_sep,
                quiet=True,
                **frozen,
            )
        return IDCDDistanceProvider(agg_domains, ch_sep=ch_sep, quiet=True, **seqdist)

    if strategy == "cat":
        cat_kwargs = _cat_init_kwargs(params, n_domains=len(domains))
        if freeze_on_original:
            bundle = build_cat_sequence_and_costs(channels=domains, **{
                key: cat_kwargs[key]
                for key in (
                    "method",
                    "norm",
                    "indel",
                    "sm",
                    "with_missing",
                    "link",
                    "cval",
                    "miss_cost",
                    "cweight",
                    "ch_sep",
                )
                if key in cat_kwargs
            })
            subset = subset_sequence_data(
                bundle["seqdata"], agg_idx, weights=agg_weights
            )
            return CATDistanceProvider(
                agg_domains,
                method=str(cat_kwargs.get("method", "OM")),
                norm=str(cat_kwargs.get("norm", "none")),
                seqdata=subset,
                combined_sm=bundle["sm"],
                combined_indel=bundle["indel"],
            )
        return CATDistanceProvider(agg_domains, **cat_kwargs)

    raise ValueError("strategy must be one of: 'idcd', 'cat', 'dat'")


def make_distance_provider(
    domains: List[SequenceData],
    strategy: str,
    distance_params: Optional[Dict[str, Any]] = None,
) -> DistanceProvider:
    """Factory for IDCD, CAT, and DAT distance providers."""
    strategy = strategy.lower()
    params = dict(distance_params or {})

    if strategy == "idcd":
        return IDCDDistanceProvider(domains, **params)
    if strategy == "cat":
        return CATDistanceProvider(domains, **params)
    if strategy == "dat":
        if "method_params" not in params:
            raise ValueError("DAT strategy requires distance_params['method_params'].")
        return DATDistanceProvider(domains, **params)
    raise ValueError("strategy must be one of: 'idcd', 'cat', 'dat'")


__all__ = [
    "DistanceProvider",
    "IDCDDistanceProvider",
    "CATDistanceProvider",
    "DATDistanceProvider",
    "SequenceDistanceProvider",
    "make_distance_provider",
    "make_aggregated_distance_provider",
]
