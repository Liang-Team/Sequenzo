"""
@Author  : 李欣怡
@File    : aggregatecases.py
@Time    : 2024/12/27 10:12
@Desc    : 
"""
import numpy as np

from sequenzo.clustering.utils.aggregate_cases import AggregateCasesResult, aggregate_cases


class DataFrameAggregator:
    def aggregate(self, x, weights=None, **kwargs):
        result = aggregate_cases(x, weights=weights, weighted=True)
        return result.to_dict()


class MatrixAggregator:
    def aggregate(self, x, weights=None, **kwargs):
        result = aggregate_cases(x, weights=weights, weighted=False)
        return result.to_dict()


class StsListAggregator:
    def aggregate(self, x, weights=None, weighted=True, **kwargs):
        result = aggregate_cases(x, weights=weights, weighted=weighted)
        return result.to_dict()


def print_wcAggregateCases(result):
    print(f"Number of disaggregated cases: {len(result['disaggWeights'])}")
    print(f"Number of aggregated cases: {len(result['aggWeights'])}")
    print(f"Average aggregated cases: {len(result['disaggWeights']) / len(result['aggWeights'])}")
    print(f"Average (weighted) aggregation: {np.mean(result['aggWeights'])}")

