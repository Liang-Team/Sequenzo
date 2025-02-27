"""
@Author  : 李欣怡
@File    : aggregatecases.py
@Time    : 2024/12/27 10:12
@Desc    : 
"""
import pandas as pd
import numpy as np


class WcAggregateCases:
    def aggregate(self, x, weights=None, **kwargs):
        """
        The appropriate aggregation method is invoked dynamically depending on the type of x
        """
        method_name = f"aggregate_{type(x).__name__}"
        method = getattr(self, method_name, None)

        if method is None:
            raise NotImplementedError(f"No aggregation method for type {type(x).__name__}")

        return method(x, weights, **kwargs)


class WcAggregateCasesInternal:
    def aggregate(self, x, weights=None):
        x = pd.DataFrame(x)
        lx = len(x)

        if weights is None:
            weights = np.ones(lx)

        ids = x.apply(lambda row: "@@@WC_SEP@@".join(row.astype(str)), axis=1)

        mcorr = [np.nan] * lx

        def myfunction(group):
            first_element = group.iloc[0]

            for idx in group:
                mcorr[idx] = first_element
            weighted_sum = np.sum(weights[group])
            return [first_element, weighted_sum]

        df = pd.DataFrame({
            'index': range(0, lx),
            'id': ids
        })

        grouped = df.groupby('id')['index'].apply(myfunction)

        agg_df = pd.DataFrame(grouped.tolist(), columns=['aggIndex', 'aggWeights'])

        aggIndex = agg_df['aggIndex']
        mcorr2 = [aggIndex[aggIndex == val].index[0] if val in aggIndex.values else -1 for val in mcorr]

        ret = {
            "aggIndex": agg_df['aggIndex'].values,
            "aggWeights": agg_df['aggWeights'].values,
            "disaggIndex": mcorr2,
            "disaggWeights": weights
        }

        return ret


class DataFrameAggregator(WcAggregateCases):
    def aggregate_DataFrame(self, x, weights=None, **kwargs):
        internal = WcAggregateCasesInternal()
        return internal.aggregate(x, weights)


class MatrixAggregator(WcAggregateCases):
    def aggregate_ndarray(self, x, weights=None, **kwargs):
        internal = WcAggregateCasesInternal()
        return internal.aggregate(x, weights)


class StsListAggregator(WcAggregateCases):
    def aggregate_stslist(self, x, weights=None, weighted=True, **kwargs):
        if weights is None and weighted:
            weights = getattr(x, "weights", None)
        internal = WcAggregateCasesInternal()
        return internal.aggregate(x, weights)


# Print function (for output)
def print_wcAggregateCases(result):
    print(f"Number of disaggregated cases: {len(result['disaggWeights'])}")
    print(f"Number of aggregated cases: {len(result['aggWeights'])}")
    print(f"Average aggregated cases: {len(result['disaggWeights']) / len(result['aggWeights'])}")
    print(f"Average (weighted) aggregation: {np.mean(result['aggWeights'])}")
