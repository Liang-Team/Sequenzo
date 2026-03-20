"""
Benchmark TanaT: LCP, LCS, Linear(EUCLID) distance matrix computation time.
Usage: python run_tanat_other.py -d <dataset_name>
"""
import sys
import os
import glob
import getopt
import warnings
import time
import pandas as pd
import numpy as np

from tanat.sequence import StateSequencePool, StateSequenceSettings
from tanat.metric.sequence import (
    LinearPairwiseSequenceMetric,
    LinearPairwiseSequenceMetricSettings,
    LCPSequenceMetric,
    LCPSequenceMetricSettings,
    LCSSequenceMetric,
    LCSSequenceMetricSettings,
)

GENERATED_DATASETS_DIR = "generated_datasets"

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:", [])
    except getopt.GetoptError:
        print("Usage: python run_tanat_other.py -d <dataset_name>")
        sys.exit(2)

    dataset_name = None
    for opt, arg in opts:
        if opt == "-d":
            dataset_name = str(arg)

    if dataset_name is None:
        print("Usage: python run_tanat_other.py -d <dataset_name>")
        sys.exit(1)

    dataset_dir = os.path.join(GENERATED_DATASETS_DIR, dataset_name)
    if not os.path.isdir(dataset_dir):
        warnings.warn(f"error: dataset directory not found: {dataset_dir}")
        sys.exit(2)
    fnames = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")) + glob.glob(os.path.join(dataset_dir, "*.dat")))
    if not fnames:
        warnings.warn(f"error: no .csv or .dat files in {dataset_dir}")
        sys.exit(2)

    # Data preparation (same as run_tanat.py)
    pd_list = []
    clusters = []
    total_ids = 0
    for i, fname in enumerate(fnames):
        ldata = pd.read_csv(fname)
        ldata['stime'] = ldata['stime'] + 1
        ldata['etime'] = ldata['etime'] + 1
        ldata = ldata[ldata['etime'] > ldata['stime']]
        ldata['id'] = ldata['id'] + total_ids
        total_ids += max(ldata['id'].unique()) + 1
        pd_list.append(ldata)
        clusters += (len(ldata['id'].unique()) * ["c" + str(i)])

    data = pd.concat(pd_list)
    id_list = list(data['id'].unique())
    id_list.sort()
    sdata = pd.DataFrame({'id': id_list, 'c': clusters})

    t_min, t_max = int(data['stime'].min()), int(data['etime'].max())
    time_points = list(range(t_min, t_max + 1))
    pdata = pd.DataFrame(-1, index=id_list, columns=time_points, dtype=np.int64)
    for _, row in data.iterrows():
        for d in range(int(row['stime']), int(row['etime']) + 1):
            pdata.loc[row['id'], d] = int(row['event'])

    rows_grid = []
    for idx in id_list:
        for t in time_points:
            rows_grid.append({
                'id': idx,
                'stime': t,
                'etime': t,
                'event': int(pdata.loc[idx, t])
            })
    data_grid = pd.DataFrame(rows_grid)

    settings = StateSequenceSettings(
        id_column="id",
        start_column="stime",
        end_column="etime",
        entity_features=["event"],
        static_features=["c"]
    )

    pool = StateSequencePool(data_grid, static_data=sdata, settings=settings)
    pool.update_entity_metadata(feature_name="event", feature_type="categorical")

    print(f"Dataset: {dataset_name}, n={len(pool)}, seq_length={len(time_points)}")
    print("=" * 60)

    # Define metrics: TanaT name -> metric object
    metrics = {
        "LCP": LCPSequenceMetric(settings=LCPSequenceMetricSettings()),
        "LCS": LCSSequenceMetric(settings=LCSSequenceMetricSettings()),
        "EUCLID": LinearPairwiseSequenceMetric(settings=LinearPairwiseSequenceMetricSettings()),
    }

    for name, metric in metrics.items():
        start = time.time()
        dist_matrix_obj = metric.compute_matrix(pool)
        elapsed = time.time() - start
        print(f"[TanaT] {name} time_elapsed: {elapsed:.4f}")

    print("=" * 60)
