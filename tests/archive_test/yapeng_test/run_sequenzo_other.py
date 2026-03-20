"""
Benchmark Sequenzo: LCP, LCS, EUCLID distance matrix computation time.
Usage: python run_sequenzo_other.py -d <dataset_name>
"""
import sys
import os
import glob
import getopt
import warnings
import time
import pandas as pd
from sequenzo import SequenceData, get_distance_matrix

GENERATED_DATASETS_DIR = "generated_datasets"

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:", [])
    except getopt.GetoptError:
        print("Usage: python run_sequenzo_other.py -d <dataset_name>")
        sys.exit(2)

    dataset_name = None
    for opt, arg in opts:
        if opt == "-d":
            dataset_name = str(arg)

    if dataset_name is None:
        print("Usage: python run_sequenzo_other.py -d <dataset_name>")
        sys.exit(1)

    dataset_dir = os.path.join(GENERATED_DATASETS_DIR, dataset_name)
    if not os.path.isdir(dataset_dir):
        warnings.warn(f"error: dataset directory not found: {dataset_dir}")
        sys.exit(2)
    fnames = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")) + glob.glob(os.path.join(dataset_dir, "*.dat")))
    if not fnames:
        warnings.warn(f"error: no .csv or .dat files in {dataset_dir}")
        sys.exit(2)

    # Data preparation (same as run_sequenzo.py)
    pd_list = []
    total_ids = 0
    for i, fname in enumerate(fnames):
        ldata = pd.read_csv(fname)
        ldata['id'] = ldata['id'] + total_ids
        total_ids += max(ldata['id'].unique()) + 1
        pd_list.append(ldata)

    data = pd.concat(pd_list)
    pdata = data.pivot_table('event', ['id'], 'stime', aggfunc='max')
    for index, row in data.iterrows():
        for d in range(row['stime'], row['etime'] + 1):
            pdata.loc[row['id'], d] = int(row['event'])
    pdata.fillna(-1, inplace=True)

    states = list(data['event'].unique())
    states.append(-1)
    ids = list(data['id'].unique())

    sequence_data = SequenceData(pdata, time=list(pdata.columns), states=states, missing_values=-1)

    print(f"Dataset: {dataset_name}, n={len(ids)}, time_points={len(pdata.columns)}")
    print("=" * 60)

    # Run LCP, LCS, EUCLID
    for metric in ["LCP", "LCS", "EUCLID"]:
        start = time.time()
        mtx = get_distance_matrix(sequence_data, metric)
        elapsed = time.time() - start
        print(f"[Sequenzo] {metric} time_elapsed: {elapsed:.4f}")

    print("=" * 60)
