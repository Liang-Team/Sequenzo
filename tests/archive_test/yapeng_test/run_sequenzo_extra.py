"""
Benchmark Sequenzo: OMspell, OMtspell, HAM, DHD
Exp2 only (TanaT does not support these metrics)

Usage: python run_sequenzo_extra.py -d <dataset_name>
Run from: tests/archive_test/yapeng_test/
"""
import sys
import os
import glob
import getopt
import warnings
import time
import numpy as np
import pandas as pd
from sequenzo import SequenceData, get_distance_matrix

GENERATED_DATASETS_DIR = "generated_datasets"


def load_data(dataset_dir):
    """Load and prepare data. Returns a fresh SequenceData each call."""
    fnames = sorted(
        glob.glob(os.path.join(dataset_dir, "*.csv"))
        + glob.glob(os.path.join(dataset_dir, "*.dat"))
    )
    if not fnames:
        raise FileNotFoundError(f"No .csv or .dat files in {dataset_dir}")

    pd_list = []
    total_ids = 0
    for fname in fnames:
        ldata = pd.read_csv(fname)
        ldata['id'] = ldata['id'] + total_ids
        total_ids += max(ldata['id'].unique()) + 1
        pd_list.append(ldata)

    data = pd.concat(pd_list)
    pdata = data.pivot_table('event', ['id'], 'stime', aggfunc='max')
    for _, row in data.iterrows():
        for d in range(row['stime'], row['etime'] + 1):
            pdata.loc[row['id'], d] = int(row['event'])
    pdata.fillna(-1, inplace=True)

    states = list(data['event'].unique())
    states.append(-1)

    sequence_data = SequenceData(
        pdata, time=list(pdata.columns), states=states, missing_values=-1
    )
    return sequence_data, len(data['id'].unique()), len(pdata.columns)


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:", [])
    except getopt.GetoptError:
        print("Usage: python run_sequenzo_extra.py -d <dataset_name>")
        sys.exit(2)

    dataset_name = None
    for opt, arg in opts:
        if opt == "-d":
            dataset_name = str(arg)

    if dataset_name is None:
        print("Usage: python run_sequenzo_extra.py -d <dataset_name>")
        sys.exit(1)

    dataset_dir = os.path.join(GENERATED_DATASETS_DIR, dataset_name)
    if not os.path.isdir(dataset_dir):
        warnings.warn(f"error: dataset directory not found: {dataset_dir}")
        sys.exit(2)

    # Get basic info from first load
    seq_info, n_seq, n_time = load_data(dataset_dir)
    print(f"Dataset: {dataset_name}, n={n_seq}, time_points={n_time}")
    print("=" * 60)

    # Metrics to benchmark
    metric_names = ["OMspell", "OMtspell", "HAM", "DHD"]

    for display_name in metric_names:
        try:
            # Fresh SequenceData for each metric to avoid state pollution
            seq_fresh, _, _ = load_data(dataset_dir)

            # Build kwargs per metric using seq_fresh
            if display_name == "OMspell":
                actual_method = "OMspell"
                kwargs = {"sm": "TRATE"}
            elif display_name == "OMtspell":
                actual_method = "OMspell"
                kwargs = {
                    "sm": "TRATE",
                    "tokdep_coeff": np.ones(len(seq_fresh.states), dtype=np.float64),
                }
            elif display_name == "HAM":
                actual_method = "HAM"
                kwargs = {}
            elif display_name == "DHD":
                actual_method = "DHD"
                kwargs = {}

            start = time.time()
            mtx = get_distance_matrix(seq_fresh, actual_method, **kwargs)
            elapsed = time.time() - start
            print(f"[Sequenzo] {display_name} time_elapsed: {elapsed:.4f}")
        except Exception as e:
            import traceback
            print(f"[Sequenzo] {display_name} ERROR: {e}")
            traceback.print_exc()

    print("=" * 60)
