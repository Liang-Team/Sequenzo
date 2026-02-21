
import sys
import os
import glob
import getopt
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

from tanat.sequence import StateSequencePool, StateSequenceSettings

from tanat.clustering import (
    HierarchicalClusterer,
    HierarchicalClustererSettings,
)
from tanat.metric.sequence import (
    LinearPairwiseSequenceMetric,
    LinearPairwiseSequenceMetricSettings,
    DTWSequenceMetric,
    DTWSequenceMetricSettings,
    EditSequenceMetric,
    EditSequenceMetricSettings,
    LCPSequenceMetric,
    LCPSequenceMetricSettings,
    LCSSequenceMetric,
    LCSSequenceMetricSettings
)

GENERATED_DATASETS_DIR = "generated_datasets"

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"c:i:m:d:",[])
    except getopt.GetoptError:
        print("parameters error")
        sys.exit(2)

    nb_clusters = 2
    it = None
    outputfile = "results.csv"
    metric_name = "edit"
    dataset_name = None

    fout = open(outputfile, "w+")
    fout.write(f'it,method,nb,voc,mean_len,max_dur,metric,cl,prep_time,time,ARI\n')
    fout.close()

    for opt, arg in opts:
       if opt in ("-c"):
           try:
               nb_clusters = int(arg)
           except:
               warnings.warn("error with argument -c: an integer must be given")
               sys.exit(2)
           if nb_clusters<2:
               warnings.warn("warning with argument -c: must be strictly above 1")
               sys.exit(2)
       elif opt in ("-i"):
           try:
               it = int(arg)
           except:
               warnings.warn("error with argument -i: an integer must be given")
               sys.exit(2)
           if it<0:
               warnings.warn("warning with argument -i: must be strictly above 0")
               sys.exit(2)
       elif opt in ("-m"):
            metric_name = str(arg)
       elif opt in ("-d"):
            dataset_name = str(arg)

    # 确定数据文件列表：-d 指定 generated_datasets 下的数据集名，否则使用位置参数
    # 支持 -d output_n5000_l30 或 -d output_n5000_l30.csv（自动去除扩展名查找目录或单文件）
    if dataset_name is not None:
        dataset_dir = os.path.join(GENERATED_DATASETS_DIR, dataset_name)
        if not os.path.isdir(dataset_dir) and "." in dataset_name:
            stem = os.path.splitext(dataset_name)[0]
            dataset_dir_alt = os.path.join(GENERATED_DATASETS_DIR, stem)
            if os.path.isdir(dataset_dir_alt):
                dataset_dir = dataset_dir_alt
            elif os.path.isfile(os.path.join(GENERATED_DATASETS_DIR, dataset_name)):
                # 文件直接在 generated_datasets 根目录
                fnames = [os.path.join(GENERATED_DATASETS_DIR, dataset_name)]
                dataset_dir = None  # 已找到
        if dataset_dir is not None and not os.path.isdir(dataset_dir):
            warnings.warn(f"error: dataset directory not found: {dataset_dir}")
            sys.exit(2)
        if dataset_dir is not None:
            fnames = sorted(glob.glob(os.path.join(dataset_dir, "*.csv")) + glob.glob(os.path.join(dataset_dir, "*.dat")))
            if not fnames:
                warnings.warn(f"error: no .csv or .dat files in {dataset_dir}")
                sys.exit(2)
    else:
        fnames = args
        if not fnames:
            print("用法: python run_tanat.py -c <聚类数> -m <metric> -d <数据集名>")
            print("  或: python run_tanat.py -c <聚类数> -m <metric> file1.csv [file2.csv ...]")
            sys.exit(1)

    start = time.time()
    pd_list=[]
    clusters = []
    total_ids=0

    for i,fname in enumerate(fnames):
        ldata = pd.read_csv(fname)
        # 与 TraMineR 一致: stime/etime +1, 过滤 etime<=stime
        ldata['stime'] = ldata['stime'] + 1
        ldata['etime'] = ldata['etime'] + 1
        ldata = ldata[ldata['etime'] > ldata['stime']]
        ldata['id'] = ldata['id'] + total_ids # add a decay of ids
        total_ids += max(ldata['id'].unique())+1
        pd_list.append( ldata )
        clusters += (len(ldata['id'].unique()) * [ "c"+str(i) ])

    data = pd.concat(pd_list)

    id_list = list(data['id'].unique())
    id_list.sort()
    sdata = pd.DataFrame({'id': id_list, 'c': clusters})

    # 与 TraMineR 一致的时间网格表示：每时间点一列，序列长度=时间点数
    # 构建 pdata: (n_ids, n_timepoints)，与 TraMineR seqdef(SPELL) 等效
    t_min, t_max = int(data['stime'].min()), int(data['etime'].max())
    time_points = list(range(t_min, t_max + 1))
    pdata = pd.DataFrame(-1, index=id_list, columns=time_points, dtype=np.int64)
    for _, row in data.iterrows():
        for d in range(int(row['stime']), int(row['etime']) + 1):
            pdata.loc[row['id'], d] = int(row['event'])

    # 将时间网格转为 spell 格式 (id, stime, etime, event)，每时间点一行
    # 这样 TanaT 的序列 = [state_t1, state_t2, ...]，与 TraMineR 一致
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

    if len(pool) != len(sdata):
        warnings.warn("error: probably the first generated sequence is empty")
        sys.exit(2)

    print(f'{len(pool)}=={len(sdata)},{len(pool.vocabulary)},{len(pool.sequence_data)/len(pool)} (时间网格模式，序列长度={len(time_points)})')

    pool.update_entity_metadata(
        feature_name="event",
        feature_type="categorical"
    )

    # metric definition
    if metric_name=="linear":
        metric = LinearPairwiseSequenceMetric(settings=LinearPairwiseSequenceMetricSettings())
    elif metric_name=="dtw":
        metric = DTWSequenceMetric(settings = DTWSequenceMetricSettings())
    elif metric_name=="edit":
        # OM with Hamming substitution (0 match, 1 mismatch) and indel=1
        metric = EditSequenceMetric(settings=EditSequenceMetricSettings(
            entity_metric="hamming",
            indel_cost=1.0,
            normalize=False
        ))
    elif metric_name=="lcp":
        metric = LCPSequenceMetric(settings=LCPSequenceMetricSettings())
    elif metric_name=="lcs":
        metric = LCSSequenceMetric(settings = LCSSequenceMetricSettings())
    else:
        print("unknown metric name")
        sys.exit(2)
    
    prep_time = time.time() - start

    # 输出距离矩阵（用于与 TraMineR 一致性检验）
    # 序列表示: 时间网格（每时间点一状态，与 TraMineR seqdef SPELL 一致）
    # 参数: OM, sm=Hamming, indel=1（与 TraMineR seqdist method=OM, sm=CONSTANT cval=1, indel=1 一致）
    start = time.time()
    dist_matrix_obj = metric.compute_matrix(pool)
    end = time.time()
    print("[TanaT] time_elapsed:", end - start)
    # if hasattr(dist_matrix_obj, 'to_numpy'):
    #     dist_matrix = dist_matrix_obj.to_numpy()
    # elif hasattr(dist_matrix_obj, 'data'):
    #     dist_matrix = np.asarray(dist_matrix_obj.data)
    # else:
    #     dist_matrix = np.asarray(dist_matrix_obj)
    # dist_df = pd.DataFrame(dist_matrix, index=id_list, columns=id_list)
    # dist_df.to_csv("dist_matrix_tanat.csv")
    # print(f"[TanaT] 距离矩阵已保存至 dist_matrix_tanat.csv, shape={dist_matrix.shape}")

    use_ward_from_features = True  # 用 sklearn Ward 直接从特征计算（支持 ward linkage）

    start = time.time()
    if use_ward_from_features:
        # pdata 已在前面构建（时间网格）
        X = pdata.reindex(id_list).values.astype(np.float64)
        clusterer = AgglomerativeClustering(
            n_clusters=nb_clusters,
            linkage="ward",
            metric="euclidean"
        )
        clusterer.fit(X)
        id_to_label = dict(zip(id_list, clusterer.labels_))
        if "id" in pool.static_data.columns:
            pool.static_data["_c"] = pool.static_data["id"].map(id_to_label).values
        else:
            pool.static_data["_c"] = pool.static_data.index.map(id_to_label).values
    else:
        # 基于距离矩阵的聚类（sklearn 不支持 ward + precomputed，故用 complete 等）
        hc_settings = HierarchicalClustererSettings(
            metric=metric,
            n_clusters=nb_clusters,
            cluster_column="_c",
            linkage="complete"
        )
        clusterer = HierarchicalClusterer(settings=hc_settings)
        clusterer.fit(pool)
    time_elapsed = time.time() - start
    print("[TanaT] time_elapsed:", time_elapsed)

    #print(pool.static_data)
    
    ARI = adjusted_rand_score(pool.static_data['c'], pool.static_data['_c'])
    s=""
    if it is not None:
        s+=f"{it},"
    s += f'TanaT,{len(pool)},{len(pool.vocabulary)},{len(pool.sequence_data)/len(pool)},{t_max},{metric.__class__.__name__},{nb_clusters},{prep_time},{time_elapsed},{ARI}'

    fout = open(outputfile, "a")
    fout.write(s+"\n")
    fout.close()
