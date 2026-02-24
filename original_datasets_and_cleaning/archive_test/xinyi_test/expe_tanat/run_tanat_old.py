
import sys
import getopt
import warnings
import time
import pandas as pd
from sklearn.metrics import adjusted_rand_score

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

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"c:i:m:",[])
    except getopt.GetoptError:
        print("parameters error")
        sys.exit(2)

    nb_clusters = 2
    it = None
    outputfile = "results.csv"
    metric_name = "edit"

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

    start = time.time()
    pd_list=[]
    clusters = []
    total_ids=0

    for i,fname in enumerate(args):
        ldata = pd.read_csv(fname)
        ldata['id'] = ldata['id'] + total_ids # add a decay of ids
        total_ids += max(ldata['id'].unique())+1
        pd_list.append( ldata )
        clusters += (len(ldata['id'].unique()) * [ "c"+str(i) ])
        #print("t_ids:",total_ids, min(ldata['id'].unique()), max(ldata['id'].unique()))

    data = pd.concat(pd_list)

    #print(data)

    settings = StateSequenceSettings(
        id_column="id",
        start_column="stime",
        end_column="etime",
        entity_features=["event"],
        static_features=["c"]
    )

    id_list = list(data['id'].unique())
    id_list.sort()
    sdata = pd.DataFrame({'id': id_list, 'c':clusters})

    pool = StateSequencePool(data, static_data=sdata, settings=settings)


    if len(pool)!=len(sdata) :
        warnings.warn("error: probably the first generated sequence is empty")
        sys.exit(2)

    print(f'{len(pool)}=={len(sdata)},{len(pool.vocabulary)},{len(pool.sequence_data)/len(pool)}')

    # events are categorical (required for some metrics)
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
        metric = EditSequenceMetric(settings = EditSequenceMetricSettings())
    elif metric_name=="lcp":
        metric = LCPSequenceMetric(settings=LCPSequenceMetricSettings())
    elif metric_name=="lcs":
        metric = LCSSequenceMetric(settings = LCSSequenceMetricSettings())
    else:
        print("unknown metric name")
        sys.exit(2)
    
    prep_time = time.time() - start

    hc_settings = HierarchicalClustererSettings(
        metric=metric,
        n_clusters=nb_clusters,
        cluster_column="_c"
    )

    clusterer = HierarchicalClusterer(settings=hc_settings)

    start = time.time()
    clusterer.fit(pool)
    time_elapsed = time.time() - start

    #print(pool.static_data)
    
    ARI = adjusted_rand_score(pool.static_data['c'], pool.static_data['_c'])
    s=""
    if it is not None:
        s+=f"{it},"
    s += f'TanaT,{len(pool)},{len(pool.vocabulary)},{len(pool.sequence_data)/len(pool)},{max(data["etime"])},{metric.__class__.__name__},{nb_clusters},{prep_time},{time_elapsed},{ARI}'

    fout = open(outputfile, "a")
    fout.write(s+"\n")
    fout.close()
