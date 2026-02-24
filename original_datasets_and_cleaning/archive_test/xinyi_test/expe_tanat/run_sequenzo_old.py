
import sys
import getopt
import warnings
import time
import pandas as pd
from sequenzo import SequenceData, get_distance_matrix, Cluster

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"c:i:",[])
    except getopt.GetoptError:
        print("parameters error")
        sys.exit(2)

    nb_clusters = 2
    it = None
    outputfile = "results_sqz.csv"

    fout = open(outputfile, "w+")
    fout.write(f'it,method,nb,voc,mean_len,metric,cl,prep_time,time,ARI\n')
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

    data = pd.concat(pd_list)

    pdata = data.pivot_table('event', ['id'], 'stime', aggfunc='max')

    for index, row in data.iterrows():
        for d in range(row['stime'], row['etime']+1):
            pdata.loc[row['id'],d]=int(row['event'])

    pdata.fillna(-1, inplace=True)

    states = list(data['event'].unique())
    states.append(-1)

    ids = list(data['id'].unique())

    sequence_data = SequenceData(pdata, time=list(pdata.columns), states=states, missing_values=-1)
    prep_time = time.time() - start
    
    # for metric in ["EUCLID", "LCS", "LCP", "OM"]:# "TWED"]:
    for metric in ["OM"]:# "TWED"]:

        start = time.time()
        
        if metric=="OM":
            mtx = get_distance_matrix(
                sequence_data,
                metric,
                sm="TRATE"
            )
        elif metric=="TWED":
            mtx = get_distance_matrix(
                sequence_data,
                metric,
                sm="CONSTANT",
                nu=0.5
            )
        else:
            mtx = get_distance_matrix(
                sequence_data,
                metric
            )

        cluster = Cluster(
            matrix=mtx, 
            entity_ids=ids, 
            clustering_method="ward"
        )  
        time_elapsed = time.time() - start

        
        ARI = ""
        s=""
        if it is not None:
            s+=f"{it},"
        s += f'Sequenzo,{len(ids)},{len(sequence_data.alphabet)},{len(data)/len(ids)},{max(data["etime"])},{metric},{nb_clusters},{prep_time},{time_elapsed},{ARI}'

        fout = open(outputfile, "a")
        fout.write(s+"\n")
        fout.close()
