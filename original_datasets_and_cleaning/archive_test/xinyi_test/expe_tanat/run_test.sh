#!/usr/bash
exp="results/EXP1_LENGTH_$(date +%F)_$(date +%s)"

mkdir -p "$exp/"
file="$exp/results.csv"

echo -e "it,method,nb,voc,mean_len,max_dur,metric,cl,prep_time,time,ARI" >> $file


for d in 20 #vocabulary size
do
for l in 8 20 #mean seq size
do
for n in 100 250 500 1000 #number of sequences
do
    for it in {1..5}
    do
        # 数据集名称，用于 generated_datasets 下的子目录
        dataset_name="n${n}_l${l}_d${d}_it${it}"
        dataset_dir="generated_datasets/${dataset_name}"
        mkdir -p "$dataset_dir"

        # generate datasets -> generated_datasets/<dataset_name>/
        #   'temporal': duraction is about 1000
        #   'temporal_small': duration is about 100
        python random-sequences-generation/generation.py -n $n -l $l -t 'temporal_small' --th=1.0 --np=1 --pd -o "${dataset_dir}/output1.dat"
        python random-sequences-generation/generation.py -n $n -l $l -t 'temporal_small' --th=1.0 --np=1 --pd -o "${dataset_dir}/output2.dat"
        python random-sequences-generation/generation.py -n $n -l $l -t 'temporal_small' --th=1.0 --np=1 --pd -o "${dataset_dir}/output3.dat"
        python random-sequences-generation/generation.py -n $n -l $l -t 'temporal_small' --th=1.0 --np=1 --pd -o "${dataset_dir}/output4.dat"

        for metric in linear dtw edit lcp lcs
        do
            python run_tanat.py -i $it -c 4 -m $metric -d $dataset_name
            tail -n +2 results.csv >> $file
        done

        for metric in OM EUCLID LCS LCP TWED
        do
            rm -f results_R.txt
            Rscript --vanilla run_traminer_clustering.R -i $it -c 4 -m $metric -d $dataset_name
            more results_R.txt >> $file
        done

        python run_sequenzo.py -i $it -c 4 -d $dataset_name
        tail -n +2 results_sqz.csv >> $file
    done
done
done
done

