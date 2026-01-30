"""
@Author  : 李欣怡
@File    : xinyi_test_dissmilarity_measures.py
@Time    : 2025/3/30 20:06
@Desc    : 
"""
import pandas as pd
import time
import os

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix

U_files = [
    'synthetic_detailed_U5_N500.csv',
    # 'synthetic_detailed_U5_N1000.csv',
    # 'synthetic_detailed_U5_N1500.csv',
    # 'synthetic_detailed_U5_N2000.csv',
    # 'synthetic_detailed_U5_N2500.csv',
    # 'synthetic_detailed_U5_N3000.csv',
    # 'synthetic_detailed_U5_N3500.csv',
    # 'synthetic_detailed_U5_N4000.csv',
    # 'synthetic_detailed_U5_N4500.csv',
    # 'synthetic_detailed_U5_N5000.csv',
    # 'synthetic_detailed_U5_N10000.csv',
    # 'synthetic_detailed_U5_N15000.csv',
    # 'synthetic_detailed_U5_N20000.csv',
    # 'synthetic_detailed_U5_N25000.csv',
    # 'synthetic_detailed_U5_N30000.csv',
    # 'synthetic_detailed_U5_N35000.csv',
    # 'synthetic_detailed_U5_N40000.csv',
    # 'synthetic_detailed_U5_N45000.csv',
    # 'synthetic_detailed_U5_N50000.csv',
    
    # 'synthetic_detailed_U25_N500.csv',
    # 'synthetic_detailed_U25_N1500.csv',
    # 'synthetic_detailed_U25_N2000.csv',
    # 'synthetic_detailed_U25_N2500.csv',
    # 'synthetic_detailed_U25_N3000.csv',
    # 'synthetic_detailed_U25_N3500.csv',
    # 'synthetic_detailed_U25_N4000.csv',
    # 'synthetic_detailed_U25_N4500.csv',
    # 'synthetic_detailed_U25_N5000.csv',
    # 'synthetic_detailed_U25_N10000.csv',
    # 'synthetic_detailed_U25_N15000.csv',
    # 'synthetic_detailed_U25_N20000.csv',
    # 'synthetic_detailed_U25_N25000.csv',
    # 'synthetic_detailed_U25_N30000.csv',
    # 'synthetic_detailed_U25_N35000.csv',
    # 'synthetic_detailed_U25_N40000.csv',
    # 'synthetic_detailed_U25_N45000.csv',
    # 'synthetic_detailed_U25_N50000.csv',
    
    # 'synthetic_detailed_U50_N500.csv',
    # 'synthetic_detailed_U50_N1000.csv',
    # 'synthetic_detailed_U50_N1500.csv',
    # 'synthetic_detailed_U50_N2000.csv',
    # 'synthetic_detailed_U50_N2500.csv',
    # 'synthetic_detailed_U50_N3000.csv',
    # 'synthetic_detailed_U50_N3500.csv',
    # 'synthetic_detailed_U50_N4000.csv',
    # 'synthetic_detailed_U50_N4500.csv',
    # 'synthetic_detailed_U50_N5000.csv',
    # 'synthetic_detailed_U50_N10000.csv',
    # 'synthetic_detailed_U50_N15000.csv',
    # 'synthetic_detailed_U50_N20000.csv',
    # 'synthetic_detailed_U50_N25000.csv',
    # 'synthetic_detailed_U50_N30000.csv',
    # 'synthetic_detailed_U50_N35000.csv',
    # 'synthetic_detailed_U50_N40000.csv',
    # 'synthetic_detailed_U50_N45000.csv',
    # 'synthetic_detailed_U50_N50000.csv',
    
    # 'synthetic_detailed_U85_N500.csv',
    # 'synthetic_detailed_U85_N1500.csv',
    # 'synthetic_detailed_U85_N2000.csv',
    # 'synthetic_detailed_U85_N2500.csv',
    # 'synthetic_detailed_U85_N3000.csv',
    # 'synthetic_detailed_U85_N3500.csv',
    # 'synthetic_detailed_U85_N4000.csv',
    # 'synthetic_detailed_U85_N4500.csv',
    # 'synthetic_detailed_U85_N5000.csv',
    # 'synthetic_detailed_U85_N10000.csv',
    # 'synthetic_detailed_U85_N15000.csv',
    # 'synthetic_detailed_U85_N20000.csv',
    # 'synthetic_detailed_U85_N25000.csv',
    # 'synthetic_detailed_U85_N30000.csv',
    # 'synthetic_detailed_U85_N35000.csv',
    # 'synthetic_detailed_U85_N40000.csv',
    # 'synthetic_detailed_U85_N45000.csv',
    # 'synthetic_detailed_U85_N50000.csv',
]

# 这里是星云存放数据文件的路径
data_dir = 'D:/college/research/QiQi/sequenzo/data_and_output/orignal data/not_real_detailed_data'

# 存储运行时间和文件名的列表
runtimes = []
filenames = []

# 循环读取每个 CSV 文件并计算运行时间
for filename in U_files:
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)
    
    _time = list(df.columns)[2:]
    states = ["Data", "Data science", "Hardware", "Research", "Software", "Support & test", "Systems & infrastructure"]
    df = df[['id', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]

    data = SequenceData(df, time=_time, id_col="id", states=states)
    
    start_time = time.time()
    # 'method' 可以换成不同的参数，比如 OMspell, HAM, DHD
    diss = get_distance_matrix(seqdata=data, method="OM", sm="CONSTANT", indel=1)
    end_time = time.time()

    runtime = end_time - start_time
    runtimes.append(runtime)
    filenames.append(filename)

    print(f"File: {filename}, Runtime: {runtime:.4f} seconds")

print(runtimes)