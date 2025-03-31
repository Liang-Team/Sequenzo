"""
@Author  : Yuqi Liang 梁彧祺
@File    : qiqi_test_hierarchical_clustering.py
@Time    : 23/03/2025 15:51
@Desc    : 
"""
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
from sequenzo.clustering import Cluster

# CSV 文件路径列表
csv_files = [
    'sampled_20000_data.csv',
    # 'df_sampled_1000_detailed_sequences.csv',
    # 'df_sampled_2000_detailed_sequences.csv',
    # 'df_sampled_3000_detailed_sequences.csv',
    # 'df_sampled_4000_detailed_sequences.csv',
    # 'df_sampled_5000_detailed_sequences.csv',
    # 'df_sampled_10000_detailed_sequences.csv',
    # 'df_sampled_15000_detailed_sequences.csv',
    # 'df_sampled_25000_detailed_sequences.csv'
]

# data_dir = '/Users/lei/Documents/Sequenzo_all_folders/sequenzo_local/test_results/relative_frequency/240210_relative_frequency_test_results'
data_dir = '/home/xinyi/data/detailed_data'

# 存储运行时间和文件名的列表
runtimes = []
filenames = []

# 循环读取每个 CSV 文件并计算运行时间
for filename in csv_files:
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)

    _time = list(df.columns)[4:]
    states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    df = df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']]

    data = SequenceData(df, time=_time, time_type="age", id_col="worker_id", states=states)
    diss = get_distance_matrix(seqdata=data, method="OMspell", sm="TRATE", indel="auto")

    start_time = time.time()
    cluster = Cluster(diss, data.ids, clustering_method='ward')
    end_time = time.time()

    runtime = end_time - start_time
    runtimes.append(runtime)
    filenames.append(filename)

    print(f"File: {filename}, Runtime: {runtime} seconds")

# 绘制折线图
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=filenames, y=runtimes, marker='o')
# plt.xticks(rotation=45, ha='right')  # 旋转 x 轴标签以避免重叠
# plt.xlabel("CSV Filename")
# plt.ylabel("Runtime (seconds)")
# plt.title("Runtime vs CSV Filename")
# plt.tight_layout()
# plt.show()