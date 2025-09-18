"""
@Author  : 李欣怡
@File    : xinyi_test_dissmilarity_measures.py
@Time    : 2025/3/30 20:06
@Desc    : 
"""
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sequenzo.define_sequence_data import SequenceData
from sequenzo.dissimilarity_measures.get_distance_matrix import get_distance_matrix
from sequenzo.clustering.KMedoids import KMedoids

# CSV 文件路径列表
csv_files = [
    # 'country_co2_emissions_missing.csv',
    # 'sampled_500_data.csv',
    # 'sampled_1000_data.csv',
    # 'sampled_1500_data.csv',
    # 'sampled_2000_data.csv',
    # 'sampled_2500_data.csv',
    # 'sampled_3000_data.csv',
    # 'sampled_3500_data.csv',
    # 'sampled_4000_data.csv',
    # 'sampled_4500_data.csv',
    # 'sampled_5000_data.csv',
    # 'sampled_10000_data.csv',
    # 'sampled_15000_data.csv',
    # 'sampled_20000_data.csv',
    # 'sampled_25000_data.csv',
    # 'sampled_30000_data.csv',
    # 'sampled_35000_data.csv',
    # 'sampled_38900_data.csv',
    # 'sampled_40000_data.csv',
    #  'sampled_45000_data.csv',
    # 'sampled_50000_data.csv',
    # 'sampled_55000_data.csv',
    # 'sampled_60000_data.csv',
]

U_files = [

    # 'synthetic_detailed_U5_N35000.csv',
    # 'synthetic_detailed_U5_N40000.csv',
    # 'synthetic_detailed_U5_N45000.csv',
    # 'synthetic_detailed_U5_N50000.csv',
    # 'synthetic_detailed_U25_N35000.csv',
    # 'synthetic_detailed_U25_N40000.csv',
    # 'synthetic_detailed_U25_N45000.csv',
    # 'synthetic_detailed_U25_N50000.csv',
    'synthetic_detailed_U50_N500.csv',
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
    # 'synthetic_detailed_U85_N35000.csv',
    # 'synthetic_detailed_U85_N40000.csv',
    # 'synthetic_detailed_U85_N45000.csv',
    # 'synthetic_detailed_U85_N50000.csv',
]

# data_dir = '/home/xinyi_test/data/detailed_data'
# data_dir = 'D:/college/research/QiQi/sequenzo/files/detialed_transposed.csv'
# data_dir = 'D:\\college\\research\\QiQi\\sequenzo\\data_and_output\\sampled_data_sets\\broad_data'
# data_dir = 'D:\\college\\research\\QiQi\\sequenzo\\data_and_output\\sampled_data_sets\\detailed_data'
data_dir = 'D:/college/research/QiQi/sequenzo/data_and_output/orignal data/not_real_detailed_data'

# 存储运行时间和文件名的列表
runtimes = []
filenames = []

# 循环读取每个 CSV 文件并计算运行时间
for filename in csv_files:
    file_path = os.path.join(data_dir, filename)
    df = pd.read_csv(file_path)
    # df = pd.read_csv(data_dir)

    # _time = list(df.columns)[4:]
    # states = ['data', 'data & intensive math', 'hardware', 'research', 'software', 'software & hardware', 'support & test']
    # df = df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']]
    #
    _time = list(df.columns)[2:]
    states = ["Data", "Data science", "Hardware", "Research", "Software", "Support & test", "Systems & infrastructure"]
    df = df[['id', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]

    # _time = list(df.columns)[4:]
    # states = ['Non-computing', 'Non-technical computing', 'Technical computing']
    # df = df[['worker_id', 'C1', 'C2', 'C3', 'C4', 'C5']]
    # df = df.drop_duplicates(subset=['worker_id'])

    # _time = list(df.columns)[1:]
    # states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

    # data = SequenceData(df, time=_time, time_type="age", id_col="worker_id", states=states)
    data = SequenceData(df, time=_time, time_type="age", id_col="id", states=states)
    # data = SequenceData(df, time=_time, time_type="year", id_col="country", states=states)

    # refseq = [[0, 1, 2], [99, 100]]

    start_time = time.time()
    diss = get_distance_matrix(seqdata=data, method="OM", sm="CONSTANT", indel=1)

    # centroids = [1, 50, 100, 150]
    # clustering = KMedoids(diss=diss, k=4, method='pamonce')
    # print(clustering)
    # print(diss)
    end_time = time.time()

    runtime = end_time - start_time
    runtimes.append(runtime)
    filenames.append(filename)

    print(f"File: {filename}, Runtime: {runtime:.4f} seconds")

print(runtimes)

# 绘制折线图
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=filenames, y=runtimes, marker='o')
# plt.xticks(rotation=45, ha='right')  # 旋转 x 轴标签以避免重叠
# plt.xlabel("CSV Filename")
# plt.ylabel("Runtime (seconds)")
# plt.title("Runtime vs CSV Filename")
# plt.tight_layout()
# plt.show()