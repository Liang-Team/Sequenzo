import time
import psutil
import os
import pandas as pd
from sequenzo import *

# 读取扩展数据
df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/orignal data/detailed_expanded_80w.csv")

df = df.sample(n=1000, random_state=42)

# 构建 SequenceData
sequence_data = SequenceData(
    df,
    time=list(df.columns)[9:],
    time_type="age",
    states=["Data", "Data science", "Hardware", "Research",
            "Software", "Support & test", "Systems & infrastructure"],
    id_col="worker_id"
)

# plot_sequence_index(sequence_data, save_as='sequence_index_80w.png')

print("="*60)
print("现在开始测试 get_distance_matrix")

# 获取当前进程
process = psutil.Process(os.getpid())

# 内存监控：计算前
mem_before = process.memory_info().rss / 1024**2  # MB
print(f"计算前内存使用: {mem_before:.2f} MB")

start_time = time.time()

om = get_distance_matrix(sequence_data, method="OM", sm="CONSTANT", indel=1)

end_time = time.time()

# 内存监控：计算后
mem_after = process.memory_info().rss / 1024**2  # MB

print(f"计算后内存使用: {mem_after:.2f} MB")
print(f"计算过程中最大内存增加: {mem_after - mem_before:.2f} MB")
print(f"OM距离矩阵计算完成，用时: {end_time - start_time:.2f} 秒")


print("="*60)
print("现在开始测试 Cluster")

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024**2  # MB
print(f"计算前内存使用: {mem_before:.2f} MB")

start_time = time.time()

cluster = Cluster(om, sequence_data.ids, clustering_method='ward')

end_time = time.time()
mem_after = process.memory_info().rss / 1024**2  # MB

print(f"计算后内存使用: {mem_after:.2f} MB")
print(f"计算过程中最大内存增加: {mem_after - mem_before:.2f} MB")
print(f"OM距离矩阵计算完成，用时: {end_time - start_time:.2f} 秒")