"""
@Author  : 李欣怡
@File    : clara_test.py
@Time    : 2025/9/17 10:37
@Desc    : 
"""
import time
import psutil
import os
import pandas as pd
from sequenzo import *

df = pd.read_csv("D:/college/research/QiQi/sequenzo/data_and_output/orignal data/detailed_expanded_80w.csv")

df = df.sample(n=1000, random_state=42)

sequence_data = SequenceData(
    df,
    time=list(df.columns)[9:],
    time_type="age",
    states=["Data", "Data science", "Hardware", "Research",
            "Software", "Support & test", "Systems & infrastructure"],
    id_col="worker_id"
)

print("="*60)
print("现在开始测试 get_distance_matrix")

# 获取当前进程
process = psutil.Process(os.getpid())

# 内存监控：计算前
mem_before = process.memory_info().rss / 1024**2  # MB
print(f"计算前内存使用: {mem_before:.2f} MB")

start_time = time.time()

result = clara(sequence_data,
                   R=250,
                   sample_size=3000,
                   kvals=range(2, 21),
                   criteria=['distance'],
                   dist_args={"method": "OM", "sm": "CONSTANT", "indel": 1},
                   parallel=True,
                   stability=True)

end_time = time.time()

# 内存监控：计算后
mem_after = process.memory_info().rss / 1024**2  # MB

print(f"计算后内存使用: {mem_after:.2f} MB")
print(f"计算过程中最大内存增加: {mem_after - mem_before:.2f} MB")
print(f"OM距离矩阵计算完成，用时: {end_time - start_time:.2f} 秒")
