"""
@Author  : 李欣怡
@File    : tool.py
@Time    : 2025/6/10 21:45
@Desc    : 
"""
import pandas as pd

df = pd.read_csv("D:/college/research/QiQi/sequenzo/files/orignal data/detailed_sequence_10_work_years_df.csv")

df_transposed = df.T

df_transposed.index = df.columns
df_transposed.columns = df_transposed.iloc[0]  # 将第一行设为新列名
df_transposed = df_transposed[1:]              # 删除第一行

# 保存为新的 CSV 文件
df_transposed.to_csv('D:/college/research/QiQi/sequenzo/files/detialed_transposed.csv', index=True)