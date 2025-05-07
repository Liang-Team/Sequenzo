"""
@Author  : 李欣怡
@File    : test.py
@Time    : 2025/5/6 20:30
@Desc    : 
"""
import numpy as np
import pandas as pd

df = pd.read_csv("D:/college/research/QiQi/sequenzo/files/orignal data/sohee/my_sequence_data_with_7.csv")

# df.fillna('', inplace=True)
# df.replace(1, "A", inplace=True)
# df.replace(2, "B", inplace=True)
# df.replace(3, "C", inplace=True)
# df.replace(4, "D", inplace=True)
# df.replace(5, "E", inplace=True)
# df.replace(6, "F", inplace=True)
df.replace(7, '', inplace=True)

df.to_csv("D:/college/research/QiQi/sequenzo/files/orignal data/sohee/my_sequence_data_with_blank.csv", index=False)