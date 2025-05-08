"""
@Author  : Yuqi Liang 梁彧祺
@File    : markov_chains.py
@Time    : 07/05/2025 11:16
@Desc    : 
"""
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

df = pd.read_csv("/Users/lei/Documents/Sequenzo_all_folders/sequenzo_local/test_data/real_data_my_paper/detailed_sequence_10_work_years_df.csv")

time_cols = [f"C{i}" for i in range(1, 11)]  # C1~C10

# 找到所有以 "C" 开头的列，并将其值转换为字符串
career_columns = [col for col in df.columns if col.startswith("C")]
df[career_columns] = df[career_columns].astype(str)

# 假设 'country' 是一列，值为 'India' 或 'US'
india_df = df[df["country"] == "india"].copy()
us_df = df[df["country"] == "us"].copy()

# 计算JS散度
def js_divergence(P, Q):
    P = np.array(P) + 1e-10
    Q = np.array(Q) + 1e-10
    M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, M) + entropy(Q, M))

# 计算转移概率矩阵
def transition_matrix(data):
    states = sorted(set(map(str, data.flatten())))
    n = len(states)
    matrix = np.zeros((n, n))
    state_to_idx = {state: idx for idx, state in enumerate(states)}

    for sequence in data:
        for (i, j) in zip(sequence[:-1], sequence[1:]):
            matrix[state_to_idx[i], state_to_idx[j]] += 1

    # 归一化为转移概率
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, where=row_sums != 0)
    return matrix, states

# 绘制转移概率矩阵热力图
def plot_transition_matrix(matrix, states, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=states, yticklabels=states)
    plt.title(title)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()

# 提取前3年和后3年的职业序列
india_early = india_df.iloc[:, 1:4].to_numpy()
india_late = india_df.iloc[:, -3:].to_numpy()

us_early = us_df.iloc[:, 1:4].to_numpy()
us_late = us_df.iloc[:, -3:].to_numpy()

# 计算转移概率矩阵
india_early_matrix, india_states = transition_matrix(india_early)
india_late_matrix, _ = transition_matrix(india_late)

us_early_matrix, us_states = transition_matrix(us_early)
us_late_matrix, _ = transition_matrix(us_late)

# 可视化转移概率矩阵
plot_transition_matrix(india_early_matrix, india_states, "India Early Transition Matrix")
plot_transition_matrix(india_late_matrix, india_states, "India Late Transition Matrix")
plot_transition_matrix(us_early_matrix, us_states, "US Early Transition Matrix")
plot_transition_matrix(us_late_matrix, us_states, "US Late Transition Matrix")

# 计算早期和晚期转移矩阵的JS散度
india_js = js_divergence(india_early_matrix.flatten(), india_late_matrix.flatten())
us_js = js_divergence(us_early_matrix.flatten(), us_late_matrix.flatten())

print(f"India JS Divergence (Early vs Late): {india_js:.4f}")
print(f"US JS Divergence (Early vs Late): {us_js:.4f}")

# 逐年计算JS散度，检测分化节点
def js_divergence_series(data):
    js_values = []
    for i in range(1, data.shape[1] - 1):
        early_matrix, _ = transition_matrix(data[:, i-1:i+1])
        late_matrix, _ = transition_matrix(data[:, i:i+2])
        js_values.append(js_divergence(early_matrix.flatten(), late_matrix.flatten()))
    return js_values

# 计算JS散度时间序列
india_js_series = js_divergence_series(india_df.iloc[:, 1:].to_numpy())
us_js_series = js_divergence_series(us_df.iloc[:, 1:].to_numpy())

# JS散度时间序列可视化
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(india_js_series) + 1), india_js_series, label='India', marker='o', color='blue')
plt.plot(range(1, len(us_js_series) + 1), us_js_series, label='US', marker='x', color='red')
plt.title("JS Divergence Over Time (India vs US)")
plt.xlabel("Year")
plt.ylabel("JS Divergence")
plt.legend()
plt.show()
