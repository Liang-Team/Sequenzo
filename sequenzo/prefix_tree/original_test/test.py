"""
@Author  : Yuqi Liang 梁彧祺
@File    : test.py
@Time    : 02/05/2025 12:27
@Desc    : 
"""
import pandas as pd
from system_level_indicators import build_prefix_tree, compute_prefix_count, compute_branching_factor, compute_js_divergence, compute_composite_score
from individual_level_indicators import IndividualDivergence
from utils import extract_sequences, get_state_space

# ------------------------
# Step 1: 读取和准备数据
# ------------------------
df = pd.read_csv("your_data.csv")
time_cols = [f"C{i}" for i in range(1, 11)]  # C1~C10
sequences = extract_sequences(df, time_cols)
states = get_state_space(sequences)

# ------------------------
# Step 2: 构建前缀树
# ------------------------
tree = build_prefix_tree(sequences)

print(tree.describe())

# ------------------------
# Step 3: 计算系统层指标
# ------------------------
T = len(time_cols)
prefix_counts = compute_prefix_count(tree, T)
branching_factors = compute_branching_factor(tree, T)
js_scores = compute_js_divergence(sequences, state_set=states)
composite_score = compute_composite_score(prefix_counts, branching_factors, js_scores)

# ------------------------
# Step 4: 计算个体层指标
# ------------------------
divergence = IndividualDivergence(sequences)
df["diverged"] = divergence.compute_diverged()
df["divergence_year"] = divergence.compute_divergence_year()
df["prefix_rarity_score"] = divergence.compute_prefix_rarity_score()
df["path_uniqueness"] = divergence.compute_path_uniqueness()

# ------------------------
# Step 5: 输出或保存结果
# ------------------------
df.to_csv("trajectory_with_indicators.csv", index=False)

# 示例：打印前5个个体的个体指标
print(df[["worker_id", "diverged", "divergence_year", "prefix_rarity_score", "path_uniqueness"]].head())

# 示例：打印系统层 composite score 趋势
import matplotlib.pyplot as plt

plt.plot(range(1, T+1), composite_score, marker='o')
plt.title("Composite Structural Divergence Score over Time")
plt.xlabel("Year")
plt.ylabel("Composite Score")
plt.grid(True)
plt.show()
