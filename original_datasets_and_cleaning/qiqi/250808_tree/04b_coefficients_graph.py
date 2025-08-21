import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: 整理你的 marginal effects 数据 (这里用 US 为例) ===
# 你可以直接把报告里复制过来
us_marginal = {
    "Junior": 0.000214,
    "Assistant": 0.000242,
    "Regular": 0.004695,
    "Senior": -0.003148,
    "Leader": -0.001741,
    "Chief or founder": -0.000262
}
india_marginal = {
    "Junior": -0.003170,
    "Assistant": -0.000858,
    "Regular": -0.020712,
    "Senior": 0.013243,
    "Leader": 0.011255,
    "Chief or founder": 0.000242
}

# === Step 2: 转换成 DataFrame ===
df = pd.DataFrame([
    {"Country": "US", "Level": k, "AME": v} for k, v in us_marginal.items()
] + [
    {"Country": "India", "Level": k, "AME": v} for k, v in india_marginal.items()
])

# 保持排序
level_order = ["Junior", "Assistant", "Regular", "Senior", "Leader", "Chief or founder"]
hue_order = ["US", "India"]

# === Step 3: 设置调色盘（与 01_tree_tests.ipynb 保持一致） ===
# US 与 India 对应色：US=#A3BFD9，India=#E8B88A
palette = {"US": "#A3BFD9", "India": "#E8B88A"}

sns.set(style="whitegrid", font_scale=1.2)

# === Step 4: 画图 ===
# 排序，确保与绘图顺序一致
df["Level"] = pd.Categorical(df["Level"], categories=level_order, ordered=True)
df["Country"] = pd.Categorical(df["Country"], categories=hue_order, ordered=True)
df = df.sort_values(["Level", "Country"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 6))
# 绘制（与模板一致）
ax = sns.barplot(
    data=df, x="Level", y="AME", hue="Country",
    palette=palette, order=level_order, hue_order=hue_order, ax=ax
)

# 与模板风格一致
ax.grid(axis="x", visible=False)
sns.despine()

# === Step 5: 添加 significance stars (根据你报告里的 p 值) ===
# US rarity: p < 0.001 → ***
# India rarity: p < 0.001 → ***
stars = {
    "US": ["", "", "***", "***", "***", ""],
    "India": ["***", "", "***", "***", "***", ""]
}
# 依据国家与等级为每一行生成 star
level_to_idx = {lvl: idx for idx, lvl in enumerate(level_order)}
df["star"] = df.apply(lambda r: stars[str(r["Country"])][level_to_idx[str(r["Level"])]]
                      if str(r["Country"]) in stars else "", axis=1)

# 为每个柱子设置柔和描边，并添加 significance stars（与 ax.patches 顺序一一对应）
# 统一星号与柱体的间距（基于数据坐标，正负一致）
ymin, ymax = ax.get_ylim()
y_range = ymax - ymin if ymax > ymin else 1.0
gap = 0.02 * y_range  # 固定为纵轴范围的 2%

for patch, (_, row) in zip(ax.patches, df.iterrows()):
    # 去掉描边，保持更优雅
    patch.set_linewidth(0)
    patch.set_edgecolor('none')

    # 添加星号（在柱子“外侧”，正数在上方，负数在下方）
    star = row.get("star", "")
    if star:
        x = patch.get_x() + patch.get_width()/2
        bbox = patch.get_bbox()
        height = patch.get_height()
        if height >= 0:
            y = bbox.ymax
            va = "bottom"
            yoff = 2
        else:
            y = bbox.ymin
            va = "top"
            yoff = -2
        ax.annotate(
            star, (x, y),
            ha="center", va=va,
            fontsize=12, color="black",
            xytext=(0, yoff), textcoords="offset points",
            clip_on=False
        )
# === Step 6: 美化 ===
plt.ylabel("Average Marginal Effect (per +1 SD rarity)")
plt.xlabel("Seniority Level")
plt.title("Marginal Effects of Rarity on Seniority (US vs India)")
plt.legend(title="Country")
plt.tight_layout()
# 保存并关闭，避免交互阻塞
out_path = "/Users/lei/Documents/Sequenzo_all_folders/Sequenzo-main/original_datasets_and_cleaning/qiqi/250808_tree/coefficients_graph.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {out_path}")
plt.close()
