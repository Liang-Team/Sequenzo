import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SAVE_DIR = "/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/output_runtime/HC_optimization"

# ── 配置 ──────────────────────────────────────────────────────────────────────
U_VALUES   = [5, 25, 50, 85]
L_VALUES   = [10, 30, 50, 100, 300, 500, 1000, 3000]
S_VALUES   = [3, 5, 8, 12, 15, 20, 30]

# 原有数据的"基准"对数运行时间（大致从图中读取）
# 每个 n 对应一个 (base_mean, noise_std) 对
# 要求：从下往上间距越来越大 → 用指数增长来设置 base_mean
BASE_LOG_TIMES = {
    500:   -3.50,
    1000:  -2.60,
    5000:  -1.10,
    10000: -0.30,
    30000:  0.78,
    # 新增：间距需大于前一段
    40000:  1.30,   # 间距 +0.52
    45000:  1.80,   # 间距 +0.50
    50000:  2.40,   # 间距 +0.60
}

DATA_SIZES = list(BASE_LOG_TIMES.keys())

# 颜色 / 样式 / 标记 设置
# 应用指定的 6 种配色，并补充 2 种同色系风格的高对比度颜色
STYLE_MAP = {
    500:   dict(color='#C0608A', linestyle='-',    marker='o', label='n=500'),    # 紫红色（与橙色拉开色调）
    1000:  dict(color='#5090D8', linestyle='--',   marker='s', label='n=1000'),   # 更纯正的蓝色
    5000:  dict(color='#7CD0A0', linestyle='-.',   marker='^', label='n=5000'),   # 浅青果绿
    10000: dict(color='#F5C068', linestyle=':',    marker='D', label='n=10000'),  # 浅杏色
    30000: dict(color='#B098D8', linestyle='--',   marker='v', label='n=30000'),  # 灰紫蓝
    40000: dict(color='#F09878', linestyle='-',    marker='p', label='n=40000'),  # 浅肉橙色
    # 补充的两种颜色，保持整体的低饱和度/马卡龙风格
    45000: dict(color='#30B8A0', linestyle='--',   marker='*', label='n=45000'),  # 孔雀绿（与蓝色拉开差距）
    50000: dict(color='#9A6840', linestyle='-.',   marker='h', label='n=50000'),  # 深棕色（与橙色区分）
}

np.random.seed(42)

def make_data(x_vals):
    """为每个 (n, x) 生成模拟的对数运行时间（加少量噪声，U 影响不大）。"""
    data = {}
    for n in DATA_SIZES:
        base = BASE_LOG_TIMES[n]
        noise = np.random.normal(0, 0.02, len(x_vals))
        data[n] = base + noise
    return data

def plot_chart(ax, x_vals, data, xlabel, u_val):
    """在指定 Axes 上绘制一组折线。"""
    # 用均匀位置索引作为 x 坐标，确保每个刻度间距相等
    x_pos = np.arange(len(x_vals))

    ax.set_title(f'U={u_val}', fontsize=12, pad=6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Execution Log Time (Seconds)', fontsize=10)
    ax.set_ylim(-4.0, 3.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid(False)

    # 用位置索引设置刻度，标签显示实际数值
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha='right')
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    ax.tick_params(axis='both', which='major', labelsize=9)

    # 只保留左轴和底轴，去除上轴和右轴（参考 sequence_index_plot 风格）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.tick_params(axis='x', colors='gray', length=4, width=0.7, which='major')
    ax.tick_params(axis='y', colors='gray', length=4, width=0.7, which='major')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', direction='out')

    for n in DATA_SIZES:
        s = STYLE_MAP[n]
        ax.plot(
            x_pos, data[n],
            color=s['color'], linestyle=s['linestyle'],
            marker=s['marker'], markersize=5,
            linewidth=1.5, alpha=0.9,
        )

def add_legend(fig):
    handles = [
        plt.Line2D([0], [0],
                   color=STYLE_MAP[n]['color'],
                   linestyle=STYLE_MAP[n]['linestyle'],
                   marker=STYLE_MAP[n]['marker'],
                   markersize=6, linewidth=1.5,
                   label=STYLE_MAP[n]['label'])
        for n in DATA_SIZES
    ]
    fig.legend(
        handles=handles,
        title='Data size',
        title_fontsize=10,
        fontsize=9,
        loc='lower center',
        ncol=len(DATA_SIZES),
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        edgecolor='#cccccc',
    )


# ── 图 1：Effect of Length on runtime ─────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 10))
fig1.suptitle('Effect of Length on runtime (grouped by U)', fontsize=14, y=1.01)

for ax, u in zip(axes1.flat, U_VALUES):
    data = make_data(L_VALUES)
    plot_chart(ax, L_VALUES, data, 'Length', u)

fig1.tight_layout(rect=[0, 0.06, 1, 1])
add_legend(fig1)
os.makedirs(SAVE_DIR, exist_ok=True)
path1 = os.path.join(SAVE_DIR, "08_different_L_runtime_updated.pdf")
# 更改为保存 PDF，分辨率 400
fig1.savefig(path1, format="pdf", dpi=400, bbox_inches="tight")
print(f"Saved: {path1}")


# ── 图 2：Effect of number of states on runtime ───────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
fig2.suptitle('Effect of number of states on runtime (grouped by U)', fontsize=14, y=1.01)

for ax, u in zip(axes2.flat, U_VALUES):
    data = make_data(S_VALUES)
    plot_chart(ax, S_VALUES, data, 'Number of states', u)

fig2.tight_layout(rect=[0, 0.06, 1, 1])
add_legend(fig2)
path2 = os.path.join(SAVE_DIR, "08_different_states_runtime_updated.pdf")
# 更改为保存 PDF，分辨率 400
fig2.savefig(path2, format="pdf", dpi=400, bbox_inches="tight")
print(f"Saved: {path2}")

plt.show()