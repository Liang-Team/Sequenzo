import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_comparison(data_dict, x_axis, x_label, y_label='Time (seconds)',
                    colors=None, legend_loc='upper left', log_scale=True,
                    title=None, save_path=None):
    if colors is None:
        colors = ['#ffadbb', '#acbfeb', '#bfe7d6']

    df = pd.DataFrame(data_dict, index=x_axis)

    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid")

    x_pos = range(len(x_axis))

    for (label, series), color in zip(df.items(), colors):
        linestyle = '--' if 'before' in label else '-'
        plt.plot(x_pos, series, marker='o', label=label,
                 color=color, linewidth=2.5, markersize=8, linestyle=linestyle)

    plt.xlabel(x_label, fontsize=14, labelpad=15)
    plt.ylabel(y_label, fontsize=14, labelpad=15)
    plt.xticks(x_pos, x_axis, fontsize=12)
    plt.yticks(fontsize=12)

    if log_scale:
        plt.yscale('log')

    if title:
        plt.title(title, fontsize=15, pad=15)

    plt.legend(title_fontsize=12, fontsize=11, loc=legend_loc)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ==========================================
# Experiment 1: Sequence Length (n=10,000, U=85)
# 3-run average
# ==========================================
sequenzo_old_exp1 = [6.74, 64.06, 167.12, 783.81]
sequenzo_new_exp1 = [1.69, 8.78, 24.55, 107.14]
tanat_exp1        = [20.37, 42.44, 115.54, 510.00]

plot_comparison(
    data_dict={'Sequenzo (before)': sequenzo_old_exp1, 'Sequenzo (after)': sequenzo_new_exp1, 'TanaT': tanat_exp1},
    x_axis=['10', '30', '50', '100'],
    x_label='Sequence Length',
    title='Exp 1: Effect of Sequence Length on OM Computation Time',
    save_path='om_exp1_sequence_length.png'
)

df_exp1 = pd.DataFrame({
    'Sequence Length': [10, 30, 50, 100],
    'Sequenzo Before (s)': sequenzo_old_exp1,
    'Sequenzo After (s)': sequenzo_new_exp1,
    'TanaT (s)': tanat_exp1,
    'Speedup': [round(old/new, 1) for old, new in zip(sequenzo_old_exp1, sequenzo_new_exp1)],
    'vs TanaT': [round(t/new, 1) for t, new in zip(tanat_exp1, sequenzo_new_exp1)],
})
print("Experiment 1: Sequence Length")
print(df_exp1.to_string(index=False))
print()


# ==========================================
# Experiment 2: Uniqueness Rate (n=10,000, L=30)
# 3-run average
# ==========================================
sequenzo_old_exp2 = [3.79, 18.09, 64.06, 75.14]
sequenzo_new_exp2 = [1.49, 3.77, 8.78, 12.40]
tanat_exp2        = [36.49, 40.53, 42.44, 40.36]

plot_comparison(
    data_dict={'Sequenzo (before)': sequenzo_old_exp2, 'Sequenzo (after)': sequenzo_new_exp2, 'TanaT': tanat_exp2},
    x_axis=['20', '50', '85', '100'],
    x_label='Uniqueness Rate (%)',
    title='Exp 2: Effect of Uniqueness Rate on OM Computation Time',
    save_path='om_exp2_uniqueness_rate.png'
)

df_exp2 = pd.DataFrame({
    'Uniqueness Rate (%)': [20, 50, 85, 100],
    'Sequenzo Before (s)': sequenzo_old_exp2,
    'Sequenzo After (s)': sequenzo_new_exp2,
    'TanaT (s)': tanat_exp2,
    'Speedup': [round(old/new, 1) for old, new in zip(sequenzo_old_exp2, sequenzo_new_exp2)],
    'vs TanaT': [round(t/new, 1) for t, new in zip(tanat_exp2, sequenzo_new_exp2)],
})
print("Experiment 2: Uniqueness Rate")
print(df_exp2.to_string(index=False))
print()


# ==========================================
# Experiment 3: Sample Size (L=30, U=85)
# 3-run average
# ==========================================
sequenzo_old_exp3 = [0.59, 2.16, 4.41, 11.10, 64.06]
sequenzo_new_exp3 = [0.15, 0.45, 1.00, 1.73, 8.78]
tanat_exp3        = [1.46, 2.43, 4.56, 7.34, 42.44]

plot_comparison(
    data_dict={'Sequenzo (before)': sequenzo_old_exp3, 'Sequenzo (after)': sequenzo_new_exp3, 'TanaT': tanat_exp3},
    x_axis=['1', '2', '3', '4', '10'],
    x_label='Sample Size (1,000)',
    title='Exp 3: Effect of Sample Size on OM Computation Time',
    save_path='om_exp3_sample_size.png'
)

df_exp3 = pd.DataFrame({
    'Sample Size': [1000, 2000, 3000, 4000, 10000],
    'Sequenzo Before (s)': sequenzo_old_exp3,
    'Sequenzo After (s)': sequenzo_new_exp3,
    'TanaT (s)': tanat_exp3,
    'Speedup': [round(old/new, 1) for old, new in zip(sequenzo_old_exp3, sequenzo_new_exp3)],
    'vs TanaT': [round(t/new, 1) for t, new in zip(tanat_exp3, sequenzo_new_exp3)],
})
print("Experiment 3: Sample Size")
print(df_exp3.to_string(index=False))
print()


# ==========================================
# Combined 3-panel figure
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.set(style="whitegrid")

colors = ['#ffadbb', '#acbfeb', '#bfe7d6']

datasets = [
    {
        'data': {'Sequenzo (before)': sequenzo_old_exp1, 'Sequenzo (after)': sequenzo_new_exp1, 'TanaT': tanat_exp1},
        'x': ['10', '30', '50', '100'],
        'xlabel': 'Sequence Length',
        'title': 'Exp 1: Sequence Length\n(n=10,000, U=85)',
    },
    {
        'data': {'Sequenzo (before)': sequenzo_old_exp2, 'Sequenzo (after)': sequenzo_new_exp2, 'TanaT': tanat_exp2},
        'x': ['20', '50', '85', '100'],
        'xlabel': 'Uniqueness Rate (%)',
        'title': 'Exp 2: Uniqueness Rate\n(n=10,000, L=30)',
    },
    {
        'data': {'Sequenzo (before)': sequenzo_old_exp3, 'Sequenzo (after)': sequenzo_new_exp3, 'TanaT': tanat_exp3},
        'x': ['1', '2', '3', '4', '10'],
        'xlabel': 'Sample Size (1,000)',
        'title': 'Exp 3: Sample Size\n(L=30, U=85)',
    },
]

for ax, ds in zip(axes, datasets):
    x_pos = range(len(ds['x']))
    for (label, series), color in zip(ds['data'].items(), colors):
        linestyle = '--' if 'before' in label else '-'
        ax.plot(x_pos, series, marker='o', label=label,
                color=color, linewidth=2.5, markersize=7, linestyle=linestyle)
    ax.set_yscale('log')
    ax.set_xlabel(ds['xlabel'], fontsize=13, labelpad=10)
    ax.set_ylabel('Time (s)', fontsize=13, labelpad=10)
    ax.set_title(ds['title'], fontsize=13, pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ds['x'], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('om_combined.png', dpi=300, bbox_inches='tight')
plt.show()
