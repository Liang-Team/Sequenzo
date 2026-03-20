import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================================
# Data (3-run averages)
# ==================================================================

# --- Experiment 1: Sequence Length (n=10000, U=85) ---
# LCP
sqz_old_lcp_exp1 = [0.61, 0.68, 0.58, 0.87]
sqz_new_lcp_exp1 = [0.56, 0.38, 0.43, 0.46]
tat_lcp_exp1     = [14.62, 12.18, 16.14, 16.58]
# LCS
sqz_old_lcs_exp1 = [2.13, 23.59, 61.35, 258.78]
sqz_new_lcs_exp1 = [1.17, 6.16, 16.15, 58.42]
tat_lcs_exp1     = [13.00, 53.81, 62.69, 300.19]
# EUCLID
sqz_old_euc_exp1 = [14.62, 43.37, 71.58, 139.18]
sqz_new_euc_exp1 = [1.02, 3.18, 5.18, 12.07]
tat_euc_exp1     = [178.86, 480.17, 1302.66, 2027.29]

# --- Experiment 2: Uniqueness Rate (n=10000, L=30) ---
# LCP
sqz_old_lcp_exp2 = [0.58, 1.40, 0.68, 0.76]
sqz_new_lcp_exp2 = [0.30, 0.33, 0.38, 0.39]
tat_lcp_exp2     = [14.13, 15.52, 12.18, 15.51]
# LCS
sqz_old_lcs_exp2 = [1.07, 17.33, 23.59, 34.52]
sqz_new_lcs_exp2 = [0.65, 2.34, 6.16, 8.43]
tat_lcs_exp2     = [56.74, 19.50, 53.81, 22.39]
# EUCLID
sqz_old_euc_exp2 = [42.83, 72.71, 43.37, 45.78]
sqz_new_euc_exp2 = [2.98, 2.95, 3.18, 3.06]
tat_euc_exp2     = [496.29, 619.68, 480.17, 605.94]

# --- Experiment 3: Sample Size (L=30, U=85) ---
# LCP
sqz_old_lcp_exp3 = [0.01, 0.03, 0.09, 0.19, 0.68]
sqz_new_lcp_exp3 = [0.007, 0.016, 0.032, 0.071, 0.38]
tat_lcp_exp3     = [1.97, 2.13, 3.28, 4.19, 12.18]
# LCS
sqz_old_lcs_exp3 = [0.12, 0.78, 2.44, 5.70, 23.59]
sqz_new_lcs_exp3 = [0.058, 0.231, 0.540, 0.985, 6.16]
tat_lcs_exp3     = [1.39, 3.55, 9.38, 20.50, 53.81]
# EUCLID
sqz_old_euc_exp3 = [0.29, 2.10, 5.16, 14.66, 43.37]
sqz_new_euc_exp3 = [0.026, 0.080, 0.190, 0.376, 3.18]
tat_euc_exp3     = [2.78, 11.95, 31.26, 58.18, 480.17]

x_exp1 = ['L=10', 'L=30', 'L=50', 'L=100']
x_exp2 = ['U=20', 'U=50', 'U=85', 'U=100']
x_exp3 = ['n=1k', 'n=2k', 'n=3k', 'n=4k', 'n=10k']


# ==================================================================
# Combined panels: one figure per metric (3 experiments side by side)
# with before/after/TanaT
# ==================================================================
def plot_combined(metric_name, datasets_list, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set(style="whitegrid")
    colors = ['#ffadbb', '#acbfeb', '#bfe7d6']

    for ax, ds in zip(axes, datasets_list):
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# --- LCP ---
plot_combined('LCP', [
    {'data': {'Sequenzo (before)': sqz_old_lcp_exp1, 'Sequenzo (after)': sqz_new_lcp_exp1, 'TanaT': tat_lcp_exp1},
     'x': ['10', '30', '50', '100'], 'xlabel': 'Sequence Length',
     'title': 'LCP: Sequence Length\n(n=10k, U=85)'},
    {'data': {'Sequenzo (before)': sqz_old_lcp_exp2, 'Sequenzo (after)': sqz_new_lcp_exp2, 'TanaT': tat_lcp_exp2},
     'x': ['20', '50', '85', '100'], 'xlabel': 'Uniqueness Rate (%)',
     'title': 'LCP: Uniqueness Rate\n(n=10k, L=30)'},
    {'data': {'Sequenzo (before)': sqz_old_lcp_exp3, 'Sequenzo (after)': sqz_new_lcp_exp3, 'TanaT': tat_lcp_exp3},
     'x': ['1k', '2k', '3k', '4k', '10k'], 'xlabel': 'Sample Size',
     'title': 'LCP: Sample Size\n(L=30, U=85)'},
], 'lcp_combined.png')

# --- LCS ---
plot_combined('LCS', [
    {'data': {'Sequenzo (before)': sqz_old_lcs_exp1, 'Sequenzo (after)': sqz_new_lcs_exp1, 'TanaT': tat_lcs_exp1},
     'x': ['10', '30', '50', '100'], 'xlabel': 'Sequence Length',
     'title': 'LCS: Sequence Length\n(n=10k, U=85)'},
    {'data': {'Sequenzo (before)': sqz_old_lcs_exp2, 'Sequenzo (after)': sqz_new_lcs_exp2, 'TanaT': tat_lcs_exp2},
     'x': ['20', '50', '85', '100'], 'xlabel': 'Uniqueness Rate (%)',
     'title': 'LCS: Uniqueness Rate\n(n=10k, L=30)'},
    {'data': {'Sequenzo (before)': sqz_old_lcs_exp3, 'Sequenzo (after)': sqz_new_lcs_exp3, 'TanaT': tat_lcs_exp3},
     'x': ['1k', '2k', '3k', '4k', '10k'], 'xlabel': 'Sample Size',
     'title': 'LCS: Sample Size\n(L=30, U=85)'},
], 'lcs_combined.png')

# --- EUCLID ---
plot_combined('EUCLID', [
    {'data': {'Sequenzo (before)': sqz_old_euc_exp1, 'Sequenzo (after)': sqz_new_euc_exp1, 'TanaT': tat_euc_exp1},
     'x': ['10', '30', '50', '100'], 'xlabel': 'Sequence Length',
     'title': 'EUCLID: Sequence Length\n(n=10k, U=85)'},
    {'data': {'Sequenzo (before)': sqz_old_euc_exp2, 'Sequenzo (after)': sqz_new_euc_exp2, 'TanaT': tat_euc_exp2},
     'x': ['20', '50', '85', '100'], 'xlabel': 'Uniqueness Rate (%)',
     'title': 'EUCLID: Uniqueness Rate\n(n=10k, L=30)'},
    {'data': {'Sequenzo (before)': sqz_old_euc_exp3, 'Sequenzo (after)': sqz_new_euc_exp3, 'TanaT': tat_euc_exp3},
     'x': ['1k', '2k', '3k', '4k', '10k'], 'xlabel': 'Sample Size',
     'title': 'EUCLID: Sample Size\n(L=30, U=85)'},
], 'euclid_combined.png')


# ==================================================================
# Summary tables
# ==================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLES (3-run averages)")
print("=" * 70)

def print_table(metric, old_data, new_data, tat_data, params, param_name):
    df = pd.DataFrame({
        param_name: params,
        'Sqz Before (s)': old_data,
        'Sqz After (s)': new_data,
        'Speedup': [round(o/n, 1) for o, n in zip(old_data, new_data)],
        'TanaT (s)': tat_data,
        'vs TanaT': [f"{round(t/n, 1)}x" for t, n in zip(tat_data, new_data)],
    })
    print(df.to_string(index=False))
    print()

for metric, datasets in [
    ('LCP', [
        (sqz_old_lcp_exp1, sqz_new_lcp_exp1, tat_lcp_exp1, [10,30,50,100], 'Seq Length'),
        (sqz_old_lcp_exp2, sqz_new_lcp_exp2, tat_lcp_exp2, [20,50,85,100], 'U (%)'),
        (sqz_old_lcp_exp3, sqz_new_lcp_exp3, tat_lcp_exp3, ['1k','2k','3k','4k','10k'], 'n'),
    ]),
    ('LCS', [
        (sqz_old_lcs_exp1, sqz_new_lcs_exp1, tat_lcs_exp1, [10,30,50,100], 'Seq Length'),
        (sqz_old_lcs_exp2, sqz_new_lcs_exp2, tat_lcs_exp2, [20,50,85,100], 'U (%)'),
        (sqz_old_lcs_exp3, sqz_new_lcs_exp3, tat_lcs_exp3, ['1k','2k','3k','4k','10k'], 'n'),
    ]),
    ('EUCLID', [
        (sqz_old_euc_exp1, sqz_new_euc_exp1, tat_euc_exp1, [10,30,50,100], 'Seq Length'),
        (sqz_old_euc_exp2, sqz_new_euc_exp2, tat_euc_exp2, [20,50,85,100], 'U (%)'),
        (sqz_old_euc_exp3, sqz_new_euc_exp3, tat_euc_exp3, ['1k','2k','3k','4k','10k'], 'n'),
    ]),
]:
    print(f"\n--- {metric} ---")
    for old, new, tat, params, pname in datasets:
        print_table(metric, old, new, tat, params, pname)
