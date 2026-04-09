"""
Extended Benchmark Visualization: Sequenzo vs TanaT
Includes: Speedup Ratio and Relative Performance Improvement

Usage: python plot_extended_benchmarks.py
Run from: tests/archive_test/yapeng_test/plot_runtime/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ======================================================================
# DATA: All times in seconds
# ======================================================================

# ---- Exp1: Sequence Length (n=10,000, U=85) ----
exp1_x = [10, 30, 50, 100, 200, 500]

exp1 = {
    'OM': {
        'sqz':   [1.69, 8.78, 24.55, 107.14, 470.00, 3044.08],
        'tanat': [20.37, 42.44, 115.54, 510.00, 2052.45, 15039.10],
    },
    'LCS': {
        'sqz':   [1.17, 6.16, 16.15, 58.42, 233.90, 1111.15],
        'tanat': [13.00, 53.81, 62.69, 300.19, 1132.53, None],
    },
    'LCP': {
        'sqz':   [0.56, 0.38, 0.43, 0.46, 0.48, 0.57],
        'tanat': [14.62, 12.18, 16.14, 16.58, 16.76, None],
    },
    'EUCLID': {
        'sqz':   [1.02, 3.18, 5.18, 12.07, 36.00, 60.80],
        'tanat': [178.86, 480.17, 1302.66, 2027.29, 3896.40, None],
    },
}

# ---- Exp2: Uniqueness Rate (n=10,000, L=30) ----
exp2_x = [1, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98, 99, 100]

exp2 = {
    'OM': {
        'sqz':   [0.97, 1.49, 1.91, 2.59, 3.77, 4.42, 5.62, 7.31, 8.78, 8.96, 10.00, 10.13, 10.20, 12.40],
        'tanat': [28.38, 36.49, 34.02, 34.16, 40.53, 33.79, 33.70, 36.53, 42.44, 36.15, 36.98, 36.40, 33.62, 40.36],
    },
    'LCS': {
        'sqz':   [0.32, 0.65, 1.03, 1.61, 2.34, 3.14, 4.06, 5.53, 6.16, 7.10, 7.64, 7.60, 7.62, 8.43],
        'tanat': [13.67, 56.74, 22.03, 17.15, 19.50, 17.93, 18.05, 26.50, 53.81, 22.18, 22.29, 22.12, 20.61, 22.39],
    },
    'LCP': {
        'sqz':   [0.33, 0.30, 0.35, 0.34, 0.33, 0.36, 0.36, 0.40, 0.38, 0.47, 0.44, 0.41, 0.40, 0.39],
        'tanat': [15.55, 14.13, 16.20, 15.71, 15.52, 16.07, 16.00, 16.30, 12.18, 16.11, 16.64, 16.47, 15.99, 15.51],
    },
    'EUCLID': {
        'sqz':   [2.44, 2.98, 3.06, 3.14, 2.95, 2.92, 2.94, 3.41, 3.18, 3.17, 3.25, 3.08, 2.98, 3.06],
        'tanat': [535.30, 496.29, 788.74, 590.09, 619.68, 609.93, 589.81, 617.00, 480.17, 602.85, 606.42, 635.56, 615.35, 605.94],
    },
}

# ---- Exp3: Sample Size (L=30, U=85) ----
exp3_x = [1, 2, 3, 4, 5, 10, 20, 30, 50]  # in thousands

exp3 = {
    'OM': {
        'sqz':   [0.15, 0.45, 1.00, 1.73, 2.18, 8.78, 31.37, 73.58, 449.96],
        'tanat': [1.46, 2.43, 4.56, 7.34, 9.95, 42.44, 139.94, 315.45, 1061.85],
    },
    'LCS': {
        'sqz':   [0.058, 0.231, 0.540, 0.985, 1.45, 6.16, 24.12, 56.97, 176.52],
        'tanat': [1.39, 3.55, 9.38, 20.50, 4.92, 53.81, 85.27, 202.41, 719.07],
    },
    'LCP': {
        'sqz':   [0.007, 0.016, 0.032, 0.071, 0.13, 0.38, 1.74, 3.91, 21.80],
        'tanat': [1.97, 2.13, 3.28, 4.19, 4.82, 12.18, 63.95, 136.44, 382.56],
    },
    'EUCLID': {
        'sqz':   [0.026, 0.080, 0.190, 0.376, 0.63, 3.18, 13.52, 32.33, 128.81],
        'tanat': [2.78, 11.95, 31.26, 58.18, 156.23, 480.17, 3275.81, 5659.93, 16684.50],
    },
}


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def compute_speedup(tanat_times, sqz_times):
    """Speedup ratio = Time(TanaT) / Time(Sequenzo). >1 means Sequenzo faster."""
    result = []
    for t, s in zip(tanat_times, sqz_times):
        if t is not None and s is not None and s > 0:
            result.append(t / s)
        else:
            result.append(None)
    return result


def compute_relative_improvement(tanat_times, sqz_times):
    """Relative improvement = [Time(TanaT) - Time(Sequenzo)] / Time(TanaT). 
    Answers: Sequenzo is X% faster than TanaT."""
    result = []
    for t, s in zip(tanat_times, sqz_times):
        if t is not None and s is not None and t > 0:
            result.append((t - s) / t * 100)
        else:
            result.append(None)
    return result


def filter_none(x_vals, y_vals):
    """Remove None entries for plotting."""
    x_out, y_out = [], []
    for x, y in zip(x_vals, y_vals):
        if y is not None:
            x_out.append(x)
            y_out.append(y)
    return x_out, y_out


# ======================================================================
# STYLE SETTINGS
# ======================================================================

COLORS = {
    'sqz': '#2196F3',      # blue
    'tanat': '#FF9800',    # orange
    'speedup': '#4CAF50',  # green
    'improvement': '#9C27B0',  # purple
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ======================================================================
# PLOT FUNCTION: 3 rows (absolute time, speedup ratio, relative improvement)
# ======================================================================

def plot_experiment(exp_data, x_values, xlabel, exp_title, filename,
                    x_is_thousands=False, log_y_time=False):
    """
    Plot one experiment with 4 metrics x 3 rows:
    Row 1: Absolute time (Sequenzo vs TanaT)
    Row 2: Speedup ratio (TanaT/Sequenzo)
    Row 3: Relative improvement (%)
    """
    metrics = ['OM', 'LCS', 'LCP', 'EUCLID']
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(exp_title, fontsize=14, fontweight='bold', y=0.98)

    for col, metric in enumerate(metrics):
        sqz = exp_data[metric]['sqz']
        tanat = exp_data[metric]['tanat']

        speedup = compute_speedup(tanat, sqz)
        improvement = compute_relative_improvement(tanat, sqz)

        x_display = x_values

        # --- Row 1: Absolute time ---
        ax = axes[0][col]
        x_sqz, y_sqz = filter_none(x_display, sqz)
        x_tat, y_tat = filter_none(x_display, tanat)

        ax.plot(x_sqz, y_sqz, 'o-', color=COLORS['sqz'], label='Sequenzo', 
                markersize=5, linewidth=1.5)
        ax.plot(x_tat, y_tat, 's-', color=COLORS['tanat'], label='TanaT', 
                markersize=5, linewidth=1.5)

        ax.set_title(metric)
        if col == 0:
            ax.set_ylabel('Time (seconds)')
        ax.legend(loc='upper left', framealpha=0.8)
        ax.grid(True, alpha=0.3)
        if log_y_time:
            ax.set_yscale('log')

        # --- Row 2: Speedup ratio ---
        ax = axes[1][col]
        x_sp, y_sp = filter_none(x_display, speedup)

        ax.plot(x_sp, y_sp, 'D-', color=COLORS['speedup'], markersize=5, linewidth=1.5)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='1x (equal)')
        ax.fill_between(x_sp, 1, y_sp, alpha=0.15, color=COLORS['speedup'])

        if col == 0:
            ax.set_ylabel('Speedup Ratio\n(TanaT / Sequenzo)')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for x, y in zip(x_sp, y_sp):
            ax.annotate(f'{y:.1f}x', (x, y), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=8, color=COLORS['speedup'])

        # --- Row 3: Relative improvement ---
        ax = axes[2][col]
        x_ri, y_ri = filter_none(x_display, improvement)

        ax.plot(x_ri, y_ri, '^-', color=COLORS['improvement'], markersize=5, linewidth=1.5)
        ax.fill_between(x_ri, 0, y_ri, alpha=0.15, color=COLORS['improvement'])

        ax.set_xlabel(xlabel)
        if col == 0:
            ax.set_ylabel('Relative Improvement (%)\n(TanaT−Sqz) / TanaT')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # Add value labels
        for x, y in zip(x_ri, y_ri):
            ax.annotate(f'{y:.0f}%', (x, y), textcoords="offset points",
                       xytext=(0, 8), ha='center', fontsize=8, color=COLORS['improvement'])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")


# ======================================================================
# GENERATE ALL PLOTS
# ======================================================================

if __name__ == '__main__':

    # --- Exp1: Sequence Length ---
    plot_experiment(
        exp_data=exp1,
        x_values=exp1_x,
        xlabel='Sequence Length',
        exp_title='Exp1: Sequence Length (n=10,000, U=85)',
        filename='exp1_sequence_length.png',
        log_y_time=True,
    )

    # --- Exp2: Uniqueness Rate ---
    plot_experiment(
        exp_data=exp2,
        x_values=exp2_x,
        xlabel='Uniqueness Rate (%)',
        exp_title='Exp2: Uniqueness Rate (n=10,000, L=30)',
        filename='exp2_uniqueness_rate.png',
    )

    # --- Exp3: Sample Size ---
    plot_experiment(
        exp_data=exp3,
        x_values=exp3_x,
        xlabel='Sample Size (1,000)',
        exp_title='Exp3: Sample Size (L=30, U=85)',
        filename='exp3_sample_size.png',
        log_y_time=True,
    )

    # --- Summary: Speedup Ratio across all experiments ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Speedup Ratio: TanaT / Sequenzo (>1 = Sequenzo Faster)', 
                 fontsize=13, fontweight='bold')

    metrics = ['OM', 'LCS', 'LCP', 'EUCLID']
    metric_markers = {'OM': 'o', 'LCS': 's', 'LCP': 'D', 'EUCLID': '^'}
    metric_colors = {'OM': '#2196F3', 'LCS': '#FF5722', 'LCP': '#4CAF50', 'EUCLID': '#9C27B0'}

    for ax_idx, (exp_name, exp_data, x_vals, xlabel) in enumerate([
        ('Exp1: Sequence Length', exp1, exp1_x, 'Sequence Length'),
        ('Exp2: Uniqueness Rate', exp2, exp2_x, 'Uniqueness Rate (%)'),
        ('Exp3: Sample Size', exp3, exp3_x, 'Sample Size (1,000)'),
    ]):
        ax = axes[ax_idx]
        for metric in metrics:
            speedup = compute_speedup(exp_data[metric]['tanat'], exp_data[metric]['sqz'])
            x_f, y_f = filter_none(x_vals, speedup)
            ax.plot(x_f, y_f, marker=metric_markers[metric], color=metric_colors[metric],
                   label=metric, markersize=5, linewidth=1.5)

        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(exp_name)
        ax.set_xlabel(xlabel)
        if ax_idx == 0:
            ax.set_ylabel('Speedup Ratio')
        ax.set_yscale('log')
        ax.legend(loc='best', framealpha=0.8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('speedup_summary.png')
    plt.close()
    print("Saved: speedup_summary.png")

    print("\nAll plots generated.")
