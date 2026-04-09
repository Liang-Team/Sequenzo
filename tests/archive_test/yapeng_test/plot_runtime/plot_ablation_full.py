"""
Ablation Experiment Visualization: 2x2 factorial (C++ optimization x OpenMP)
Full version: ABCD group comparison
"""
import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Data: n=10k, L=30, U=85 baseline (seconds)
# ============================================================
metrics = ['OMspell', 'OMtspell', 'TWED', 'OMloc', 'OMslen']

# Group A: Pre, OMP OFF (single-threaded original code) - baseline
A = np.array([62.38, 65.42, 178.04, 514.31, 493.32])

# Group B: Pre, OMP ON (multi-threaded original code)
B = np.array([12.12, 13.81, 65.95, 133.11, 132.95])

# Group C: Post, OMP OFF (single-threaded optimized code)
C = np.array([57.86, 61.03, 160.17, 97.85, 92.07])

# Group D: Post, OMP ON (multi-threaded optimized code) - final version
D = np.array([11.72, 12.52, 45.36, 20.96, 29.02])

# ============================================================
# Calculate all speedup ratios
# ============================================================
# Row comparison (OpenMP effect)
A_div_B = A / B   # OpenMP effect (pre-opt): A->B
C_div_D = C / D   # OpenMP effect (post-opt): C->D

# Column comparison (C++ optimization effect)
A_div_C = A / C   # C++ optimization effect (single-threaded): A->C
B_div_D = B / D   # C++ optimization effect (multi-threaded): A->C under parallel

# Diagonal comparison (total effect)
A_div_D = A / D   # Total speedup: baseline -> final

# Cross comparison
A_div_B_vs_C_div_D = (A / B) / (C / D)  # OpenMP effect change
A_div_C_vs_B_div_D = (A / C) / (B / D)  # C++ optimization effect change

# ============================================================
# Figure 1: ABCD runtime comparison + speedup annotations
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Subplot 1: Runtime (bar chart)
ax1 = axes[0, 0]
x = np.arange(len(metrics))
width = 0.2
bars_A = ax1.bar(x - 1.5*width, A, width, label='A: Pre + OMP OFF', color='#d62728', alpha=0.85)
bars_B = ax1.bar(x - 0.5*width, B, width, label='B: Pre + OMP ON', color='#ff7f0e', alpha=0.85)
bars_C = ax1.bar(x + 0.5*width, C, width, label='C: Post + OMP OFF', color='#2ca02c', alpha=0.85)
bars_D = ax1.bar(x + 1.5*width, D, width, label='D: Post + OMP ON', color='#1f77b4', alpha=0.85)
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Runtime Comparison (n=10,000, L=30, U=85)')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(loc='upper left', fontsize=8)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: A vs B (OpenMP effect, pre-opt)
ax2 = axes[0, 1]
colors = ['#3498db' if s > 4 else '#f39c12' if s > 3 else '#e74c3c' for s in A_div_B]
bars = ax2.bar(metrics, A_div_B, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel('Speedup (A/B)')
ax2.set_title('A→B: OpenMP Effect (Pre-optimization)')
for bar, val in zip(bars, A_div_B):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylim(0, max(A_div_B) * 1.15)
ax2.grid(axis='y', alpha=0.3)

# Subplot 3: C vs D (OpenMP effect, post-opt)
ax3 = axes[0, 2]
colors = ['#3498db' if s > 4 else '#f39c12' if s > 3 else '#e74c3c' for s in C_div_D]
bars = ax3.bar(metrics, C_div_D, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax3.set_ylabel('Speedup (C/D)')
ax3.set_title('C→D: OpenMP Effect (Post-optimization)')
for bar, val in zip(bars, C_div_D):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylim(0, max(C_div_D) * 1.15)
ax3.grid(axis='y', alpha=0.3)

# Subplot 4: A vs C (C++ optimization effect, single-threaded)
ax4 = axes[1, 0]
colors = ['#27ae60' if s > 3 else '#f39c12' if s > 1.5 else '#e74c3c' for s in A_div_C]
bars = ax4.bar(metrics, A_div_C, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax4.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax4.set_ylabel('Speedup (A/C)')
ax4.set_title('A→C: C++ Optimization Effect (OMP OFF)')
for bar, val in zip(bars, A_div_C):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.set_ylim(0, max(A_div_C) * 1.15)
ax4.grid(axis='y', alpha=0.3)

# Subplot 5: B vs D (C++ optimization effect, multi-threaded)
ax5 = axes[1, 1]
colors = ['#27ae60' if s > 3 else '#f39c12' if s > 1.5 else '#e74c3c' for s in B_div_D]
bars = ax5.bar(metrics, B_div_D, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax5.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax5.set_ylabel('Speedup (B/D)')
ax5.set_title('B→D: C++ Optimization Effect (OMP ON)')
for bar, val in zip(bars, B_div_D):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax5.set_ylim(0, max(B_div_D) * 1.15)
ax5.grid(axis='y', alpha=0.3)

# Subplot 6: A vs D (total speedup)
ax6 = axes[1, 2]
colors = ['#27ae60' if s > 10 else '#3498db' if s > 5 else '#f39c12' for s in A_div_D]
bars = ax6.bar(metrics, A_div_D, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax6.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax6.set_ylabel('Speedup (A/D)')
ax6.set_title('A→D: Total Speedup (Baseline → Final)')
for bar, val in zip(bars, A_div_D):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax6.set_ylim(0, max(A_div_D) * 1.15)
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_abcd_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ablation_abcd_comparison.png")

# ============================================================
# Figure 2: 2x2 Factorial matrix view
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: OpenMP effect comparison (A->B vs C->D)
ax1 = axes[0]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax1.bar(x - width/2, A_div_B, width, label='Pre-opt (A→B)', color='#e74c3c', alpha=0.85)
bars2 = ax1.bar(x + width/2, C_div_D, width, label='Post-opt (C→D)', color='#3498db', alpha=0.85)
ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax1.set_ylabel('Speedup')
ax1.set_title('OpenMP Parallel Effect: Pre vs Post Optimization')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
for bar, val in zip(bars1, A_div_B):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.1f}x', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, C_div_D):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.1f}x', ha='center', va='bottom', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Right: C++ optimization effect comparison (A->C vs B->D)
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, A_div_C, width, label='OMP OFF (A→C)', color='#2ca02c', alpha=0.85)
bars2 = ax2.bar(x + width/2, B_div_D, width, label='OMP ON (B→D)', color='#9b59b6', alpha=0.85)
ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1)
ax2.set_ylabel('Speedup')
ax2.set_title('C++ Optimization Effect: OMP OFF vs OMP ON')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
for bar, val in zip(bars1, A_div_C):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, B_div_D):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_factorial_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ablation_factorial_comparison.png")

# ============================================================
# Figure 3: Speedup decomposition (stacked bar chart) - multiplicative relationship
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metrics))
width = 0.5

# Log stacking to show multiplicative relationship: A/D = (A/C) x (C/D)
log_cpp = np.log(A_div_C)
log_omp = np.log(C_div_D)

bars1 = ax.bar(x, log_cpp, width, label='C++ Optimization (A→C)', color='#2ca02c', alpha=0.85)
bars2 = ax.bar(x, log_omp, width, bottom=log_cpp, label='OpenMP Parallel (C→D)', color='#3498db', alpha=0.85)

# Add annotations
for i, (cpp, omp, total) in enumerate(zip(A_div_C, C_div_D, A_div_D)):
    # Total speedup
    ax.text(i, log_cpp[i] + log_omp[i] + 0.08, f'{total:.1f}x', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Component annotations
    if log_cpp[i] > 0.3:
        ax.text(i, log_cpp[i]/2, f'{cpp:.2f}x', ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold')
    if log_omp[i] > 0.3:
        ax.text(i, log_cpp[i] + log_omp[i]/2, f'{omp:.1f}x', ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold')

ax.set_ylabel('log(Speedup)')
ax.set_title('Speedup Decomposition: Total = C++ × OpenMP\n(n=10,000, L=30, U=85)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Right y-axis showing actual multipliers
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
yticks_log = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
yticks_linear = [f'{np.exp(y):.1f}x' for y in yticks_log]
ax2.set_yticks(yticks_log)
ax2.set_yticklabels(yticks_linear)
ax2.set_ylabel('Actual Speedup')

plt.tight_layout()
plt.savefig('ablation_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ablation_decomposition.png")

# ============================================================
# Figure 4: Heatmap - ABCD performance across datasets
# ============================================================
datasets_short = ['n=1k', 'n=5k', 'n=10k\n(base)', 'U=30', 'U=50', 'L=10', 'L=50']

# Group D data (Post, OMP ON) - all datasets
D_all = np.array([
    [0.25, 0.21, 0.45, 0.25, 0.38],   # n=1k
    [3.05, 3.19, 10.50, 4.84, 6.80],  # n=5k
    [11.72, 12.52, 45.36, 20.96, 29.02],  # n=10k, L=30, U=85
    [2.38, 2.47, 5.77, 3.08, 3.65],   # U=30
    [4.70, 4.97, 15.30, 7.27, 10.11], # U=50
    [1.76, 1.91, 4.39, 2.58, 3.18],   # L=10
    [31.91, 33.90, 127.99, 54.52, 78.60],  # L=50
])

# Group C data (Post, OMP OFF)
C_all = np.array([
    [0.68, 0.74, 1.47, 1.00, 1.08],   # n=1k
    [14.53, 15.34, 37.79, 23.55, 22.80],  # n=5k
    [57.86, 61.03, 160.17, 97.85, 92.07],  # n=10k, L=30, U=85
    [8.26, 8.62, 18.69, 12.04, 12.03],   # U=30
    [20.83, 21.97, 52.98, 33.12, 32.00], # U=50
    [6.54, 7.30, 14.73, 10.69, 11.61],   # L=10
    [162.43, 169.18, 461.17, 267.53, 257.90],  # L=50
])

# OpenMP speedup (C->D)
omp_speedup_all = C_all / D_all

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(omp_speedup_all, cmap='YlOrRd', aspect='auto', vmin=1, vmax=6)

# Add value annotations
for i in range(len(datasets_short)):
    for j in range(len(metrics)):
        text = ax.text(j, i, f'{omp_speedup_all[i, j]:.1f}x',
                       ha='center', va='center', color='black', fontsize=10)

ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(datasets_short)))
ax.set_xticklabels(metrics)
ax.set_yticklabels(datasets_short)
ax.set_title('OpenMP Speedup (C→D) Across Datasets')
ax.set_xlabel('Metric')
ax.set_ylabel('Dataset')

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Speedup', rotation=-90, va='bottom')

plt.tight_layout()
plt.savefig('ablation_heatmap_omp.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ablation_heatmap_omp.png")

# ============================================================
# Figure 5: Summary table view (for reports)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')

# Create table data
table_data = [
    ['Metric', 'A (Pre,OFF)', 'B (Pre,ON)', 'C (Post,OFF)', 'D (Post,ON)', 
     'A/B\n(OMP Pre)', 'C/D\n(OMP Post)', 'A/C\n(C++ OFF)', 'B/D\n(C++ ON)', 'A/D\n(Total)'],
]
for i, m in enumerate(metrics):
    row = [
        m,
        f'{A[i]:.2f}s',
        f'{B[i]:.2f}s',
        f'{C[i]:.2f}s',
        f'{D[i]:.2f}s',
        f'{A_div_B[i]:.2f}x',
        f'{C_div_D[i]:.2f}x',
        f'{A_div_C[i]:.2f}x',
        f'{B_div_D[i]:.2f}x',
        f'{A_div_D[i]:.1f}x',
    ]
    table_data.append(row)

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

# Set header style
for j in range(10):
    table[(0, j)].set_facecolor('#4a90d9')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Set data row style
for i in range(1, 6):
    for j in range(10):
        if j == 0:
            table[(i, j)].set_facecolor('#e8e8e8')
            table[(i, j)].set_text_props(fontweight='bold')
        elif j in [5, 6]:  # OMP speedup
            table[(i, j)].set_facecolor('#fff3cd')
        elif j in [7, 8]:  # C++ speedup
            table[(i, j)].set_facecolor('#d4edda')
        elif j == 9:  # Total speedup
            table[(i, j)].set_facecolor('#cce5ff')

ax.set_title('Ablation Experiment Summary: 2×2 Factorial Design\n(n=10,000, L=30, U=85)', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('ablation_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ablation_summary_table.png")

# ============================================================
# Print full summary
# ============================================================
print("\n" + "="*80)
print("Ablation Experiment Full Summary (n=10k, L=30, U=85)")
print("="*80)

print("\n[Runtime (seconds)]")
print(f"{'Metric':<12} {'A(Pre,OFF)':<12} {'B(Pre,ON)':<12} {'C(Post,OFF)':<13} {'D(Post,ON)':<12}")
print("-"*60)
for i, m in enumerate(metrics):
    print(f"{m:<12} {A[i]:<12.2f} {B[i]:<12.2f} {C[i]:<13.2f} {D[i]:<12.2f}")

print("\n[Speedup Summary]")
print(f"{'Metric':<12} {'A/B':<10} {'C/D':<10} {'A/C':<10} {'B/D':<10} {'A/D':<10}")
print(f"{'':12} {'(OMP Pre)':<10} {'(OMP Post)':<10} {'(C++ OFF)':<10} {'(C++ ON)':<10} {'(Total)':<10}")
print("-"*70)
for i, m in enumerate(metrics):
    print(f"{m:<12} {A_div_B[i]:<10.2f} {C_div_D[i]:<10.2f} {A_div_C[i]:<10.2f} {B_div_D[i]:<10.2f} {A_div_D[i]:<10.1f}")

print("\n[Key Findings]")
print(f"1. OMloc: C++ single-thread {A_div_C[3]:.1f}x, multi-thread {B_div_D[3]:.1f}x, total {A_div_D[3]:.1f}x (max)")
print(f"2. OMslen: C++ single-thread {A_div_C[4]:.1f}x, multi-thread {B_div_D[4]:.1f}x, total {A_div_D[4]:.1f}x")
print(f"3. OMspell/OMtspell: C++ only ~1.1x, mainly from OpenMP (~5x)")
print(f"4. TWED: C++ {A_div_C[2]:.2f}x, OpenMP(Post) {C_div_D[2]:.1f}x, total {A_div_D[2]:.1f}x")
print(f"5. OpenMP effect slightly reduced after optimization (A/B vs C/D)")

print("\n[2x2 Factorial Interpretation]")
print("+-------------+-----------------+-----------------+")
print("|             |   OMP OFF       |   OMP ON        |")
print("+-------------+-----------------+-----------------+")
print("| Pre (orig)  |   A (baseline)  |   B             |")
print("|             |                 |   A/B = OMP eff |")
print("+-------------+-----------------+-----------------+")
print("| Post (opt)  |   C             |   D (final)     |")
print("|             |   A/C = C++ eff |   A/D = total   |")
print("+-------------+-----------------+-----------------+")
