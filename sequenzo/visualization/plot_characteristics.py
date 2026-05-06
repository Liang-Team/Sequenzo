"""
@Author  : Yuqi Liang 梁彧祺
@File    : plot_characteristics.py
@Time    : 2025/9/24 23:22
@Desc    : Plot longitudinal and cross-sectional sequence characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib.font_manager import FontProperties

from sequenzo.sequence_characteristics_indicators.simple_characteristics import get_number_of_transitions
from sequenzo.sequence_characteristics_indicators.within_sequence_entropy import get_within_sequence_entropy
from sequenzo.sequence_characteristics_indicators.turbulence import get_turbulence
from sequenzo.sequence_characteristics_indicators.complexity_index import get_complexity_index
from sequenzo.sequence_characteristics_indicators.overall_cross_sectional_entropy import get_cross_sectional_entropy
from sequenzo.visualization.utils.utils import set_up_time_labels_for_x_axis


def plot_longitudinal_characteristics(seqdata,
                                      pick_ids=None,
                                      k=9,
                                      selection='first',
                                      order_by="complexity",
                                      figsize=(8, 6),
                                      fontsize=12,
                                      title=None,
                                      show_title=True,
                                      xlabel="Normalized Values",
                                      ylabel="Sequence ID",
                                      save_as=None,
                                      dpi=200,
                                      custom_colors=None,
                                      show_sequence_ids=False,
                                      id_as_column=True):
    """Create a horizontal bar chart for four sequence characteristics."""
    df_t = get_number_of_transitions(seqdata=seqdata, norm=True).iloc[:, 1]
    df_e = get_within_sequence_entropy(seqdata=seqdata, norm=True)
    if isinstance(df_e, pd.DataFrame):
        df_e = df_e.iloc[:, 1]

    df_tb = get_turbulence(seqdata=seqdata, norm=True, type=2, id_as_column=True)
    if isinstance(df_tb, pd.DataFrame):
        df_tb = df_tb.iloc[:, 1]

    df_c = get_complexity_index(seqdata=seqdata)
    if isinstance(df_c, pd.DataFrame):
        df_c = df_c.iloc[:, 1]

    metrics = pd.DataFrame({
        "Transitions": df_t,
        "Entropy": df_e,
        "Turbulence": df_tb,
        "Complexity": df_c
    })

    if hasattr(seqdata, 'ids') and seqdata.ids is not None:
        metrics.index = seqdata.ids

    if pick_ids is not None:
        num_sequences = len(pick_ids)
        if num_sequences > 15:
            warnings.warn(
                f"Plotting {num_sequences} sequences may cause overplotting issues. "
                f"Consider reducing to 15 or fewer sequences for better visualization.",
                UserWarning,
            )
    elif k > 15:
        warnings.warn(
            f"Plotting {k} sequences may cause overplotting issues. "
            f"Consider reducing to 15 or fewer sequences for better visualization.",
            UserWarning,
        )

    if pick_ids is not None:
        metrics = metrics.loc[pick_ids]
    else:
        key = order_by.capitalize()
        if key not in metrics.columns:
            key = "Complexity"

        metrics_sorted = metrics.sort_values(key, ascending=False)
        if selection == 'first':
            metrics = metrics_sorted.head(k)
        elif selection == 'last':
            metrics = metrics_sorted.tail(k)
        else:
            metrics = metrics_sorted.head(k)

    labels = list(metrics.index) if show_sequence_ids else list(range(1, len(metrics) + 1))
    y = np.arange(len(metrics))
    bar_h = 0.18

    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    axis_gray = '#666666'

    if show_title and title is not None:
        plt.title(title, fontsize=fontsize + 2, color=axis_gray)

    default_colors = {
        'Transitions': '#74C9B4',
        'Entropy': '#A6E3D0',
        'Turbulence': '#F9E79F',
        'Complexity': '#F6CDA3',
    }
    if isinstance(custom_colors, dict):
        colors = {**default_colors, **custom_colors}
    elif isinstance(custom_colors, (list, tuple)) and len(custom_colors) == 4:
        ordered_keys = ['Transitions', 'Entropy', 'Turbulence', 'Complexity']
        colors = {k: v for k, v in zip(ordered_keys, custom_colors)}
    else:
        colors = default_colors

    plt.barh(y + 0.30, metrics["Transitions"].values, height=bar_h, label="Transitions", color=colors['Transitions'])
    plt.barh(y + 0.10, metrics["Entropy"].values, height=bar_h, label="Entropy", color=colors['Entropy'])
    plt.barh(y - 0.10, metrics["Turbulence"].values, height=bar_h, label="Turbulence", color=colors['Turbulence'])
    plt.barh(y - 0.30, metrics["Complexity"].values, height=bar_h, label="Complexity", color=colors['Complexity'])

    plt.yticks(y, labels)
    plt.xlim(0, 1)

    ax.set_xlabel(xlabel, labelpad=8, fontsize=fontsize, color=axis_gray)
    ylabel_props = FontProperties(stretch='expanded')
    ax.set_ylabel(ylabel, labelpad=6, fontproperties=ylabel_props, fontsize=fontsize, color=axis_gray)
    plt.legend(loc="lower right", fontsize=max(6, fontsize - 1))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color(axis_gray)
    ax.spines['bottom'].set_color(axis_gray)
    ax.spines['left'].set_position(('outward', 2))
    ax.spines['bottom'].set_position(('outward', 4))
    ax.tick_params(axis='x', which='major', colors=axis_gray, length=4, width=0.7, direction='out', pad=4, labelsize=max(6, fontsize - 1))
    ax.tick_params(axis='y', which='major', colors=axis_gray, length=4, width=0.7, direction='out', pad=3, labelsize=max(6, fontsize - 1))
    ax.set_ylim(-0.5, len(metrics) - 0.5)
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    if save_as:
        if not any(save_as.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
            save_as += '.png'
        plt.savefig(save_as, dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.close()

    if id_as_column:
        metrics_result = metrics.copy()
        metrics_result['ID'] = metrics_result.index
        metrics_result = metrics_result[['ID', 'Transitions', 'Entropy', 'Turbulence', 'Complexity']].reset_index(drop=True)
        return metrics_result

    metrics.index.name = 'ID'
    return metrics


def plot_cross_sectional_characteristics(seqdata,
                                         figsize=(10, 6),
                                         fontsize=12,
                                         title="Cross-sectional entropy over time",
                                         show_title=True,
                                         xlabel="Time",
                                         ylabel="Entropy (0-1)",
                                         line_color="#74C9B4",
                                         save_as=None,
                                         dpi=200,
                                         return_data=False,
                                         custom_state_colors=None):
    """Visualize normalized cross-sectional entropy across time points."""
    res = get_cross_sectional_entropy(seqdata, weighted=True, norm=True, return_format="dict")
    freq = res["Frequencies"]
    ent = res["per_time_entropy_norm"] if "per_time_entropy_norm" in res and res["per_time_entropy_norm"] is not None else res["Entropy"]
    n_valid = res.get("ValidStates", None)

    try:
        sorted_cols = sorted(freq.columns, key=lambda x: int(x))
        freq = freq[sorted_cols]
        ent = ent.loc[sorted_cols]
        if n_valid is not None:
            n_valid = n_valid.loc[sorted_cols]
    except (ValueError, TypeError):
        pass

    if custom_state_colors is not None:
        _ = custom_state_colors

    axis_gray = '#666666'
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.plot(ent.index, ent.values, marker='o', color=line_color, linewidth=2, markersize=4)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(ylabel, fontsize=fontsize, color=axis_gray)

    if show_title and title:
        ax1.set_title(title, fontsize=fontsize + 1, color=axis_gray)

    set_up_time_labels_for_x_axis(seqdata, ax1, color=axis_gray)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('gray')
    ax1.spines['bottom'].set_color('gray')
    ax1.spines['left'].set_linewidth(0.7)
    ax1.spines['bottom'].set_linewidth(0.7)
    ax1.spines['left'].set_position(('outward', 5))
    ax1.spines['bottom'].set_position(('outward', 5))
    ax1.tick_params(axis='both', colors=axis_gray, labelsize=max(6, fontsize - 1), length=4, width=0.7)
    ax1.set_xlabel(xlabel, fontsize=fontsize, color=axis_gray)

    plt.tight_layout()
    if save_as:
        if not any(save_as.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
            save_as += '.png'
        plt.savefig(save_as, dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.close()

    if return_data:
        return {"Frequencies": freq, "Entropy": ent, "ValidStates": n_valid}
    return None
