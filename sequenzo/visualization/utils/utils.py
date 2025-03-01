"""
@Author  : Yuqi Liang 梁彧祺
@File    : utils.py
@Time    : 01/03/2025 10:16
@Desc    : 
"""
import numpy as np
import matplotlib.pyplot as plt


def set_up_time_labels_for_x_axis(seqdata, ax):
    # Extract time labels (years)

    time_labels = np.array(seqdata.cleaned_time)

    # Determine the number of time steps
    num_time_steps = len(time_labels)

    # Dynamic X-Tick Adjustment
    if num_time_steps <= 10:
        # If 10 or fewer time points, show all labels
        xtick_positions = np.arange(num_time_steps)
    elif num_time_steps <= 20:
        # If 10–20 time points, show every 2nd label
        xtick_positions = np.arange(0, num_time_steps, step=2)
    else:
        # More than 20 time points → Pick 10 evenly spaced tick positions
        xtick_positions = np.linspace(0, num_time_steps - 1, num=10, dtype=int)

    # Set x-ticks and labels dynamically
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(time_labels[xtick_positions], fontsize=10, rotation=0, ha="center", color="black")



