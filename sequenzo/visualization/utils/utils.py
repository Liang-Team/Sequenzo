"""
@Author  : Yuqi Liang 梁彧祺
@File    : utils.py
@Time    : 01/03/2025 10:16
@Desc    : 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from io import BytesIO
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any

from sequenzo import SequenceData


def set_up_time_labels_for_x_axis(seqdata: SequenceData,
                                  ax: Axes) -> None:
    """
    Helper function to set up time labels for the x-axis.

    :param seqdata: (SequenceData) A SequenceData object containing time information
    :param ax: (matplotlib.axes.Axes) The axes to set labels on
    """
    # Extract time labels (year or age)
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
    # Note that here is black, but in the index plot the x label is gray
    # as I set it in the index plot function: ax.tick_params(axis='x', colors='gray', length=4, width=0.7)
    # TODO: think about the uniform color setting for the x label in the whole project


def create_standalone_legend(
        colors: Dict[str, str],
        labels: List[str],
        ncol: int = 5,
        figsize: Tuple[int, int] = (8, 1),
        fontsize: int = 10,
        dpi: int = 200
        ) -> BytesIO:
    """
    Creates a standalone legend image without borders.

    Parameters:
        colors: Dictionary mapping labels to color values
        labels: List of state labels to include in the legend
        ncol: Number of columns in the legend
        figsize: Size of the figure (width, height)
        fontsize: Font size for legend text
        dpi: Resolution of the output image

    Returns:
        BytesIO: In-memory buffer containing the legend image
    """
    # Create a new figure for the legend
    legend_fig = plt.figure(figsize=figsize)
    ax = legend_fig.add_subplot(111)
    ax.axis('off')  # Hide the axes

    # Create handles for the legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors.get(s, "gray")) for s in labels]

    # Create the legend without a frame
    legend = ax.legend(
        handles,
        labels,
        loc='center',
        ncol=min(ncol, len(labels)),
        frameon=False,  # No border around legend
        fontsize=fontsize
    )

    # Save to memory buffer
    buffer = BytesIO()
    legend_fig.savefig(
        buffer,
        format='png',
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0  # Remove padding
    )
    plt.close(legend_fig)
    buffer.seek(0)

    return buffer


def combine_plot_with_legend(
        main_image_buffer: BytesIO,
        legend_buffer: BytesIO,
        output_path: Optional[str] = None,
        dpi: int = 200,
        padding: int = 10
) -> Image.Image:
    """
    Combines a main plot image with a legend image, placing the legend below the main plot.
    This means that it saves the combined image to a file if an output path is provided,
    which is different from the function `save_and_show_results` as that is responsible for visualizations that do not require cropping.

    Parameters:
        main_image_buffer: Buffer containing the main plot image
        legend_buffer: Buffer containing the legend image
        output_path: Optional path to save the combined image
        dpi: Resolution for the saved image
        padding: Padding between main image and legend in pixels

    Returns:
        PIL.Image: The combined image
    """
    # Open images from buffers
    main_img = Image.open(main_image_buffer)
    legend_img = Image.open(legend_buffer)

    # Calculate dimensions for combined image
    combined_width = max(main_img.width, legend_img.width)
    combined_height = main_img.height + padding + legend_img.height

    # Create new blank image
    combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

    # Paste main image at top
    combined_img.paste(main_img, (0, 0))

    # Center and paste legend below main image with padding
    legend_x = (combined_width - legend_img.width) // 2
    combined_img.paste(legend_img, (legend_x, main_img.height + padding))

    # Save if output path is provided
    if output_path:
        combined_img.save(output_path, dpi=(dpi, dpi))

    return combined_img


def save_figure_to_buffer(fig, dpi: int = 200) -> BytesIO:
    """
    Saves a matplotlib figure to a BytesIO buffer.

    Parameters:
        fig: Matplotlib figure to save
        dpi: Resolution of the output image

    Returns:
        BytesIO: Buffer containing the figure image
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return buffer


def save_and_show_results(save_as, dpi=200):
    if save_as:
        # Ensure the filename has an extension
        if not any(save_as.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
            save_as = f"{save_as}.png"  # Add default .png extension

        plt.savefig(save_as, dpi=dpi, bbox_inches='tight')

    plt.show()
    # Release resources
    plt.close()

