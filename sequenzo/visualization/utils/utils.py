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
                                  ax: Axes,
                                  color: str = "gray") -> None:
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
    ax.set_xticklabels(time_labels[xtick_positions], fontsize=10, rotation=0, ha="center", color=color)


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


def save_and_show_results(save_as=None, dpi=200, show=True):
    """
    Save and optionally display matplotlib figure.

    Parameters:
    - save_as: Optional filename to save the figure to
    - dpi: Resolution for saved image
    - show: Whether to display the figure (default: True)

    Returns:
    - The current figure object (plt.gcf())
    """
    if save_as:
        if not any(save_as.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
            save_as += '.png'
        plt.savefig(save_as, dpi=dpi, bbox_inches='tight')

    # Show if explicitly requested or if saving (for visual confirmation)
    if show or save_as:
        plt.show()

    # Prevent memory leak by closing after show/save
    plt.close()

    # Return the current figure
    return plt.gcf()



def determine_layout(num_items: int,
                     layout: str = 'column',
                     nrows: Optional[int] = None,
                     ncols: Optional[int] = None) -> Tuple[int, int]:
    """
    Determine subplot layout (rows, columns) based on the number of items to plot.
    TODO: 1. change to all visualizations that require multiple graphs 2. 发包，因为我也改了颜色了

    Parameters:
        num_items (int): Total number of subplots needed.
        layout (str): Layout strategy ('column' or 'grid'), used if nrows/ncols not provided.
        nrows (int, optional): Number of rows (manual override).
        ncols (int, optional): Number of columns (manual override).

    Returns:
        Tuple[int, int]: A tuple of (nrows, ncols) for subplot layout.

    Raises:
        ValueError: If layout config is invalid or does not fit all subplots.
    """
    # Check partial input
    if (nrows is None and ncols is not None) or (ncols is None and nrows is not None):
        raise ValueError("If manually specifying layout, both 'nrows' and 'ncols' must be provided.")

    # Manual override
    if nrows is not None and ncols is not None:
        total_slots = nrows * ncols
        if total_slots < num_items:
            raise ValueError(f"Provided layout ({nrows}x{ncols}) is too small for {num_items} plots.")
        return nrows, ncols

    # Automatic layout
    if layout == 'column':
        ncols = 3
        nrows = (num_items + ncols - 1) // ncols
    elif layout == 'grid':
        ncols = int(np.ceil(np.sqrt(num_items)))
        nrows = (num_items + ncols - 1) // ncols
    else:
        raise ValueError(f"Unsupported layout style: '{layout}'")

    return nrows, ncols

