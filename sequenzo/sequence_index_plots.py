"""
@Author  : 梁彧祺
@File    : sequence_index_plots.py
@Time    : 29/12/2024 09:08
@Desc    : 
    Generate sequence index plots.
    TODO: when not single, graphs are not yet integrated with SequenceData
    including pre-loaded colormap, legend
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from PIL import Image
from seqdef import SequenceData


### Data Handling & Preprocessing ###

def _load_data(file_path):
    """
    Load data from a pickle file or DataFrame and return matrix and ID list.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, pd.DataFrame):
                id_list = data.index.tolist()
                matrix = data.values
            elif isinstance(data, np.ndarray):
                id_list = None
                matrix = data
            else:
                raise ValueError("Unsupported data type in pickle file.")

    except Exception as e:
        print(f"Error while loading the file: {e}")
        print("Attempting to reload with Pandas compatibility options...")
        data = pd.read_pickle(file_path, compression=None)
        if isinstance(data, pd.DataFrame):
            id_list = data.index.tolist()
            matrix = data.values
        else:
            raise ValueError("Pandas failed to load data properly.")

    return matrix, id_list


def preprocess_data(data):
    """
    Map string-based states in the data matrix to integer indices.
    TODO: rename this function as its indication is too ambiguous

    preprocess_data 的核心作用是将字符串类型的数据（如 'software'）转换为数值型数据（如 0, 1 等），
    因为字符串无法直接用于 imshow 函数，而 imshow 是用来处理数值矩阵并将其以颜色形式可视化的。
    """
    unique_states = np.unique(data)
    state_mapping = {state: idx for idx, state in enumerate(unique_states)}
    mapped_data = np.vectorize(state_mapping.get)(data)
    return mapped_data, state_mapping


### Plotting ###

# Main User-Facing Function
def sequence_index_plot(seqdata: SequenceData,
                        id_group_df=None,
                        custom_order=None,
                        sortv=None,
                        age_labels=None,
                        title=None,
                        facet_ncol=2,
                        xlabel="Age",
                        ylabel="Sequence ID",
                        save_as=None,
                        dpi=200):
    """
    Unified function for sequence index plotting.

    :param seqdata: (SequenceData) A SequenceData object containing preprocessed sequence data.
    :param id_group_df: (pd.DataFrame, optional) DataFrame containing IDs and grouping information for sequences.
    :param custom_order: (dict, optional) Custom sorting order for group labels.
    :param age_labels: (list, optional) Labels for the x-axis (e.g., ages or time points).
    :param title: (str, optional) Title for the plot.
    :param facet_ncol: (int, default=2) Number of columns for grouped plots.
    :param save_as: (str, optional) File path to save the plot.
    :param dpi: (int, default=200) Resolution of the saved plot.

    :return None.
    """
    if not isinstance(seqdata, SequenceData):
        raise TypeError("❌ Input data must be a SequenceData object.")

    if id_group_df is None:
        # Call the single plot function if no grouping is provided
        _sequence_index_plot_single(seqdata, age_labels=age_labels,
                                    figsize=(10, 6), title=title,
                                    xlabel=xlabel, ylabel=ylabel, save_as=save_as, dpi=dpi)
    else:

        if facet_ncol == 1:
            # Call the grouped plot function if grouping is provided
            _sequence_index_plot_grouping_ncol_1(seqdata, id_group_df=id_group_df,
                                                 age_labels=age_labels, title=title,
                                                 xlabel=xlabel, ylabel=ylabel,
                                                 facet_ncol=facet_ncol, save_as=save_as, dpi=dpi)
        else:
            _sequence_index_plot_grouping_ncol_not_1(seqdata, id_group_df=id_group_df,
                                                     age_labels=age_labels, title=title,
                                                     xlabel=xlabel, ylabel=ylabel,
                                                     facet_ncol=facet_ncol, save_as=save_as, dpi=dpi)


def _crop_bottom_whitespace(fig, threshold=240, margin=5):
    """
    Crop the bottom whitespace from a matplotlib figure and save or return the cropped image.

    :param fig: matplotlib.figure.Figure
        The Matplotlib figure to crop.
    :param threshold: int, default=240
        Pixel intensity threshold to detect whitespace.
    :param margin: int, default=5
        Additional margin to leave after cropping.

    :return: PIL.Image.Image
        The cropped image as a PIL Image object.
    """
    from io import BytesIO

    # Save the figure to an in-memory buffer
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    # Open the buffer as an image
    with Image.open(buffer) as img:
        img_array = np.array(img)

        # Detect non-white rows
        is_not_white = np.any(img_array < threshold, axis=-1)
        non_white_rows = np.where(is_not_white.any(axis=1))[0]

        if len(non_white_rows) > 0:
            first_content_row = non_white_rows[0]
            last_content_row = non_white_rows[-1]
            cropped = img.crop((0, first_content_row, img.width, last_content_row + margin))
        else:
            # If no content detected, return the original image
            cropped = img

        return cropped


def _sequence_index_plot_grouping_ncol_1(data, categories, id_group_df, age_labels=None,
                                         palette=None, reverse_colors=True, title=None,
                                         facet_ncol=2, save_as=None, dpi=200):
    """
    Plot sequence index plots with grouping.

    :param data: np.array or pd.DataFrame
        Sequence data matrix where rows are sequences and columns are time points.
    :param categories: list
        List of category labels corresponding to the data values.
    :param id_group_df: pd.DataFrame
        DataFrame containing IDs and grouping information for sequences.
    :param age_labels: list, optional
        Labels for the x-axis (e.g., ages or time points).
    :param palette: dict, optional
        Custom color palette mapping categories to colors.
    :param reverse_colors: bool, default=True
        Whether to reverse the color scheme.
    :param title: str, optional
        Title for the plot.
    :param facet_ncol: int, default=2
        Number of columns for grouped plots.
    :param save_as: str, optional
        File path to save the plot.
    :param dpi: int, default=200 as some personal computers might not have enough resources for more than 200
        Resolution of the saved plot.
    """
    num_colors = len(categories)
    spectral_colors = sns.color_palette("Spectral", num_colors)
    if reverse_colors:
        spectral_colors = list(reversed(spectral_colors))
    cmap = ListedColormap(spectral_colors) if palette is None else ListedColormap(
        [palette.get(cat, '#000000') for cat in categories])

    group_column_name = id_group_df.columns[1]
    group_labels = id_group_df[group_column_name].unique()
    facet_nrow = int(np.ceil(len(group_labels) / facet_ncol))

    width = 10
    height_per_subplot = 6

    fig = plt.figure(figsize=(width, height_per_subplot * facet_nrow))
    gs = plt.GridSpec(facet_nrow, facet_ncol + 1, figure=fig,
                      width_ratios=[*[1] * facet_ncol, 0.15],
                      wspace=0.2,
                      hspace=0.3,
                      top=0.92,  # 根据实际效果调整这个值
                      bottom=0.05)

    legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(i / (num_colors - 1)))
                      for i in range(num_colors)]

    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.legend(legend_patches, categories,
                     loc='center left',
                     bbox_to_anchor=(0, 0.7)
                     )
    legend_ax.axis('off')

    for idx, group in enumerate(group_labels):
        ax = fig.add_subplot(gs[idx // facet_ncol, idx % facet_ncol])

        # Check for group existence in id_group_df
        if group not in id_group_df[group_column_name].values:
            print(f"Group '{group}' not found in id_group_df.")
            ax.axis('off')
            continue

        group_data = data[id_group_df[group_column_name] == group]

        if len(group_data) > 0:
            sorted_data = group_data[np.lexsort(group_data.T[::-1])]
            im = ax.imshow(sorted_data, aspect='auto', cmap=cmap, interpolation='nearest')
            ax.set_title(f"{group} (N={len(group_data):,})", fontsize=10, pad=5, loc='center')

            if age_labels:
                ax.set_xticks(range(len(age_labels)))
                ax.set_xticklabels(age_labels, fontsize=8)
            else:
                ax.set_xticks(range(sorted_data.shape[1]))
                ax.set_xticklabels(range(1, sorted_data.shape[1] + 1), fontsize=8)

            num_yticks = min(10, sorted_data.shape[0])
            ytick_positions = np.linspace(0, sorted_data.shape[0] - 1, num_yticks, dtype=int)
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels([f"{int(x + 1):,}" for x in ytick_positions], fontsize=8)
        else:
            ax.axis('off')

        # Remove the spines (frame) and only keep the ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Adjust tick appearance
        ax.tick_params(axis='both', length=4, width=1, colors='black', labelsize=8)

        # Set consistent x and y axis labels
        ax.set_xlabel('Age', fontsize=10, labelpad=10)
        ax.set_ylabel('Sequence ID', fontsize=10, labelpad=10)

    if title is not None:
        # Center title above subplots only
        subplot_width = facet_ncol / (facet_ncol + 0.15)  # Exclude the legend width
        fig.suptitle(title, fontsize=15, x=subplot_width / 2, y=0.94)  # Adjusted y for spacing

    _crop_bottom_whitespace(fig)

    if save_as:
        # Ensure the file extension is valid
        if not save_as.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            save_as += '.png'  # Default to .png if no valid extension provided

        # Crop the figure and save
        cropped_image = _crop_bottom_whitespace(fig)
        cropped_image.save(save_as)  # Save the cropped image
        plt.close()
    else:
        # Show the cropped figure
        cropped_image = _crop_bottom_whitespace(fig)
        cropped_image.show()  # This opens the image using the default viewer


# Generalized function for sorting group labels
def sort_group_labels(group_labels, custom_order=None):
    """
    Sort group labels either using a custom order or natural sorting.

    :param group_labels: list
        List of group labels to be sorted.
    :param custom_order: dict, optional
        A dictionary mapping labels to their desired order.
        Example: {1: 'Software focused', 2: 'Hardware focused', ...}.

    :return: list
        Sorted group labels.
    """
    if custom_order:
        # If a custom order is provided, use it to sort the labels
        ordered_labels = sorted(group_labels, key=lambda x: list(custom_order.keys()).index(x))
    else:
        # Fall back to natural sorting (numerical or alphabetical)
        try:
            ordered_labels = sorted(group_labels, key=lambda x: int(x))
        except ValueError:
            # If labels are not all numerical, sort them alphabetically
            ordered_labels = sorted(group_labels, key=str)
    return ordered_labels


def _sequence_index_plot_grouping_ncol_not_1(data, categories, id_group_df,
                                             custom_order=None, age_labels=None,
                                             palette=None, reverse_colors=True, title=None,
                                             facet_ncol=2, save_as=None, dpi=200):
    """
    Plot sequence index plots with dynamic layout adjustments for better aesthetics.

    :param data: np.array or pd.DataFrame
        Sequence data matrix where rows are sequences and columns are time points.
    :param categories: list
        List of category labels corresponding to the data values.
    :param id_group_df: pd.DataFrame
        DataFrame containing IDs and grouping information for sequences.
    :param age_labels: list, optional
        Labels for the x-axis (e.g., ages or time points).
    :param palette: dict, optional
        Custom color palette mapping categories to colors.
    :param reverse_colors: bool, default=True
        Whether to reverse the color scheme.
    :param title: str, optional
        Title for the plot.
    :param facet_ncol: int, default=2
        Number of columns for grouped plots.
    :param save_as: str, optional
        File path to save the plot.
    :param dpi: int, default=200
        Resolution of the saved plot.
    """
    num_colors = len(categories)
    spectral_colors = sns.color_palette("Spectral", num_colors)
    if reverse_colors:
        spectral_colors = list(reversed(spectral_colors))
    cmap = ListedColormap(spectral_colors) if palette is None else ListedColormap(
        [palette.get(cat, '#000000') for cat in categories])

    group_column_name = id_group_df.columns[1]

    # Sort group labels using the new function
    group_labels = sort_group_labels(id_group_df[group_column_name].unique(), custom_order=custom_order)

    facet_nrow = int(np.ceil(len(group_labels) / facet_ncol))

    # Dynamic figure size calculation
    base_width_per_col = 6  # Base width per column
    base_height_per_row = 3  # Base height per row
    width = facet_ncol * base_width_per_col
    height = facet_nrow * base_height_per_row

    fig = plt.figure(figsize=(width, height))
    gs = plt.GridSpec(facet_nrow, facet_ncol + 1, figure=fig,
                      width_ratios=[*[1] * facet_ncol, 0.15],
                      wspace=0.4 + 0.1 * (facet_ncol - 2),  # Adjusted spacing for more columns
                      hspace=0.6)  # Consistent vertical spacing

    legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap(i / (num_colors - 1)))
                      for i in range(num_colors)]

    # Adjust legend position for facet_ncol
    legend_anchor_y = 0.8 if facet_ncol >= 4 else 0.7  # Move legend up for ncol=4
    legend_ax = fig.add_subplot(gs[:, -1])
    legend_ax.legend(legend_patches, categories,
                     loc='center left',
                     bbox_to_anchor=(0, legend_anchor_y))
    legend_ax.axis('off')

    for idx, group in enumerate(group_labels):

        ax = fig.add_subplot(gs[idx // facet_ncol, idx % facet_ncol])

        # Check for group existence in id_group_df
        if group not in id_group_df[group_column_name].values:
            print(f"Group '{group}' not found in id_group_df.")
            ax.axis('off')
            continue

        group_data = data[id_group_df[group_column_name] == group]
        if len(group_data) > 0:
            sorted_data = group_data[np.lexsort(group_data.T[::-1])]
            im = ax.imshow(sorted_data, aspect='auto', cmap=cmap, interpolation='nearest')

            # Use textual label from custom_order if provided
            if custom_order and group in custom_order:
                title_label = custom_order[group]
            else:
                title_label = group  # Fallback to numeric label if no custom order

            ax.set_title(f"{title_label} (N={len(group_data):,})", fontsize=10, pad=5, loc='center')

            if age_labels:
                ax.set_xticks(range(len(age_labels)))
                ax.set_xticklabels(age_labels, fontsize=8)
            else:
                ax.set_xticks(range(sorted_data.shape[1]))
                ax.set_xticklabels(range(1, sorted_data.shape[1] + 1), fontsize=8)

            num_yticks = min(10, sorted_data.shape[0])
            ytick_positions = np.linspace(0, sorted_data.shape[0] - 1, num_yticks, dtype=int)
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels([f"{int(x + 1):,}" for x in ytick_positions], fontsize=8)
        else:
            ax.axis('off')

        # Remove the spines (frame) and only keep the ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Adjust tick appearance
        ax.tick_params(axis='both', length=4, width=1, colors='black', labelsize=8)

        # Set consistent x and y axis labels with increased padding
        ax.set_xlabel('Age', fontsize=10, labelpad=15)
        ax.set_ylabel('Sequence ID', fontsize=10, labelpad=15)

    if title is not None:
        # Center title above subplots only
        subplot_width = facet_ncol / (facet_ncol + 0.15)  # Exclude the legend width
        fig.suptitle(title, fontsize=15, x=subplot_width / 2, y=0.94)  # Adjusted y for spacing

    if save_as:
        # Ensure the file extension is valid
        if not save_as.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            save_as += '.png'  # Default to .png if no valid extension provided

        # Crop the figure and save
        cropped_image = _crop_bottom_whitespace(fig)
        cropped_image.save(save_as)  # Save the cropped image
        plt.close()
    else:
        # Show the cropped figure
        cropped_image = _crop_bottom_whitespace(fig)
        cropped_image.show()  # This opens the image using the default viewer


def _sequence_index_plot_single(seqdata: SequenceData,
                                sortv="from.start",
                                age_labels=None,
                                figsize=(10, 6),
                                title=None,
                                xlabel="Time",
                                ylabel="Sequence Index",
                                save_as=None,
                                dpi=200):
    """
    Generate a single sequence index plot.

    :param seqdata: (SequenceData) A SequenceData object.
    """
    # Get sequence values as NumPy array
    sequence_values = seqdata.values.copy()

    # Ensure no NaN values interfere with sorting
    if np.isnan(sequence_values).any():
        sequence_values = np.where(np.isnan(sequence_values), -1, sequence_values)

    print("Sequence values before sorting:")
    print(sequence_values)

    # Sorting logic
    if sortv == "from.start":
        first_state = np.argmax(sequence_values != -1, axis=1)  # Find first valid state
        sorted_indices = np.argsort(first_state)  # Sort by first occurrence
    elif sortv == "lexical":
        sorted_indices = np.lexsort(sequence_values.T[::-1])  # Lexicographic sorting
    elif sortv == "from.end":
        last_state = np.argmax(sequence_values[:, ::-1] != -1, axis=1)
        sorted_indices = np.argsort(last_state)
    else:
        sorted_indices = np.arange(sequence_values.shape[0])  # No sorting

    print("Sorted indices:")
    print(sorted_indices)

    sorted_data = sequence_values[sorted_indices]
    print("Sequence values after sorting:")
    print(sorted_data)

    # Use colormap from SequenceData
    cmap = seqdata.get_colormap()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(sorted_data, aspect='auto', cmap=cmap, interpolation='nearest')

    # X-axis labels
    if age_labels:
        ax.set_xticks(range(len(age_labels)))
        ax.set_xticklabels(age_labels, fontsize=10, ha='right', color='black')
    else:
        ax.set_xticks(range(sorted_data.shape[1]))
        ax.set_xticklabels(range(1, sorted_data.shape[1] + 1), fontsize=10, ha='right', color='black')

    # Y-axis labels
    ax.set_yticks(range(0, len(sorted_data), max(1, len(sorted_data) // 10)))
    ax.set_yticklabels(range(1, len(sorted_data) + 1, max(1, len(sorted_data) // 10)), fontsize=10, color='black')

    # Remove unnecessary frame borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xlabel(xlabel, fontsize=12, labelpad=10, color='black')
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10, color='black')

    if title is not None:
        ax.set_title(title, fontsize=14, color='black')

    # Use legend from SequenceData
    ax.legend(*seqdata.get_legend(), bbox_to_anchor=(1.05, 1), loc='upper left')

    if save_as:
        plt.savefig(save_as, dpi=dpi)
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    import pandas as pd
    import pickle
    from seqdef import SequenceData
    from sequence_index_plots import sequence_index_plot

    df = pd.read_csv(
        '/test_data/real_data_my_paper/detailed_sequence_10_work_years_df.csv')

    unique_states_list = ['data', 'data & intensive math',
                          'hardware', 'research',
                          'software', 'software & hardware',
                          'support & test']

    sequence_columns_list = []

    for i in range(1, 11):
        sequence_columns_list.append(f'C{i}')

    sequence_data = SequenceData(data=df,
                                 var=sequence_columns_list,
                                 states=unique_states_list
                                 )

    sequence_index_plot(sequence_data,
                        sortv='lexical',
                        xlabel='Work year')


# if __name__ == '__main__':
#     # TODO jupyter notebook 版本调用这个api不太行，但是这里就没啥问题
#
#     original_df = pd.read_csv(
#         '/Users/lei/Library/CloudStorage/OneDrive-Nexus365/yuqi_dphil/output/occupational_segregation_paper/sequence_analysis/detailed_technical_computing_occupations/sequence_visualization/detailed_sequence_10_work_years_df.csv')
#
#     seqdef_path = '/test_data/real_data_my_paper/detailed_sequences_OM_trate/seqdef_38902_numpyData_cell.pkl'
#
#     with open(seqdef_path, 'rb') as file:
#         xinyi_original_seqdef = pickle.load(file)
#
#     xinyi_original_seqdef, state_mapping = preprocess_data(xinyi_original_seqdef)
#
#     membership_table = pd.read_csv('/test_results/detailed_sequences_OM_trate/membership_table.csv')
#
#     # TODO: 这里实在是太啰嗦了，要好好精简一下，这些步骤太复杂了，根本没有必要
#     # TODO：而且跟只有 xinyi_original_seqdef 的时候可视化，以及 cluster 完了以后可视化，同样用这可视化的函数，逻辑竟然对不上了
#     # Update categories to match string states
#     # unique_states = ['data', 'data & intensive math', 'hardware',
#     #               'research', 'software', 'software & hardware',
#     #               'support & test']
#     #
#     # state_mapping = {state: idx for idx, state in enumerate(unique_states)}
#     #
#     categories = list(state_mapping.keys())
#
#     # Assume the user has defined the name of each cluster
#     custom_order = {
#         1: 'Software focused',
#         2: 'Software and hardware focused',
#         3: 'Support and test focused',
#         4: 'Data focused',
#         5: 'Hardware focused',
#         6: 'Research focused',
#         7: 'Data and intensive math focused'
#     }
#
#     # Define age labels
#     time_points = 10
#
#     sequence_index_plot(xinyi_original_seqdef, categories,
#                         id_group_df=membership_table,
#                         custom_order=custom_order,
#                         age_labels=[f'{i}' for i in range(1, time_points + 1)],
#                         title=None,
#                         facet_ncol=3,
#                         save_as='detailed_sequences_sequence_index_plot')
#
#     # TODO: 明早起来再用R traiminer测试一下，保证整体走下来咱们结果都跟R是一致的
#     # TODO: 发现出图的cluster 各个顺序不是一致的，有点奇怪，需要调整，估计是按照 membership_table 来的，所以需要在图里面说
#     # TODO：要按照用户指定的顺序展示，比如 1-8
#
#     # original_df = pd.read_csv(
#     #     '/Users/lei/Library/CloudStorage/OneDrive-Nexus365/yuqi_dphil/output/occupational_segregation_paper/sequence_analysis/detailed_technical_computing_occupations/sequence_visualization/detailed_sequence_10_work_years_df.csv')
#     # file_path = '/Users/lei/Library/CloudStorage/OneDrive-Nexus365/sequenzo/test_data/real_data_my_paper/seqdef_38902_numpyData_cell.pkl'
#     #
#     # # Load data
#     # with open(file_path, 'rb') as file:
#     #     data = pickle.load(file)
#     #
#     # # Preprocess data to map strings to integers
#     # data, state_mapping = preprocess_data(data)
#     # # Update categories to match string states
#     # categories = list(state_mapping.keys())
#     # # Define age labels
#     # time_points = data.shape[1]
#
#     # For Single Plot:
#     # seq_index_plot(sub_data, categories,
#     #         sub_df,
#     #         title='Index Plot',
#     #         facet_ncol=2,
#     #         save_as='sequence_index_plot_full')
#
#     # For Grouped Plot when ncol=1:
#     # sequence_index_plot(data, categories,
#     #                     original_df,
#     #                     age_labels=[f'{i}' for i in range(1, time_points + 1)],
#     #                     title='Index Plot',
#     #                     facet_ncol=1,
#     #                     save_as='sequence_index_plot_1ncol')
#
#     # For Grouped Plot when ncol>1:
#     # sequence_index_plot(data, categories,
#     #         original_df,
#     #         age_labels=[f'{i}' for i in range(1, time_points + 1)],
#     #         title='Index Plot',
#     #         facet_ncol=4,
#     #         save_as='sequence_index_plot_4ncol.png')
