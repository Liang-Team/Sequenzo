import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 绘制折线图
def getPlot(data, save_path, x_label):
    df = pd.DataFrame(data)

    # Plot setup
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Plotting both R and Python data
    # #9ca6c1
    # #edb6a9
    plt.plot(df['datasize'], df['R'], marker='o', label='R TraMineR', color='#acbfeb', linewidth=2, markersize=8)
    plt.plot(df['datasize'], df['python'], marker='s', label='Python Sequenzo', color='#ffadbb', linewidth=2,
             markersize=8)

    # Title and labels with more padding
    plt.xlabel(x_label, fontsize=14, labelpad=35)
    plt.ylabel('Execution Time (seconds)', fontsize=14, labelpad=35)

    # Adding a legend with the specified caption
    plt.legend(title="Legend", title_fontsize='13', loc='upper left')

    # Adding a grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjusting layout to ensure there's no clipping of labels or titles
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # plt.show()

# 星云打算要保存图片的位置
root = 'D:\\college\\research\\QiQi\\sequenzo\\data_and_output\\output_runtime\\250917\\broad\\'    # 最后的'\\'不能省略！

# L：large，S：small
# P：Python，R：R

OM_L_P = [0.1634, 0.5144, 1.0909, 1.7923, 2.8382, 3.9170, 5.0936, 12.6564, 16.3761, 29.9426, 44.0124]
OMspell_L_P = [0.1903, 0.5693, 1.1992, 1.8981, 2.8195, 3.9448, 5.2859, 13.2945, 17.5147, 30.7114, 46.6869]
HAM_L_P = [0.1635, 0.5429, 1.0840, 1.7906, 2.7179, 3.5864, 5.1354, 11.5371, 17.5572, 29.3489, 46.0097]
DHD_L_P = [0.1656, 0.5324, 1.1068, 1.8925, 2.6625, 3.5881, 5.2072, 11.8355, 18.4596, 31.1915, 46.0167]

OM_S_P = [0.0515, 0.0567, 0.0626, 0.0662, 0.0779, 0.0905, 0.1049, 0.1237, 0.1427, 0.1642]
OMspell_S_P = [0.0509, 0.0605, 0.0688, 0.0827, 0.0951, 0.1091, 0.1231, 0.1470, 0.1663, 0.1896]
HAM_S_P = [0.0511, 0.0541, 0.0586, 0.0720, 0.0760, 0.0887, 0.1084, 0.1263, 0.1420, 0.1648]
DHD_S_P = [0.0515, 0.0536, 0.0605, 0.0711, 0.0781, 0.0915, 0.1037, 0.1253, 0.1410, 0.1612]


P_L_list = [OM_L_P, OMspell_L_P, HAM_L_P, DHD_L_P]
P_S_list = [OM_S_P, OMspell_S_P, HAM_S_P, DHD_S_P]

OM_L_R = [1.6130, 1.4983, 2.7857, 4.8201, 9.5889, 21.7180, np.nan, np.nan, np.nan, np.nan, np.nan]
OMspell_L_R = [1.4329, 1.7171, 3.0430, 4.8429, 9.8204, 16.8022, np.nan, np.nan, np.nan, np.nan, np.nan]
HAM_L_R = [1.5158, 1.4584, 2.6418, 4.5744, 8.9954, 15.3799, np.nan, np.nan, np.nan, np.nan, np.nan]
DHD_L_R = [1.4195, 1.4849, 2.6368, 4.6087, 9.0473, 16.6878, np.nan, np.nan, np.nan, np.nan, np.nan]

OM_S_R = [0.3917, 0.4013, 0.4269, 0.4311, 0.4599, 0.4928, 0.5903, 0.9063, 0.9134, 1.4874]
OMspell_S_R = [0.3866, 0.4183, 0.4478, 0.4526, 0.5210, 0.5682, 0.6355, 1.0184, 1.0340, 1.4734]
HAM_S_R = [0.4695, 0.4898, 0.5224, 0.5626, 0.6091, 0.6251, 0.7628, 1.3300, 1.4270, 1.4828]
DHD_S_R = [0.4702, 0.5019, 0.5195, 0.5447, 0.6097, 0.6275, 0.7215, 1.1694, 1.2267, 1.5629]

large_size = ['0.5', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5', '5.5']
small_size = ['500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000']

R_L_list = [OM_L_R, OMspell_L_R, HAM_L_R, DHD_L_R]
R_S_list = [OM_S_R, OMspell_S_R, HAM_S_R, DHD_S_R]

# 要保存的文件名
title_L = [root+'large_broad_OM_runtime_vs_datasize.jpg',
         root+'large_broad_OMspell_runtime_vs_datasize.jpg',
         root+'large_broad_HAM_runtime_vs_datasize.jpg',
         root+'large_broad_DHD_runtime_vs_datasize.jpg']
title_S = [root+'small_broad_OM_runtime_vs_datasize.jpg',
         root+'small_broad_OMspell_runtime_vs_datasize.jpg',
         root+'small_broad_HAM_runtime_vs_datasize.jpg',
         root+'small_broad_DHD_runtime_vs_datasize.jpg']

for python, R, output in zip(P_L_list, R_L_list, title_L):
    data_broad = {
                    'datasize': large_size,
                    'python': python,
                    'R': R,
                 }
    getPlot(data_broad, output, x_label='Data Size (10,000)')   # 我改了参数设置，这样星云就不用手动改标签了
    
for python, R, output in zip(P_S_list, R_S_list, title_S):
    data_broad = {
                    'datasize': small_size,
                    'python': python,
                    'R': R,
                 }
    getPlot(data_broad, output, x_label='Data Size')