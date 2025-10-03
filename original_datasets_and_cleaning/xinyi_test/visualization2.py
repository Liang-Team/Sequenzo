import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def getPlot_unique(data_dict, save_path):
    x = [3.5, 4, 4.5, 5]
    df = pd.DataFrame(data_dict, index=x)

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    colors = ['#5698C3', '#FCA104', '#9c7ab5', '#92a470']

    for (label, series), color in zip(df.items(), colors):
        plt.plot(df.index, series, marker='o', label=label,
                 color=color, linewidth=2, markersize=8)

    plt.xlabel('Data Size (10,000)', fontsize=14, labelpad=20)
    plt.ylabel('Execution Time (seconds)', fontsize=14, labelpad=20)

    plt.xticks([3.5, 4, 4.5, 5])

    plt.legend(title="Uniqueness Rate", title_fontsize=13, loc='upper left')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 固定参数（OM, TRATE, 1），对比 3.5w 4w 4.5w 5w 的数据规模下，不同 U 的时间
U5 = [7.0133, 15.2674, 22.4451, 39.1537]
U25 = [34.2809, 48.4953, 65.1049, 102.3304]
U50 = [61.0589, 87.9582, 115.8026, 179.0892]
U85 = [187.1809, 273.1139, np.nan, np.nan]

data_dict = {
    "U=5%": U5,
    "U=25%": U25,
    "U=50%": U50,
    "U=85%": U85
}

getPlot_unique(data_dict, "D:\\college\\research\\QiQi\\sequenzo\\data_and_output\\output_runtime\\250917\\unique_rate_runtime.jpg")