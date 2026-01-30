import pandas as pd
from sequenzo import *

df = pd.read_csv("/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/orignal data/not_real_detailed_data/synthetic_detailed_U5_N50000.csv")
_time = list(df.columns)[2:]
states = ["Data", "Data science", "Hardware", "Research", "Software", "Support & test", "Systems & infrastructure"]
df = df[['id', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]

sequence_data = SequenceData(df, time=_time, id_col="id", states=states)

# ================
# R 的 index plot
# ================

# R_membership_table = pd.read_csv("/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/output_CLARA_clustering/U5_5w_clustering_10000.csv")
#
# R_membership_table = R_membership_table.iloc[:, [0, 7]] # cluster = 8
# R_membership_table.columns = ['Entity ID', 'Cluster']
#
# plot_sequence_index(seqdata=sequence_data,
#                     group_dataframe=R_membership_table,
#                     group_column_name="Cluster",
#                     # group_labels=cluster_labels,
#                     save_as='CLARA_index_plot'
#                     )

# ======================
# sequenzo 的 index plot
# =====================
membership_table = pd.read_csv("/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/output_CLARA_clustering/my_U5_5w_clustering_1w.csv")

membership_table = membership_table.iloc[:, [6]]    # cluster = 8
membership_table.columns = ['Cluster']
membership_table['Entity ID'] = sequence_data.ids

plot_sequence_index(seqdata=sequence_data,
                    group_dataframe=membership_table,
                    group_column_name="Cluster",
                    # group_labels=cluster_labels,
                    save_as='my_CLARA_index_plot'
                    )
