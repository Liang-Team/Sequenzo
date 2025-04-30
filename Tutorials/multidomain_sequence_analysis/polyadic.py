import pandas as pd
from sequenzo import *
import seaborn as sns
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 1. Load child and parent sequence data (assuming you have processed DataFrames from LSOG or other sources)
df_child = load_dataset("polyadic_seqc1")
df_child["id"] = range(1, len(df_child) + 1)

df_parent = load_dataset("polyadic_seqp1")
df_parent["id"] = range(1, len(df_parent) + 1)

# 2. Create SequenceData objects (including id, state dictionary, etc.)
seq_child = SequenceData(
    data=df_child,
    time_type="age",
    time=[str(i) for i in range(15, 40)],
    states=["S", "M0", "M1", "M2", "M3+", "D"],
    id_col="id",
    weights=None
)

seq_parent = SequenceData(
    data=df_parent,
    time_type="age",
    time=[str(i) for i in range(15, 40)],
    states=["S", "M0", "M1", "M2", "M3+", "D"],
    id_col="id",
    weights=None
)

# 3. Run linked_polyad analysis
result_df = linked_polyadic_sequence_analysis(
    seqlist=[seq_child, seq_parent],
    # method="OM",  # or "HAM", "OMspell", "CHI2"
    method="HAM",
    # distance_parameters={"sm": "CONSTANT", "indel": 1},
    a=1,  # Random pairing by sequence
    T=1500,
    random_seed=123,
    n_jobs=4,
    return_df=True  # Return a DataFrame directly
)

# 4. View results
print(result_df.head())

# 5. Optional: Plot by gender (assuming you have a sample_df that contains sex information)
df_samplec1 = load_dataset("polyadic_samplec1")  # Contains 'sex' column (0 = female, 1 = male)

# Add gender information
result_df["Sex"] = df_samplec1["sex"].values

# 6. Plot density of U and V by gender

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.kdeplot(data=result_df, x="U", hue="Sex", ax=axs[0])
axs[0].set_title("U Density by Gender")

sns.kdeplot(data=result_df, x="V", hue="Sex", ax=axs[1])
axs[1].set_title("V Density by Gender")

plt.tight_layout()
plt.savefig("uv_density_by_gender.png", dpi=200)  # Save the figure
plt.show()

