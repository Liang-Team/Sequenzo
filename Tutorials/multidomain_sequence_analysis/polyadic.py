"""
@Author  : Yuqi Liang 梁彧祺
@File    : polyadic.py
@Time    : 30/04/2025 10:39
@Desc    :

    ObservedDist:
    The average observed pairwise distance between sequences in each polyad (e.g. child + parents),
    based on the specified distance method (e.g. "HAM").
    This measures actual similarity within the group.

    U:
    The difference between the mean randomized distance across T simulations and the observed distance:
    U = mean_random_distance - observed_distance.

    V:
    The proportion of randomized distances greater than the observed distance.
    V = P(D_random > D_observed) over the T randomizations.
    Values near 1.0 suggest that the observed sequences are much more similar than random groupings.

    V>0.95
    A binary indicator (0 or 1):
    if V > 0.95, meaning the similarity in the polyad is statistically significant at the 5% level.
    Useful as a flag for identifying meaningful linked lives.

    Example interpretation:
        ObservedDist: 18.0
        U: -1.43
        V: 0.430
        V>0.95: 0

        This polyad has:
        1. An observed average distance of 18.0
        2. U is negative → its sequences are less similar than expected
        3. V is 0.43 → 43% of random pairings are more dissimilar → not statistically significant
        4. V>0.95 = 0 → no strong linked life pattern

    # TODO: Last row of the result_df is null in gender.
"""
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
    labels=["S", "M0", "M1", "M2", "M3+", "D"],
    id_col="id",
    weights=None
)

seq_parent = SequenceData(
    data=df_parent,
    time_type="age",
    time=[str(i) for i in range(15, 40)],
    states=["S", "M0", "M1", "M2", "M3+", "D"],
    labels=["S", "M0", "M1", "M2", "M3+", "D"],
    id_col="id",
    weights=None
)

# 3. Run linked_polyad analysis
result_df, merged_seq_data = linked_polyadic_sequence_analysis(
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

# 5. Plot by gender (assuming you have a sample_df that contains sex information)
df_samplec1 = load_dataset("polyadic_samplec1")  # Contains 'sex' column (0 = female, 1 = male)

# Add gender information
result_df["Sex"] = df_samplec1["sex"].map({0: "Female", 1: "Male"})

print("Number of significant polyads (V > 0.95):", result_df["V>0.95"].sum())

result_df.to_csv('polyadic_result_df.csv', index=False)

# 6. Plot density of U and V by gender

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.kdeplot(data=result_df, x="U", hue="Sex", ax=axs[0])
axs[0].set_title("U Density by Gender")

sns.kdeplot(data=result_df, x="V", hue="Sex", ax=axs[1])
axs[1].set_title("V Density by Gender")

plt.tight_layout()
plt.savefig("uv_density_by_gender.png", dpi=200)  # Save the figure
plt.show()

