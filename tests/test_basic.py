"""
@Author  : Yuqi Liang 梁彧祺
@File    : test_basic.py
@Time    : 26/02/2025 13:13
@Desc    :
"""

# %%
import sequenzo.define_sequence_data import SequenceData
import pandas as pd

def test_version():
    assert sqz.__version__ is not None  # Ensure version is not empty

# %%
import sequenzo.define_sequence_data import SequenceData
from sequenzo.visualization import plot_transition_matrix, print_transition_matrix, compute_transition_matrix
# test_version()


# %%
test = pd.read_csv("nba_products_mx.csv")
test = test.dropna()

time = ['week_' + str(i) for i in range(0, 53)]

# %%

states = [0, 1, 10, 11, 100, 101, 110, 111]
labels = ['none', 'noa only', 'm360 only', 'm360 + noa', 'fc only', 'fc + noa', 'fc + m360', 'all']

# %%


# %% 
sq = sqz.SequenceData(
    data=test,
    time=time,
    states=states,
    id_col="id",
    labels=labels,
    start=0
)

# %%
sqz.plot_sequence_index(sq, title="Sequence Index Plot")
# %%
tt = compute_transition_matrix(sq)
# %%
print_transition_matrix(sq, tt)
# %%
plot_transition_matrix(sq)

# %%
