"""
@Author  : Yuqi Liang 梁彧祺
@File    : qiqi_test_OM.py
@Time    : 30/03/2025 22:45
@Desc    : 
"""
# Import necessary libraries
# Your calling code (e.g., in a script or notebook)

from sequenzo import * # Import the package, give it a short alias
import pandas as pd # Data manipulation

# List all the available datasets in Sequenzo
# Now access functions using the alias:
print('Available datasets in Sequenzo: ', list_datasets())

# Load the data that we would like to explore in this tutorial
# `df` is the short for `dataframe`, which is a common variable name for a dataset
df = load_dataset('country_co2_emissions')

# Create a SequenceData object

# Define the time-span variable
time_list = list(df.columns)[1:]

states = ['Very Low', 'Low', 'Middle', 'High', 'Very High']

# TODO: write a try and error: if no such a parameter, then ask to pass the right ones
# sequence_data = SequenceData(df, time=time, time_type="year", id_col="country", ids=df['country'].values, states=states)

sequence_data = SequenceData(df, time=time_list, time_type="year", id_col="country", states=states)

om = get_distance_matrix(seqdata=sequence_data,
                         method='OM',
                         sm="TRATE",
                         indel="auto")
print(om)