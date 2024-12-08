import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data")
input_csv = os.path.join(data_dir, "repo_metadata.csv")

# Read the metadata from the CSV file
metadata_df = pd.read_csv(input_csv)

# Calculate the required statistics
stats = metadata_df.describe().transpose()

# Extract the necessary statistics
stats_table = stats[['25%', '50%', '75%', 'mean']].rename(columns={
    '25%': 'Q1',
    '50%': 'Median',
    '75%': 'Q3',
    'mean': 'Average'
})

# Display the table
print(stats_table)