import pandas as pd
from pathlib import Path
import os
# Get the directory of the current file
current_folder = Path(__file__).parent
# Get the upper-level folder
upper_folder = current_folder.parent

#
project = 'linux'
# # File paths
git_features_path = f"{current_folder}/data/{project}/{project}_jit_features.csv"
bug_inducing_commits_features_path = f'''{upper_folder}/bic-fea-collection/output/features/{project}_inducing_commits_features.csv'''
# Read the CSV files
git_features = pd.read_csv(git_features_path)
bic_features = pd.read_csv(bug_inducing_commits_features_path)

# Perform a left join
merged_data = bic_features.merge(
    git_features,
    how="left",
    left_on="inducing_commit_hash",  # Replace with the column name in the left table
    right_on="commit_id"  # Replace with the column name in the right table,
)
merged_data = merged_data.drop(columns=["commit_id"])
# columns = ["project"] + [col for col in merged_data.columns if col != "project"]
# merged_data = merged_data[columns]

# Display the merged data
print(merged_data)

# Optionally save the result to a new CSV file
merged_data.to_csv(f"{current_folder}/data/{project}/bszz_merged_{project}_features.csv", index=False)