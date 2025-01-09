import pandas as pd

project = 'bitcoin'
# File paths
react_git_features_path = f"/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/{project}/{project}_git_features.csv"
facebook_react_inducing_commits_features_path = f"/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/{project}/{project}_bszz_inducing_commits_features.csv"

# Read the CSV files
git_features = pd.read_csv(react_git_features_path)
bic_features = pd.read_csv(facebook_react_inducing_commits_features_path)

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
merged_data.to_csv(f"/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/{project}/bszz_merged_{project}_features.csv", index=False)