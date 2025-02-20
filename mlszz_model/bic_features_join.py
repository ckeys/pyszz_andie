import pandas as pd

project = 'innoldb'
# File paths
git_features_path = f"/home/huayo708/andie/pyszz_andie/mlszz_model/data/{project}/{project}_jit_features.csv"
bug_inducing_commits_features_path = f'''/home/huayo708/andie/pyszz_andie/bic-fea-collection/output/features/{project}_inducing_commits_features.csv'''
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
merged_data.to_csv(f"/home/huayo708/andie/pyszz_andie/mlszz_model/data/{project}/bszz_merged_{project}_features.csv", index=False)