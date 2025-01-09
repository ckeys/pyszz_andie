import pandas as pd

# Input and output file paths
input_file = '/mlszz_model/data/benchmark/total.csv'
output_file = '/mlszz_model/data/benchmark/preprocessed_total.csv'

# Step 1: Load the CSV file
df = pd.read_csv(input_file)

# Step 2: Labelling - Add a new column based on commit comparison
df['label'] = (df['commit'] == df['bug_commit_hash']).astype(int)

# Step 3: Drop specified columns
columns_to_drop = ['repo_name', 'id', 'fix_commit_hash', 'best_scenario_issue_date',
                   'language', 'inducing_commit_hash', 'modified_files']
df = df.drop(columns=columns_to_drop)

# Step 4: Save the preprocessed DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Preprocessed data saved to {output_file}")