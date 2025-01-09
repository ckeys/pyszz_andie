import pandas as pd
import json


def read_szz_output(input_file):
    # Step 1: Read the JSON Lines file
    def read_json_lines(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)

    # Step 2: Load the data into a DataFrame
    df = read_json_lines(input_file)

    # Step 3: Retain only 'fix_commit_hash' and 'inducing_commit_hash' columns
    df = df[['fix_commit_hash', 'inducing_commit_hash']]

    df['inducing_commit_hash'] = df['inducing_commit_hash'].apply(lambda x: x if x else [None])
    df_exploded = df.explode('inducing_commit_hash').reset_index(drop=True)
    new_df = df_exploded[['inducing_commit_hash']].copy()
    szz_buggy_commit = new_df.dropna(subset=['inducing_commit_hash']).reset_index(drop=True)
    return szz_buggy_commit

def read_mlszz_output(input_file):
    mlszz_output_df = pd.read_csv(input_file)
    mlszz_buggy_commit = mlszz_output_df[['inducing_commit_hash', 'MLSZZ_BUGGY']]
    mlszz_buggy_commit = mlszz_buggy_commit[mlszz_buggy_commit['MLSZZ_BUGGY'] == 1]
    return mlszz_buggy_commit

if __name__ == '__main__':
    # Define the input and output file paths
    input_file_dir = '/Users/andie/PycharmProjects/pyszz_andie/defect_prediction/data'  # Replace with your actual file path
    # output_file = '/path/to/your/react_predictions.csv'  # Replace with your desired output path
    lszz_buggy_commit = read_szz_output(f'''{input_file_dir}/react_bic_l.json''')
    rszz_buggy_commit = read_szz_output(f'''{input_file_dir}/react_bic_r.json''')
    maszz_buggy_commit = read_szz_output(f'''{input_file_dir}/react_bic_ma.json''')
    mlszz_output_file = '/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/react/react_mlszz_output_predictions.csv'
    mlszz_output_df = pd.read_csv(mlszz_output_file)
    mlszz_buggy_commit = mlszz_output_df[['inducing_commit_hash', 'MLSZZ_BUGGY']]

