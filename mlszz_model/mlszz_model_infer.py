import pandas as pd
import pickle  # or import joblib
import numpy as np
# import joblib

project = 'tor_1'
# Load the saved model
with open('final_random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Alternatively, using joblib
# loaded_model = joblib.load('final_random_forest_model.joblib')
retain_columns = ['is_Friday', 'is_latest_bic', 'is_earliest_bic', 'is_largest_mod', 'candidate_commit_to_fix',
                  'lines_of_modified_code', 'ns', 'nd', 'nf', 'entropy', 'exp', 'rexp', 'sexp', 'ndev', 'age', 'nuc',
                  'fix', 'la', 'ld', 'lt', 'label']
# Prepare new data (ensure it has the same features as the training data)
new_data = pd.read_csv(f'/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/{project}/bszz_merged_{project}_features.csv')
new_data.dropna(inplace=True)
new_data_features = new_data[retain_columns[:-1]]  # Exclude the 'label' column

# Make predictions
predictions = loaded_model.predict(new_data_features)
prediction_probabilities = loaded_model.predict_proba(new_data_features)[:, 1]  # Probability for class '1'
# react = 0.1
# bitcoin = 0.1
predictions = (prediction_probabilities >= 0.5).astype(int)

# Merge predictions with the original new_data
# Option 1: Add predictions as new columns to new_data
new_data_with_predictions = new_data.copy()
new_data_with_predictions['MLSZZ_BUGGY'] = predictions
new_data_with_predictions['Probability'] = prediction_probabilities

# Option 2: Concatenate the predictions DataFrame with new_data
# predictions_df = pd.DataFrame({
#     'Prediction': predictions,
#     'Probability': prediction_probabilities
# })
# new_data_with_predictions = pd.concat([new_data.reset_index(drop=True), predictions_df], axis=1)

# Save the combined DataFrame to a new CSV file
output_file_path = f'/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/{project}/{project}_mlszz_output_predictions.csv'
new_data_with_predictions.to_csv(output_file_path, index=False)

print(f"Predictions saved to '{output_file_path}'.")