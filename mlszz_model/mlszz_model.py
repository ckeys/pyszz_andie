import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import pickle
import joblib

# File path
input_file = '/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/preprocessed_total.csv'

# Load the preprocessed data
df = pd.read_csv(input_file)

# Step 1: Remove 'bug_commit_hash' and 'commit' columns
df = df.drop(columns=['bug_commit_hash', 'commit'])
retain_columns = [
    'is_Friday','is_latest_bic','is_earliest_bic','is_largest_mod',
    'candidate_commit_to_fix','lines_of_modified_code','ns','nd','nf',
    'entropy','exp','rexp','sexp','ndev','age','nuc','fix','la','ld','lt','label'
]
df = df[retain_columns]

# Step 2: Separate features and target variable
X = df.drop(columns=['label'])
y = df['label']

# Step 3: Initialize k-fold cross-validation (Stratified for classification)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Metrics initialization
auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
feature_importances = []

# Hyperparameter tuning (optional but recommended)
# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define scoring metrics
scoring = {
    'AUC': 'roc_auc',
    'Precision': make_scorer(precision_score),
    'Recall': make_scorer(recall_score),
    'F1': make_scorer(f1_score)
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=kf,
    scoring='roc_auc',  # Optimize for AUC
    n_jobs=-1,
    verbose=2
)

# Step 4: Perform hyperparameter tuning with cross-validation
print("Starting Grid Search for Hyperparameter Tuning...")
grid_search.fit(X, y)

# Best parameters from Grid Search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Initialize the model with best parameters
best_rf = RandomForestClassifier(**best_params, random_state=42)

# Re-initialize cross-validation with StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Step 5: Perform k-fold cross-validation with the best model
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    best_rf.fit(X_train, y_train)

    # Make predictions
    y_pred = best_rf.predict(X_test)
    y_pred_prob = best_rf.predict_proba(X_test)[:, 1]  # Probability scores for AUC
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    best_threshold = thresholds[np.argmax(recall)]
    print(f"Best Threshold for Highest Recall: {best_threshold:.4f}")
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append metrics
    auc_scores.append(auc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Record feature importance
    feature_importances.append(best_rf.feature_importances_)

# Average feature importance across all folds
avg_feature_importance = pd.Series(np.mean(feature_importances, axis=0), index=X.columns)

# Step 6: Print average metrics
print("\nModel Evaluation Metrics (10-fold cross-validation):")
print(f"AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# Step 7: Display feature importance
print("\nFeature Importance (Average across folds):")
print(avg_feature_importance.sort_values(ascending=False))

# Optional: Plot feature importance
plt.figure(figsize=(10, 8))
avg_feature_importance.sort_values(ascending=True).plot.barh()
plt.title('Average Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Step 8: Train final model on the entire dataset with best parameters
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X, y)

# Step 9: Save the trained model using pickle or joblib
# Using pickle
with open('final_random_forest_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)
print("\nModel saved as 'final_random_forest_model.pkl'.")

# Alternatively, using joblib (often faster for large models)
joblib.dump(final_model, 'final_random_forest_model.joblib')
print("Model saved as 'final_random_forest_model.joblib'.")
'''
Model Evaluation Metrics (10-fold cross-validation):
AUC: 0.9018 ± 0.0289
Precision: 0.7235 ± 0.0390
Recall: 0.7136 ± 0.0359
F1-Score: 0.7179 ± 0.0300

Feature Importance (Average across folds):
is_latest_bic              0.331668
la                         0.069801
lines_of_modified_code     0.061345
candidate_commit_to_fix    0.060445
age                        0.054925
lt                         0.051488
nuc                        0.040895
rexp                       0.040288
exp                        0.039220
ld                         0.038205
is_largest_mod             0.038024
sexp                       0.037124
ndev                       0.035131
nf                         0.022710
entropy                    0.022329
is_earliest_bic            0.020221
nd                         0.012582
fix                        0.011029
ns                         0.008641
is_Friday                  0.003930
dtype: float64

Model saved as 'final_random_forest_model.pkl'.
Model saved as 'final_random_forest_model.joblib'.
'''
