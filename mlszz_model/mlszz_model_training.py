import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# File path
input_file = '/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/benchmark/preprocessed_total.csv'

# Load the preprocessed data
df = pd.read_csv(input_file)

# Step 1: Remove 'bug_commit_hash' and 'commit' columns
df = df.drop(columns=['bug_commit_hash', 'commit'])

# Step 2: Separate features and target variable
X = df.drop(columns=['label'])
y = df['label']

# Step 3: Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Metrics initialization
auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
feature_importances = []

# Step 4: Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability scores for AUC

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
    feature_importances.append(model.feature_importances_)

# Average feature importance across all folds
avg_feature_importance = pd.Series(sum(feature_importances) / len(feature_importances), index=X.columns)

# Step 5: Print average metrics
print("Model Evaluation Metrics (5-fold cross-validation):")
print(f"AUC: {sum(auc_scores) / len(auc_scores):.4f}")
print(f"Precision: {sum(precision_scores) / len(precision_scores):.4f}")
print(f"Recall: {sum(recall_scores) / len(recall_scores):.4f}")
print(f"F1-Score: {sum(f1_scores) / len(f1_scores):.4f}")

# Step 6: Display feature importance
print("\nFeature Importance:")
print(avg_feature_importance.sort_values(ascending=False))

# # Step 7: Plot feature importance
# plt.figure(figsize=(10, 6))
# avg_feature_importance.sort_values(ascending=False).plot(kind='bar')
# plt.title('Feature Importance')
# plt.ylabel('Importance Score')
# plt.xlabel('Features')
# plt.tight_layout()
# plt.show()