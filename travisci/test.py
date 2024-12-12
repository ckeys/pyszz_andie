import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/andie/PycharmProjects/pyszz_andie/travisci/data/build_level_data.csv"
df = pd.read_csv(file_path)

# Filter the data for build_status == "failed" and create a copy
filtered_df = df[df['build_status'] == 'failed'].copy()

# Label the data based on 'build_quadrant'
filtered_df['label'] = filtered_df['build_quadrant'].apply(lambda x: 1 if x == 'long_broken' else 0)

# Encode 'commit_day_night' and 'build_day_night' into numerical values
filtered_df['commit_day_night'] = filtered_df['commit_day_night'].map({'day': 0, 'night': 1})
filtered_df['build_day_night'] = filtered_df['build_day_night'].map({'day': 0, 'night': 1})

# Drop unnecessary columns
columns_to_drop = [
    'build_status', 'build_quadrant', 'tr_build_id',
    'gh_repository_name', 'gh_lang', 'gh_build_started_at',
    'git_trigger_commit'
]
X = filtered_df.drop(columns=columns_to_drop + ['label'])
y = filtered_df['label']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns  # Includes 'dist'
numerical_features = X.select_dtypes(exclude=['object']).columns

# Debugging: Print columns for verification
print("Categorical Features:", categorical_features.tolist())
print("Numerical Features:", numerical_features.tolist())

# Preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-hot encode categorical variables
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))  # Fill missing numerical values with mean
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a pipeline with preprocessing and classifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform cross-validation
try:
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())
except ValueError as e:
    print("Error during cross-validation:", e)
    print("Check for non-numeric data in the pipeline or data preparation steps.")
# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Predict probabilities for AUC calculation
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_text = classification_report(y_test, y_pred)

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)

# Display results
print("Test set accuracy:", accuracy)
print("Classification report:\n", classification_report_text)
print("AUC Score:", auc_score)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Count the occurrences of each label
label_counts = filtered_df['label'].value_counts()

# Calculate percentages
label_percentages = filtered_df['label'].value_counts(normalize=True) * 100

# Display results
print("Label Distribution:")
print(label_counts)
print("\nLabel Distribution (Percentages):")
print(label_percentages)

# Optional: Visualize the distribution
plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar')
plt.title("Distribution of Labels (0 and 1)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()


# Extract feature names after preprocessing
def get_feature_names(preprocessor, X):
    # Get numerical feature names
    num_features = numerical_features.tolist()

    # Get categorical feature names after one-hot encoding
    cat_feature_indices = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    cat_features = cat_feature_indices.tolist()

    return num_features + cat_features


# Get feature names from the preprocessor
feature_names = get_feature_names(preprocessor, X)

# Get feature importances from the Random Forest classifier
feature_importances = clf.named_steps['classifier'].feature_importances_

# Create a DataFrame for better visualization
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Print feature importance
print("\nFeature Importance:")
print(importances_df)

# Optional: Plot the feature importance
plt.figure(figsize=(10, 6))
importances_df.set_index('Feature').plot(kind='bar', legend=False, figsize=(12, 8))
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.tight_layout()
plt.show()