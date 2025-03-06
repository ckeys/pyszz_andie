import pandas as pd
from defect_prediction.szz_json_data_conversion import read_szz_output, read_mlszz_output, read_bug_commits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler


def plot_class_distribution(distribution, label_name, fold_number):
    """Plots the class distribution."""
    classes = list(distribution.keys())
    counts = list(distribution.values())

    sns.barplot(x=classes, y=counts)
    plt.title(f'Class Distribution for {label_name} - Fold {fold_number}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['False', 'True'])
    plt.show()


# Function to read and process each project's data
def process_project(project_name):
    # Base directory paths
    data_dir = '/Users/andie/PycharmProjects/pyszz_andie/defect_prediction/data'
    mlszz_dir = '/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data'

    input_file_dir = f"{data_dir}/{project_name}"

    # Read buggy commit lists
    lszz_buggy_commit = read_szz_output(f"{input_file_dir}/{project_name}_bic_l.json")
    rszz_buggy_commit = read_szz_output(f"{input_file_dir}/{project_name}_bic_r.json")
    maszz_buggy_commit = read_szz_output(f"{input_file_dir}/{project_name}_bic_ma.json")
    mlszz_buggy_commit = read_mlszz_output(f"{mlszz_dir}/{project_name}/{project_name}_mlszz_output_predictions.csv")
    bug_commit_df = read_bug_commits(f"{input_file_dir}/{project_name}_bug_commit.csv")

    # Convert lists to sets for fast lookup
    lszz_set = set(lszz_buggy_commit['inducing_commit_hash'])
    rszz_set = set(rszz_buggy_commit['inducing_commit_hash'])
    maszz_set = set(maszz_buggy_commit['inducing_commit_hash'])
    mlszz_set = set(mlszz_buggy_commit['inducing_commit_hash'])
    bug_commits_set = set(bug_commit_df['hash'])

    # Read JIT features
    features_file = f"{mlszz_dir}/{project_name}/{project_name}_jit_features.csv"
    df_features = pd.read_csv(features_file)

    # Create boolean flags
    df_features['LSZZ_BUGGY'] = df_features['commit_id'].isin(lszz_set)
    df_features['RSZZ_BUGGY'] = df_features['commit_id'].isin(rszz_set)
    df_features['MASZZ_BUGGY'] = df_features['commit_id'].isin(maszz_set)
    df_features['MLSZZ_BUGGY'] = df_features['commit_id'].isin(mlszz_set)
    df_features['BUGGY'] = df_features['commit_id'].isin(bug_commits_set)

    # Ensure boolean data type
    for col in ['LSZZ_BUGGY', 'RSZZ_BUGGY', 'MASZZ_BUGGY', 'MLSZZ_BUGGY', 'BUGGY']:
        df_features[col] = df_features[col].astype(bool)

    # Add project_name column for identification
    df_features['project_name'] = project_name

    return df_features


if __name__ == '__main__':
    project_name = "All"
    # input_file_dir = f'/Users/andie/PycharmProjects/pyszz_andie/defect_prediction/data/{project_name}'  # Replace with your actual file path
    # # output_file = '/path/to/your/react_predictions.csv'  # Replace with your desired output path
    # lszz_buggy_commit = read_szz_output(f'''{input_file_dir}/{project_name}_bic_l.json''')
    # rszz_buggy_commit = read_szz_output(f'''{input_file_dir}/{project_name}_bic_r.json''')
    # maszz_buggy_commit = read_szz_output(f'''{input_file_dir}/{project_name}_bic_ma.json''')
    # mlszz_output_file = f'/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/{project_name}/{project_name}_mlszz_output_predictions.csv'
    # mlszz_buggy_commit = read_mlszz_output(mlszz_output_file)
    # bug_commits_file = f'/Users/andie/PycharmProjects/pyszz_andie/defect_prediction/data/{project_name}/{project_name}_bug_commit.csv'
    # bug_commit_df = read_bug_commits(bug_commits_file)
    #
    # # Create sets for faster lookup
    # lszz_set = set(lszz_buggy_commit['inducing_commit_hash'])
    # rszz_set = set(rszz_buggy_commit['inducing_commit_hash'])
    # maszz_set = set(maszz_buggy_commit['inducing_commit_hash'])
    # mlszz_set = set(mlszz_buggy_commit['inducing_commit_hash'])
    # bug_commits_set = set(bug_commit_df['hash'])
    # # /Users/andie/PycharmProjects/pyszz_andie/defect_prediction/data/bitcoin/bitcoin_bic_l.json
    #
    # # Alternatively, if 'inducing_commit_hash' contains lists, flatten them first
    # def flatten_list_column(df, column):
    #     return set([item for sublist in df[column] for item in sublist])
    #
    # # Define the path to the main features CSV
    # features_file = f'/Users/andie/PycharmProjects/pyszz_andie/mlszz_model/data/{project_name}/{project_name}_jit_features.csv'
    #
    # # Read the main features DataFrame
    # df_features = pd.read_csv(features_file)
    # # Now, create the boolean flags
    # df_features['LSZZ_BUGGY'] = df_features['commit_id'].isin(lszz_set)
    # df_features['RSZZ_BUGGY'] = df_features['commit_id'].isin(rszz_set)
    # df_features['MASZZ_BUGGY'] = df_features['commit_id'].isin(maszz_set)
    # df_features['MLSZZ_BUGGY'] = df_features['commit_id'].isin(mlszz_set)
    # df_features['BUGGY'] = df_features['commit_id'].isin(bug_commits_set)
    #
    # # Convert boolean flags to True/False (optional, as they are already boolean)
    # df_features['LSZZ_BUGGY'] = df_features['LSZZ_BUGGY'].astype(bool)
    # df_features['RSZZ_BUGGY'] = df_features['RSZZ_BUGGY'].astype(bool)
    # df_features['MASZZ_BUGGY'] = df_features['MASZZ_BUGGY'].astype(bool)
    # df_features['MLSZZ_BUGGY'] = df_features['MLSZZ_BUGGY'].astype(bool)
    # df_features['BUGGY'] = df_features['BUGGY'].astype(bool)
    # df_features.to_csv('test_data.csv', index=False)
    # # Display the first few rows to verify
    # print("Main Features DataFrame:")
    # print(df_features)
    # Initialize an empty list to store DataFrames
    all_features = []
    project_names = ['innoldb', 'MCZbase', 'operations-puppet', 'Parrot', 'systemd', 'tor_1']
    # Process each project and collect results
    for project in project_names:
        try:
            df = process_project(project)
            all_features.append(df)
        except Exception as e:
            print(f"Error processing {project}: {e}")
    # Combine all project DataFrames into one
    final_df = pd.concat(all_features, ignore_index=True)

    # df = df_features
    df = final_df
    # Define feature columns and label columns
    feature_cols = ['ns', 'nd', 'nf', 'entropy', 'exp', 'rexp', 'sexp', 'ndev', 'age', 'nuc', 'fix', 'la', 'ld', 'lt']
    # Define training labels and evaluation label
    train_labels = ['LSZZ_BUGGY','RSZZ_BUGGY', 'MASZZ_BUGGY', 'MLSZZ_BUGGY']
    eval_label = 'BUGGY'  # label_cols = ['MASZZ_BUGGY','MLSZZ_BUGGY']
    # eval_label = 'MASZZ_BUGGY'
    # Select features and labels
    X = df[feature_cols]
    # y_labels = df[label_cols]
    # We will use the evaluation label for test set ground truth
    y_eval_full = df[eval_label]

    # Check for missing values
    print("\nMissing values in features:")
    print(X.isnull().sum())

    # Optionally, handle missing values (e.g., imputation)
    # For simplicity, let's fill missing values with the mean of each column
    X = X.fillna(X.mean())

    # Verify that there are no missing values
    print("\nMissing values after imputation:")
    print(X.isnull().sum())

    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score)
    }

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    from sklearn.model_selection import StratifiedShuffleSplit

    # This configuration creates 10 random splits with 30% of the data as the test set.
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.6, random_state=42)
    # Define the machine learning pipeline
    from imblearn.under_sampling import ClusterCentroids
    from sklearn.cluster import KMeans

    pipeline = ImbPipeline([
        # ('scaler', StandardScaler()),
        # ('over', BorderlineSMOTE(sampling_strategy='auto', random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy='auto', random_state=42)),
        # ('under', ClusterCentroids(estimator=KMeans(n_init=10, random_state=42), random_state=42)),
        # ("under", AllKNN()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 7. Dictionary to Store Class Distributions
    class_distributions = {label: [] for label in train_labels}

    # Dictionary to store results
    # 6. Define Evaluation Metrics
    metrics = {
        'Label': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'ROC AUC': []
    }

    # 8. Custom Cross-Validation Loop
    for train_label in train_labels:
        print(f"\n{'=' * 50}\nProcessing label: {train_label}\n{'=' * 50}")
        y_train_full = df[train_label]

        fold_number = 1
        for train_idx, test_idx in cv.split(X, y_eval_full):
            for i in range(1):
                print(f"\nFold {fold_number}:")
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                # Training target is based on the current training label
                y_train = y_train_full.iloc[train_idx]
                # Evaluation ground truth is taken from the BUGGY column
                y_true_eval = y_eval_full.iloc[test_idx]
                # Using pandas value_counts()
                print("Distribution of BUGGY:")
                print(y_true_eval.value_counts())

                # Fit the pipeline on the training data
                pipeline.fit(X_train, y_train)

                # Access the resampled training data
                # smote = pipeline.named_steps['over']
                under = pipeline.named_steps['under']
                # X_smote, y_smote = smote.fit_resample(X_train, y_eval_full.iloc[train_idx])
                X_smote, y_smote = X_train, y_eval_full.iloc[train_idx]
                X_resampled, y_resampled = under.fit_resample(X_smote, y_smote)
                # X_resampled, y_resampled = X_smote, y_smote

                # Record class distribution after resampling
                distribution = Counter(y_resampled)
                class_distributions[train_label].append(distribution)
                print(f"Resampled class distribution: {distribution}")
                # plot_class_distribution(distribution, label, fold_number)

                # Predict on the test set
                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]

                # Calculate metrics
                acc = accuracy_score(y_true_eval, y_pred)
                prec = precision_score(y_true_eval, y_pred, zero_division=0)
                rec = recall_score(y_true_eval, y_pred, zero_division=0)
                f1 = f1_score(y_true_eval, y_pred, zero_division=0)
                roc = roc_auc_score(y_true_eval, y_proba)

                # Store metrics
                metrics['Label'].append(train_label)
                metrics['Accuracy'].append(acc)
                metrics['Precision'].append(prec)
                metrics['Recall'].append(rec)
                metrics['F1-Score'].append(f1)
                metrics['ROC AUC'].append(roc)

                print(f"Accuracy: {acc:.4f}")
                print(f"Precision: {prec:.4f}")
                print(f"Recall: {rec:.4f}")
                print(f"F1-Score: {f1:.4f}")
                print(f"ROC AUC: {roc:.4f}")

            fold_number += 1

    # 9. Aggregate Results into DataFrame
    results = pd.DataFrame(metrics)
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-adjust display width
    pd.set_option('display.max_colwidth', None)  # Show full content of each column
    print("\nFinal Cross-Validation Performance Metrics:")
    print(results.groupby('Label').describe())

    # 10. Visualization: Boxplots for Different Labels
    plt.figure(figsize=(18, 10))
    plt.rcParams.update({
        'font.size': 14,  # default text size
        'axes.titlesize': 16,  # title font size
        'axes.labelsize': 14,  # x and y labels
        'xtick.labelsize': 12,  # x tick labels
        'ytick.labelsize': 12,  # y tick labels
        'legend.fontsize': 12  # legend font size
    })
    sns.set_context("talk", font_scale=1.2)  # "talk" increases font sizes, adjust font_scale as needed
    # Grouped Boxplots by Label
    ax = sns.boxplot(data=results.melt(id_vars='Label', var_name='Metric', value_name='Score'),
                     x='Metric', y='Score', hue='Label', palette='Set2')
    ax.set_title(f'Performance Metrics by Label \n $\mathbf{{{project_name}}}$', fontsize=18)
    ax.set_xlabel('Metric', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    plt.ylim(0, 1)
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.show()

    # # 11. Visualize Class Distributions Across Folds for Each Label
    # for label in label_cols:
    #     if class_distributions[label]:
    #         distributions = class_distributions[label]
    #         counts_false = [d.get(0, 0) for d in distributions]
    #         counts_true = [d.get(1, 0) for d in distributions]
    #
    #         df_dist = pd.DataFrame({
    #             'Fold': range(1, len(distributions) + 1),
    #             'False': counts_false,
    #             'True': counts_true
    #         })
    #         df_dist_melted = df_dist.melt(id_vars='Fold', value_vars=['False', 'True'],
    #                                       var_name='Class', value_name='Count')
    #
    #         plt.figure(figsize=(10, 6))
    #         sns.barplot(x='Fold', y='Count', hue='Class', data=df_dist_melted)
    #         plt.title(f'Class Distribution After Resampling for {label}')
    #         plt.xlabel('Fold')
    #         plt.ylabel('Number of Samples')
    #         plt.legend(title='Class')
    #         plt.show()
