import pandas as pd
import os


# Step 1: Read the CSV file
def read_repo_metadata(file_path):
    """
    Reads the repo_metadata.csv file into a DataFrame.
    """
    try:
        repo_data = pd.read_csv(file_path)
        return repo_data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Step 2: Sort the DataFrame
def sort_repositories(repo_data):
    """
    Sorts the repositories by pull_requests, contributors, and forks in descending order.
    """
    if repo_data is not None:
        sorted_repos = repo_data.sort_values(
            by=['pull_requests', 'contributors', 'forks'],
            ascending=[False, False, False]
        )
        return sorted_repos
    else:
        print("No data available to sort.")
        return None

# Main function to execute the process
if __name__ == "__main__":
    # Specify the file path
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    file_path = os.path.join(parent_directory, "data", "repo_metadata.csv")
    # Read the metadata
    repo_data = read_repo_metadata(file_path)
    # Sort the repositories
    sorted_repos = sort_repositories(repo_data)

    # Show all rows and columns temporarily
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    # Display the sorted DataFrame or save it
    if sorted_repos is not None:
        print(sorted_repos.head())  # Display the first few rows
        # Optionally save the sorted DataFrame
        # sorted_repos.to_csv("sorted_repo_metadata.csv", index=False)