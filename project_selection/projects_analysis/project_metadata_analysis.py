import requests
import pandas as pd
import os

# Replace these variables with your own details
repo_name = "DemocracyClub/yournextrepresentative"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data")
input_csv = os.path.join(data_dir, "unique_repo_names.csv")
output_csv = os.path.join(data_dir, "repo_metadata.csv")

# API URL for searching repositories
search_url = f"https://api.github.com/search/repositories?q={repo_name} in:name"
token = "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"
# Headers for authentication
headers = {
    "Authorization": f"token {token}"
}

# Read repository names from the CSV file
repo_df = pd.read_csv(input_csv)
repo_names = repo_df["repo_name"].tolist()

# List to store metadata for each repository
metadata_list = []
# Function to fetch repository metadata
def fetch_repo_metadata(repo_name):
    search_url = f"https://api.github.com/search/repositories?q={repo_name} in:name"
    search_response = requests.get(search_url, headers=headers)
    search_data = search_response.json()

    if search_data["total_count"] > 0:
        repo_info = search_data["items"][0]  # Taking the first matched repository
        owner = repo_info["owner"]["login"]
        repo = repo_info["name"]

        # API URL for repository details
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(repo_url, headers=headers)
        repo_data = response.json()
        # print(repo_data)

        # Fetching contributors details
        contributors_url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
        contributors_response = requests.get(contributors_url, headers=headers)
        contributors_data = contributors_response.json()
        # print(contributors_data)

        # Fetching pull requests details
        pulls_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all"
        pulls_response = requests.get(pulls_url, headers=headers)
        pulls_data = pulls_response.json()
        # print(pulls_data)

        # Extracting required information
        stars = repo_data.get("stargazers_count", 0)
        forks = repo_data.get("forks_count", 0)
        contributors = len(contributors_data)
        pull_requests = len(pulls_data)
        language = repo_data.get("language", "Unknown")  # Get the primary language

        print({
            "owner": owner,
            "repo": repo,
            "stars": stars,
            "forks": forks,
            "contributors": contributors,
            "pull_requests": pull_requests,
            "language": language
        })
        return {
            "owner": owner,
            "repo": repo,
            "stars": stars,
            "forks": forks,
            "contributors": contributors,
            "pull_requests": pull_requests,
            "language": language
        }
    else:
        return None

# Loop through all repository names and fetch metadata
for repo_name in repo_names:
    print(f"Fetching metadata for: {repo_name}")
    try:
        metadata = fetch_repo_metadata(repo_name)
        if metadata:
            metadata_list.append(metadata)
    except Exception as e:
        print(e)
        continue

# Create a DataFrame from the metadata list
metadata_df = pd.DataFrame(metadata_list)
# Save the metadata to a CSV file
metadata_df.to_csv(output_csv, index=False)

print(f"Metadata for {len(metadata_list)} repositories saved to {output_csv}")
