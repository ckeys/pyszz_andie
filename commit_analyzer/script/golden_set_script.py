import json


def calculate_unique_repo_names(json_file_path):
    # Initialize an empty set to hold unique repo_name values
    unique_repo_names = set()

    # Open the JSON file and read its contents
    with open(json_file_path, 'r') as file:
        data = json.load(file)

        # Iterate through the list of dictionaries
        for record in data:
            # Extract the repo_name value
            repo_name = record.get('repo_name')

            # Add repo_name to the set (sets automatically handle uniqueness)
            if repo_name:
                unique_repo_names.add(repo_name)

    # Return the set of unique repo_names
    return unique_repo_names


# Path to the JSON file
json_file_path = '/Users/andie/PycharmProjects/icse2021-szz-replication-package/json-input-raw/bugfix_commits_all.json'

# Calculate unique repo names
unique_repo_names = calculate_unique_repo_names(json_file_path)

# Print the unique repo names
print(f"Unique repo names:{len(unique_repo_names)}")
# for repo_name in unique_repo_names:
#     print(repo_name)