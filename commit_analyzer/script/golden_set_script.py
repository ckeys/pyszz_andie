import json


def calculate_unique_repo_names(json_file_path):
    """
    Calculate unique repository names from a JSON file.

    This function reads a JSON file containing a list of dictionaries,
    each representing a repository or commit record, and extracts unique
    repository names (specified by the `repo_name` key). The function uses
    a set to ensure only unique names are included, ignoring duplicates.

    Parameters:
    ----------
    json_file_path : str
        The path to the JSON file containing the input data. Each dictionary
        entry in the JSON should include the key `repo_name`.

    Returns:
    -------
    set
        A set of unique repository names found in the input JSON file.

    Example:
    -------
    Given a JSON file `bugfix_commits_all.json` with the following contents:

        [
            {"repo_name": "repo1", "commit_id": "123"},
            {"repo_name": "repo2", "commit_id": "456"},
            {"repo_name": "repo1", "commit_id": "789"}
        ]

    Calling the function with this file path:

        unique_repo_names = calculate_unique_repo_names('/path/to/bugfix_commits_all.json')

    Would return:

        {'repo1', 'repo2'}

    Usage:
    ------
    # Path to the JSON file
    json_file_path = '/path/to/your/file.json'

    # Calculate unique repo names
    unique_repo_names = calculate_unique_repo_names(json_file_path)

    # Print the count and names of unique repositories
    print(f"Unique repo names count: {len(unique_repo_names)}")
    for repo_name in unique_repo_names:
        print(repo_name)

    Error Handling:
    --------------
    - If the file at `json_file_path` does not exist or cannot be accessed,
      a `FileNotFoundError` will be raised.
    - If the JSON structure lacks the `repo_name` key in some entries, those
      entries will be ignored without affecting the output.

    Dependencies:
    ------------
    - json: This function uses Python's built-in `json` library for reading and
      parsing JSON files.
    """

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