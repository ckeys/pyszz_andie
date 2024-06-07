import requests
import json
import os
import time
import yaml
import os
from commit_analyzer.util.read_config_file import read_config_file

def check_github_repo_exists(repo_name, headers):
    url = f"https://api.github.com/repos/{repo_name}"
    response = requests.get(url, headers=headers)
    time.sleep(3)
    if response.status_code == 200:
        return True
    elif response.status_code == 404:
        return False
    elif response.status_code == 403:  # Rate limit exceeded
        print("Rate limit exceeded. Sleeping for a while...")
        time.sleep(60*10)  # Sleep for 60 seconds
        return check_github_repo_exists(repo_name)  # Retry the request
    else:
        response.raise_for_status()


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to check if a file exists and clean its content
def clean_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            print(f'''Cleaning {file_path}''')
            file.write('[]')  # Write empty JSON object

def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def append_to_json_file(file_path, repo_data):
    if os.path.exists(file_path):
        data_list = read_json_file(file_path)
        if len(data_list) == 0:
            data_list = []
    else:
        data_list = []

    data_list.append(repo_data)
    write_json_file(file_path, data_list)


def main():
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    yml_file_path = current_directory.split(os.sep)[:-1]
    # Example usage
    config = read_config_file()
    print(config)
    GITHUB_TOKEN = config['GITHUB_TOKEN']
    BASE_PATH = config['BASE_PATH']
    input_file_path = f"{BASE_PATH}/in/bugfix_commits_all.json"
    output_file_path = f"{BASE_PATH}/in/invalid_project.json"
    valid_output_file_path = f"{BASE_PATH}/in/valid_project.json"
    # Check and clean the files
    clean_json_file(output_file_path)
    clean_json_file(valid_output_file_path)
    #
    repo_data_list = read_json_file(input_file_path)
    # Replace 'your_personal_access_token' with your actual GitHub personal access token
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}'
    }
    for repo_data in repo_data_list:
        repo_name = repo_data['repo_name']
        if not check_github_repo_exists(repo_name, headers):
            append_to_json_file(output_file_path, repo_data)
            print(f"Repository '{repo_name}' is not available on GitHub. Added to invalid projects.")
        else:
            append_to_json_file(valid_output_file_path, repo_data)
            print(f"Repository '{repo_name}' is available on GitHub.")
    #
    # In total, there are 1115 projects for studied
    # 37 are invalid projects
    # 1078 are valid projects

if __name__ == "__main__":
    main()
