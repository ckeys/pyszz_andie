import os
import yaml

def read_config_yml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    github_token_path = config.get("GITHUB_TOKEN_PATH")
    with open(github_token_path, 'r') as file:
        github_token = file.read().strip()
    config['GITHUB_TOKEN'] = github_token
    config.pop('GITHUB_TOKEN_PATH')
    return config

def read_config_file():
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    base_directory = current_directory.split(os.sep)[:-1]
    path = os.sep.join(base_directory+['conf','commit_minner.yml'])
    return read_config_yml(path)

if __name__ == '__main__':
    read_config_file()