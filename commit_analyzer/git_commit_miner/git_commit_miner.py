import os
import csv
import json
import argparse
import logging as log
from shutil import copytree
from tempfile import mkdtemp
from git import Commit, Repo
from shutil import rmtree


class GitCommitMiner:

    def __init__(self, repo_full_name: str, repos_dir: str = None):
        self.project_name = repo_full_name.replace("/", "_")
        TEMP_WORKING_DIR = '_szztemp'
        repo_url = f'https://test:test@github.com/{repo_full_name}.git'  # using test:test as git login to skip private repos during clone
        self._repository = None
        os.makedirs(TEMP_WORKING_DIR, exist_ok=True)
        self.__temp_dir = mkdtemp(dir=os.path.join(os.getcwd(), TEMP_WORKING_DIR))
        log.info(f"Create a temp directory : {self.__temp_dir}")
        print(f"Create a temp directory : {self.__temp_dir}")
        self._repository_path = os.path.join(self.__temp_dir, repo_full_name.replace('/', '_'))
        if not os.path.isdir(self._repository_path):
            if repos_dir:
                repo_dir = os.path.join(repos_dir, repo_full_name)
                if os.path.isdir(repo_dir):
                    copytree(repo_dir, self._repository_path, symlinks=True)
                else:
                    log.error(f'unable to find local repository path: {repo_dir}')
                    exit(-4)
            else:
                log.info(f"Cloning repository {repo_full_name}...")
                print(f"Cloning repository {repo_full_name}...")
                Repo.clone_from(url=repo_url, to_path=self._repository_path)

        self._repository = Repo(self._repository_path)

    def __del__(self):
        log.info("cleanup objects...")
        print(f"Clean up {self._repository_path}")
        self.__cleanup_repo()
        self.__clear_gitpython()

    def __cleanup_repo(self):
        """ Cleanup of local repository used by SZZ """
        log.info(f"Clean up {self.__temp_dir}")
        print(f"Clean up {self.__temp_dir}")
        try:
            if os.path.isdir(self.__temp_dir):
                rmtree(self.__temp_dir)
        except Exception as e:
            print("Error cleaning up repo:", e)

    def __clear_gitpython(self):
        """ Cleanup of GitPython due to memory problems """
        if self._repository:
            self._repository.close()
            self._repository.__del__()

    def mine_commit_history(self, output_dir='./commit_history'):

        # Initialize an empty list to store commit information
        commit_info = []

        # Iterate through all commits in the repository
        for commit in self._repository.iter_commits(reverse=True):
            commit_data = {
                'author_name': commit.author.name,
                'date_time': commit.authored_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'commit_hash': commit.hexsha,
                'commit_message': commit.message.strip()
            }
            log.info(f"Commit Data: {commit_data}")
            print(f"Commit Data: {commit_data}")
            commit_info.append(commit_data)

        # Define the output file path
        log.info("Start to write to csv file!")
        print("Start to write to csv file!")
        csv_output_file = f'''{output_dir}/{self.project_name}_commit_history_data.csv'''
        if os.path.exists(csv_output_file):
            return

        with open(csv_output_file, 'w', newline='') as csvfile:
            fieldnames = ['author_name', 'date_time', 'commit_hash', 'commit_message']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for commit_data in commit_info:
                writer.writerow(commit_data)
        log.info("CSV file writting is done!")
        print("CSV file writing is done!")
        log.info("Start to write to json file!")
        print("Start to write to json file!")
        json_output_file = f'''{output_dir}/{self.project_name}_commit_history_data.json'''
        with open(json_output_file, 'w') as jsonfile:
            json.dump(commit_info, jsonfile, indent=4)

        print("Commit history successfully exported to", json_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mine commit history of a git repository.')
    parser.add_argument('-repo_full_name', dest='repo_full_name', type=str, help='Path to the git repository')
    parser.add_argument('-repo_dir', dest='repo_dir', type=str, help='Name of the git repository to be')
    parser.add_argument('-output_dir', dest='output_dir', type=str, default='./output/commit_history',
                        help='Output file path (directory and filename prefix)')
    args = parser.parse_args()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)

    with open(f'{parent_directory}/data/unique_repo_names.csv', 'r',newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            repo_name = row['repo_name']

            git_miner = GitCommitMiner(repo_name, args.repo_dir)
            json_output_file = f'''{args.output_dir}/{git_miner.project_name}_commit_history_data.json'''
            csv_output_file = f'''{args.output_dir}/{git_miner.project_name}_commit_history_data.csv'''
            if not os.path.exists(json_output_file):
                print(f"Currently Processing {repo_name}")
                log.info(f"Currently Processing {repo_name}")
                git_miner.mine_commit_history(args.output_dir)
            else:
                print(f"Currently Processing {repo_name} but it has already exist!")
