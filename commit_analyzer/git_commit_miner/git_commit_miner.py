import os
import csv
import json
import argparse
import math
import logging as log
import subprocess
import time
import re
import pandas as pd
from time import time as ts
from shutil import copytree
from tempfile import mkdtemp
from git import Commit, Repo
from shutil import rmtree
import multiprocessing
from datetime import timezone
from collections import defaultdict
from collections import Counter

lock = multiprocessing.Lock()

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
        log.info("Cleanup objects...")
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
            return pd.read_csv(csv_output_file)

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
        df = pd.DataFrame(commit_info)
        return df


    def get_latest_modified_date_before_commit(self, commit, file_path):
        tree = commit.tree
        # Find the file in the commit tree
        for blob in tree.traverse():
            if blob.path == file_path:
                break
        else:
            log.error(f"File '{file_path}' not found in commit '{commit.hexsha}'")
            return None

        latest_modification_ts = None
        # Traverse the history of the file until the latest modification before the commit
        for parent_commit in commit.iter_parents():
            try:
                parent_blob = parent_commit.tree / blob.path
                # If the blob is found, return the modification date of the parent commit
                if parent_blob:
                    modification_ts = parent_commit.committed_date

                    if parent_commit.committed_date < commit.committed_date:
                        if not latest_modification_ts or modification_ts > latest_modification_ts:
                            latest_modification_ts = modification_ts
                            break
            except Exception as e:
                log.error(f"{e}")
                continue

        # If no modification before the commit is found, return None
        return latest_modification_ts

    def is_binary(self, data):
        """
        Check if the given data is binary.

        Args:
            data (bytes): The file content as bytes.

        Returns:
            bool: True if binary, False otherwise.
        """
        text_characters = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
        return bool(data.translate(None, text_characters))

    def get_lines_of_code_before_change(self, commit: Commit):
        LT = None
        if not commit.parents:
            return LT
        # Get the parent commit (the commit before the current one)
        parent_commit = commit.parents[0]

        if not commit.parents:
            # If there are no parents, it's the initial commit, and there's nothing before it.
            return None, None

        lines_of_code_before_change = {}

        # Get the list of files changed in the commit
        for diff in commit.diff(parent_commit):
            file_path = diff.a_path if diff.a_path else diff.b_path
            try:
                # Access file content in the parent commit's tree
                blob = parent_commit.tree[file_path]
                file_content = blob.data_stream.read()

                # Check if the file is binary
                if self.is_binary(file_content):
                    print(f"Skipping binary file: {file_path}")
                    continue

                # Decode as UTF-8
                decoded_content = file_content.decode('utf-8')

                # TODO: Remove annotations and handle only source code files
                lines_of_code = len(decoded_content.splitlines())
            except KeyError:
                # File did not exist in the parent commit
                lines_of_code = 0
            except UnicodeDecodeError:
                # Non-UTF-8 encoded file
                print(f"Skipping non-UTF-8 encoded file: {file_path}")
                continue
            except Exception as e:
                # Handle other unexpected errors
                print(f"Unexpected error processing file {file_path}: {e}")
                continue

            lines_of_code_before_change[file_path] = lines_of_code
        if lines_of_code_before_change:
            average_lines_of_code = sum(lines_of_code_before_change.values()) / len(lines_of_code_before_change)
        else:
            average_lines_of_code = 0
        LT = average_lines_of_code
        return LT

    def purpose_of_change(self, commit: Commit):
        # List of keywords that indicate a fix
        fix_patterns = [
            r'\bfix(es|ed)?\b',
            r'\bbug(s|fix(es|ed)?)?\b',
            r'\berror(s|fix(es|ed)?)?\b',
            r'\brepair(s|ed)?\b',
            r'\bpatch(ed|es)?\b',
            r'\bdefect(s|fix(es|ed)?)?\b',
            r'\bcorrect(s|ed|ing)?\b',
            r'\bissue(s|fix(es|ed)?)?\b',
            r'\bresolve(s|d)?\b',
            r'\bdebug(s|ged|ging)?\b',
            r'\bflaw(s|less|ed)?\b',
            r'\bfault(s|ed|less)?\b'
        ]
        commit_message_lower = commit.message.lower()
        for pattern in fix_patterns:
            if re.search(pattern, commit_message_lower):
                return 1
        return 0

    def get_last_change(self, file_path, commit):
        current_commit = commit.parents[0]
        while current_commit:
            # Check if the file exists in the current commit
            try:
                blob = current_commit.tree / file_path
                if blob:
                    # File exists in this commit, return the commit
                    return current_commit
            except Exception as e:
                log.error(e)  # File doesn't exist in this commit

            # Move to the parent commit
            if current_commit.parents:
                current_commit = current_commit.parents[0]
            else:
                break  # No more parent commits, end the loop
        # File not found in the commit history
        return None

    def calculate_REXP(self, commit):
        author_name = commit.author.name
        author_email = commit.author.email
        commit_date = commit.authored_datetime
        current_year = commit_date.year
        rev_list_command = [
            'git',
            'rev-list',
            f'--author="{author_name} <{author_email}>"',
            f'--before="{commit_date}"',
            '--all',
            '--format=format:"%H | %ad" --date=format:%Y',
            "|awk '!seen[$1]++'"
        ]
        cwd = self._repository.working_dir
        result = subprocess.run(' '.join(rev_list_command), shell=True, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        # hist_commits_with_year = self.repository.git.execute(rev_list_command, shell=True).strip().split('\n')[1:]
        hist_commits_with_year = result.stdout.strip().split('\n')[1:]
        year_to_commit = dict()
        for commit_with_year in hist_commits_with_year:
            hist_commit, year = commit_with_year.split("|")
            if year.strip() not in year_to_commit:
                year_to_commit[year.strip()] = [hist_commit.strip()]
            else:
                year_to_commit[year.strip()].append(hist_commit.strip())
        REXP = 0.0
        for year, list_of_changes in year_to_commit.items():
            n = (current_year - int(year))
            REXP += len(list_of_changes) / float(n + 1)
        return REXP

    def get_commit_time(self, commit_hash):
        # Get the commit object
        commit = self._repository.commit(commit_hash)
        # Get the commit time and convert it to a datetime object
        commit_time = commit.committed_datetime

        return commit_time


    def calculate_SEXP(self, commit: Commit):
        author_name = commit.author.name
        author_email = commit.author.email
        commit_date = commit.authored_datetime

        git_command = [
            "git",
            "log",
            f'--author="{author_name} <{author_email}>"',
            f'--before="{commit_date}"',
            "--all",
            '--format="%H"',
            "--name-only",
            "|",
            "awk",
            "'/^[0-9a-f]{40}$/{if(commit!=\"\") print commit\" | \"files; commit=$0; files=\"\"; next} {if($0!=\"\") files=files\",\"$0} END{print commit\" | \"files}'"
        ]
        cwd = self._repository.working_dir
        result = subprocess.run(' '.join(git_command), shell=True, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

        # hist_commits_with_filename = self.repository.git.execute(git_command).strip().split('\n')[1:]
        hist_commits_with_filename = result.stdout.strip().split('\n')[1:]
        current_modified_subsystem = set([f.split("/")[0] for f in commit.stats.files.keys()])
        SEXP = 0.0
        for hist in hist_commits_with_filename:
            hist_commit_num, modified_files = hist.split("|")
            hist_commit_num = hist_commit_num.strip()
            modified_files = modified_files.strip().split(",")
            modified_files = [f for f in modified_files if f != '']
            modified_subsystem = set([f.split("/")[0] for f in modified_files])
            if len(current_modified_subsystem & modified_subsystem) > 0:
                SEXP += 1.0
        return SEXP

    def get_commit_changes(self, commit: Commit):
        # Get the parent commit
        if not commit.parents:
            print("The commit has no parents, it's the initial commit.")
            return []
        parent = commit.parents[0]

        # Get the diff between the commit and its parent
        diffs = parent.diff(commit, create_patch=True)

        # Extract the list of changed files
        changes = [diff.a_path if diff.a_path else diff.b_path for diff in diffs]
        return changes

    def calculate_entropy(self, commit: Commit):
        changes = self.get_commit_changes(commit)
        # Count the occurrences of each file change
        file_changes = Counter(changes)
        # Calculate the total number of changes
        total_changes = sum(file_changes.values())
        # Calculate the entropy
        entropy = 0
        for count in file_changes.values():
            probability = count / total_changes
            entropy -= probability * math.log(probability, 2)
        return entropy

    def get_touched_files(self, commit):
        """
        Get the list of touched files in a commit.

        Args:
            commit: A Git commit object.

        Returns:
            List of file paths for files touched in the commit.
        """
        # Check if the commit has parents
        if not commit.parents:
            # Root commit, return an empty list or handle it appropriately
            return []

        # Safely calculate the diff against the first parent
        try:
            return [diff.a_path if diff.a_path else diff.b_path for diff in commit.diff(commit.parents[0])]
        except Exception as e:
            print(f"Error calculating diff for commit {commit}: {e}")
            return []

    def calculate_age(self, commit):
        """
        Calculate the age of modifications in the commit.
        """
        touched_files = self.get_touched_files(commit)
        commit_ts = commit.committed_date
        total_interval = 0
        file_count = 0

        for file_path in touched_files:
            last_modified_ts = self.get_latest_modified_date_before_commit(commit, file_path)
            if last_modified_ts:
                time_interval = commit_ts - last_modified_ts
                total_interval += float(time_interval) / float(86400)
                file_count += 1

        if file_count == 0:
            return 0
        average_age = total_interval / file_count
        return average_age


    def calcualte_ndev(self, commit):
        touched_files = self.get_touched_files(commit)
        developers = set()
        commit_hash = commit.hexsha
        for touched_file in touched_files:
            git_command = [
                'git',
                'log',
                '--pretty=format:%an | %ae',
                commit_hash,
                '--',
                touched_file
            ]
            list_of_developers = self._repository.git.execute(git_command).strip().split('\n')[1:]
            for d in list_of_developers:
                developers.add(d)
        return len(developers)

    def calculate_ndevelopers(self, commit):
        touched_files = self.get_touched_files(commit)
        developers = set()
        for touched_file in touched_files:
            for parent_commit in commit.parents:
                try:
                    parent_blob = parent_commit.tree / touched_file
                    if parent_blob:
                        # Add the author of the parent commit to the set of developers
                        developers.add(parent_commit.author.email)
                except Exception as e:
                    print(f'''[Error]: {e}''')
                    continue
        return len(developers)

    def calculate_diffusion_metrics(self, commit):
        modified_directories = set()
        subsystems = set()

        for file_path in self.get_touched_files(commit):
            # Extract root directory name
            root_directory = file_path.split('/')[0]
            modified_directories.add("/".join(file_path.split('/')[:-1]))

            # Extract subsystem (root directory name)
            subsystems.add(root_directory)

        num_subsystems = len(subsystems)
        num_modified_directories = len(modified_directories)

        entropy = self.calculate_entropy(commit)

        return num_subsystems, num_modified_directories, entropy

    def calculate_nuc(self, commit):
        commit_hash = commit.hexsha
        modified_files = self.get_touched_files(commit)
        unique_last_changes = set()
        for file_path in modified_files:
            git_command = [
                'git',
                'log',
                '--pretty=format:%H',
                commit_hash,
                '--',
                file_path
            ]
            list_of_commits = self._repository.git.execute(git_command).strip().split('\n')[1:]
            for c in list_of_commits:
                unique_last_changes.add(c)
        return len(unique_last_changes)

    def calculate_author_metrics_optimized(self, commit):
        author_name = commit.author.name
        author_email = commit.author.email
        commit_date = commit.authored_datetime
        commit_num = commit.hexsha
        subsystem_changes = {}  # Store subsystem changes for SEXP calculation
        print(commit_num)
        rev_list_command = [
            'git',
            'rev-list',
            f"""--author={author_name} <{author_email}>""",
            f"""--before={commit_date}""",
            '--all'
        ]
        hist_commit_hashes = self._repository.git.execute(rev_list_command).strip().split('\n')[1:]
        hist_modified_files = set()
        developer_changes = []  # Store developer changes for REXP calculation
        exp_added_lines = 0
        exp_removed_lines = 0

        start_time = time.time()
        log.info(f'''---> Have to analyze {len(hist_commit_hashes)} commits in total''')
        if commit.hexsha in hist_commit_hashes:
            hist_commit_hashes.remove(commit.hexsha)
        log.info(f'''---> Have to analyze {len(hist_commit_hashes)} commits in total''')
        for i in range(0, len(hist_commit_hashes)):
            commit_hash = hist_commit_hashes[i]
            commit_obj = self._repository.commit(commit_hash)
            diff_stat_tmp = commit_obj.stats
            diff_filenames_tmp = commit_obj.stats.files
            diff_filenames = list(diff_filenames_tmp.keys())
            hist_modified_files.update(diff_filenames)
            # changes = diff_stat.split(',')
            exp_added_lines += diff_stat_tmp.total['insertions']
            exp_removed_lines += diff_stat_tmp.total['deletions']
            developer_changes.append(
                {'date': self._repository.commit(commit_hash).committed_datetime, 'num_changes': 1,
                 'files': diff_filenames})
            import os
            # Store subsystem changes for SEXP calculation
            for filename in diff_filenames:
                subsystem = os.path.dirname(filename).split(os.path.sep)[0]  # Get root directory name
                if subsystem in subsystem_changes:
                    subsystem_changes[subsystem] += 1
                else:
                    subsystem_changes[subsystem] = 1
        exp_changed_lines = (exp_added_lines + exp_removed_lines)
        end_time = time.time()
        # Calculate REXP
        REXP = self.calculate_REXP(commit)
        # Calculate SEXP
        SEXP = self.calculate_SEXP(commit)

        log.info(f"It takes {(end_time - start_time) / 60} minutes to run the historical commits analyse!")
        return {"commit": commit_num, "exp_of_files": len(hist_modified_files),
                "exp_of_codes": exp_changed_lines, "EXP": len(hist_commit_hashes),
                "REXP": REXP, "SEXP": SEXP}

    def calculate_commit_features(self, commit):
        if isinstance(commit, str):
            commit = self._repository.commit(commit)
        res_dic = dict()
        num_subsystems, num_modified_directories, entropy = self.calculate_diffusion_metrics(commit)
        res_dic['num_subsystems'] = num_subsystems
        res_dic['num_subsystems'] = num_subsystems
        res_dic['num_modified_directories'] = num_modified_directories
        res_dic['entropy'] = entropy
        res_dic['LT'] = self.get_lines_of_code_before_change(commit)
        res_dic['FIX'] = self.purpose_of_change(commit)
        age = self.calculate_age(commit)
        res_dic['age'] = age
        ndev = self.calcualte_ndev(commit)
        res_dic['ndev'] = ndev
        res_dic['nuc'] = self.calculate_nuc(commit)
        add = commit.stats.total['insertions']
        deleted = commit.stats.total['deletions']
        num_files = commit.stats.total['files']
        lines = commit.stats.total['lines']
        experience_dict = self.calculate_author_metrics_optimized(commit)
        # This creates a new dictionary
        res_dic.update(experience_dict)
        commit_modified_files = list(commit.stats.files.keys())
        res_dic['lines_of_added'] = add
        res_dic['lines_of_deleted'] = deleted
        res_dic['lines_of_modified'] = lines
        res_dic['num_files'] = num_files
        res_dic['modified_files'] = commit_modified_files
        res_dic['is_Friday'] = 1 if commit.committed_datetime.weekday() == 4 else 0
        return res_dic

def main(repo_name, repo_dir, output_dir):
    git_miner = GitCommitMiner(repo_name, repo_dir)
    df = git_miner.mine_commit_history(output_dir)
    # Initialize a list to hold results
    results = []
    for index, row in df.iterrows():
        res = git_miner.calculate_commit_features(commit=row['commit_hash'])
        results.append(res)  # Append each `res` dictionary to the list
        print(f'''Row {index}, result:{res}''')
        if index==5:
            break

    # Save results to a JSON file
    res_path = f'''{output_dir}/{repo_name.replace("/","_")}_commit_features.json'''
    with open(res_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    # Save results to a CSV file
    res_path = f'''{output_dir}/{repo_name.replace("/", "_")}_commit_features.csv'''
    results_df = pd.DataFrame(results)
    results_df.to_csv(res_path, index=False)
    print("Results successfully saved to JSON and CSV.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mine commit history of a git repository.')
    parser.add_argument('-repo_full_name', dest='repo_full_name', type=str, help='Path to the git repository')
    parser.add_argument('-repo_dir', dest='repo_dir', type=str, help='Name of the git repository to be')
    parser.add_argument('-output_dir', dest='output_dir', type=str, default='../data/commit_history',
                        help='Output file path (directory and filename prefix)')
    args = parser.parse_args()

    repo_name = args.repo_full_name
    repo_dir = args.repo_dir
    output_dir = args.output_dir
    main(repo_name, repo_dir, output_dir)