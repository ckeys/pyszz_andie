import os
import logging as log
import time
import math
import argparse
import pandas as pd
from tempfile import mkdtemp
from shutil import copytree
from datetime import datetime, timedelta
from typing import List, Set, Tuple, Dict
from git import Commit, Repo
from datetime import timezone
from pydriller import RepositoryMining, ModificationType


class Options:
    # Sets the global home of the project (useful for running external tools)
    PYSZZ_HOME = os.path.dirname(os.path.realpath(__file__))

    TEMP_WORKING_DIR = '_szztemp'


class CodeRepoFeatureMiner(object):
    def __init__(self, repo_full_name: str, repos_dir: str = None):
        """
                Init an abstract SZZ to use as base class for SZZ implementations.
                AbstractSZZ uses a temp folder to clone and interact with the given git repo, where
                the name of the repo folder will be the full name having '/' replaced with '_'.
                The init method also set the default_ignore_regex for modified lines.

                :param str repo_full_name: full name of the Git repository to clone and interact with
                :param str repo_url: url of the Git repository to clone
                :param str repos_dir: temp folder where to clone the given repo
                """
        self._repository = None
        repo_url = f'https://test:test@github.com/{repo_full_name}.git'

        os.makedirs(Options.TEMP_WORKING_DIR, exist_ok=True)
        self.__temp_dir = mkdtemp(dir=os.path.join(os.getcwd(), Options.TEMP_WORKING_DIR))
        log.info(f"Create a temp directory : {self.__temp_dir}")
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
                Repo.clone_from(url=repo_url, to_path=self._repository_path)

        self._repository = Repo(self._repository_path)

    def get_commit_object(self, commit_hash):
        # 使用提交哈希获取Commit对象
        commit = self._repository.commit(commit_hash)
        return commit

    def calculate_REXP(self, developer_changes, commit):
        total_weighted_changes = 0
        total_weight = 0
        commit_date_aware = commit.committed_datetime.replace(tzinfo=timezone.utc)
        # Iterate over developer's changes
        for change in developer_changes:
            # Calculate age of the change (in years)
            age = (commit_date_aware - change['date']).days / 365

            # Weighted change is the number of changes divided by the age
            weighted_change = change['num_changes'] / (age + 1)

            # Sum weighted changes and total weight
            total_weighted_changes += weighted_change

        # Calculate REXP
        REXP = total_weighted_changes
        return REXP

    def calculate_SEXP(self, developer_changes, subsystem_changes):
        total_changes_to_subsystems = 0

        # Iterate over developer's changes
        for change in developer_changes:
            # Check if the change modified any subsystems
            for subsystem in subsystem_changes:
                if subsystem in change['files']:
                    total_changes_to_subsystems += 1

        # Calculate SEXP
        SEXP = total_changes_to_subsystems
        return SEXP

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
        # here need to exclude the commit itself
        print(rev_list_command)
        commit_hashes = self._repository.git.execute(rev_list_command).strip().split('\n')
        if commit.hexsha in commit_hashes:
            commit_hashes.remove(commit.hexsha)
        hist_modified_files = set()
        developer_changes = []  # Store developer changes for REXP calculation
        exp_added_lines = 0
        exp_removed_lines = 0

        log.info(f'''---> Have to analyze {len(commit_hashes)} commits in total''')
        print(f'''---> Have to analyze {len(commit_hashes)} commits in total''')
        start_time = time.time()
        for i in range(0, len(commit_hashes)):
            try:
                commit_hash = commit_hashes[i]
                commit = self._repository.commit(commit_hash)
                diff_stat = commit.stats
                lines_added = diff_stat.total['insertions']
                lines_deleted = diff_stat.total['deletions']
                num_changes = lines_added + lines_deleted
                num_files_changed = diff_stat.total['files']
                diff_filenames = set(diff_stat.files.keys())
                # diff_stat = self._repository.git.diff("-- ", f"{commit_hash}^..{commit_hash}")
                # diff_filenames = self._repository.git.diff("--name-only", f"{commit_hash}^..{commit_hash}").splitlines()
                hist_modified_files.update(diff_filenames)
                # changes = diff_stat.split(',')
                # exp_added_lines += sum(
                #     [int(change.strip().split()[0]) if 'insertion' in change else 0 for change in changes])
                # exp_removed_lines += sum(
                #     [int(change.strip().split()[0]) if 'deletion' in change else 0 for change in changes])
                # Store developer changes for REXP calculation
                developer_changes.append(
                    {'date': self._repository.commit(commit_hash).committed_datetime, 'num_changes': num_changes,
                     'files': diff_filenames})
                import os
                # Store subsystem changes for SEXP calculation
                for filename in diff_filenames:
                    subsystem = os.path.dirname(filename).split(os.path.sep)[0]  # Get root directory name
                    if subsystem in subsystem_changes:
                        subsystem_changes[subsystem] += 1
                    else:
                        subsystem_changes[subsystem] = 1

            except Exception as e:
                log.error(commit_hashes[i])
                log.error(e)
                continue
        exp_changed_lines = (exp_added_lines + exp_removed_lines)
        end_time = time.time()
        # Calculate REXP
        REXP = self.calculate_REXP(developer_changes, commit)
        # Calculate SEXP
        SEXP = self.calculate_SEXP(developer_changes, subsystem_changes)
        contain_defect_fix = self.contains_defect_fix(commit)
        log.info(f"It takes {(end_time - start_time) / 60} minutes to run the historical commits analyse!")
        return {"commit": commit_num, "exp_of_files": len(hist_modified_files),
                "exp_of_codes": exp_changed_lines, "exp_of_commits": len(commit_hashes),
                "REXP": REXP, "SEXP": SEXP, 'contain_defect_fix': contain_defect_fix}

    def get_merge_commits(self, commit_hash: str) -> Set[str]:
        merge = set()
        repo_mining = RepositoryMining(single=commit_hash, path_to_repo=self.repository_path).traverse_commits()
        for commit in repo_mining:
            try:
                if commit.merge:
                    merge.add(commit.hash)
            except Exception as e:
                log.error(f'unable to analyze commit: {self.repository_path} {commit.hash}')

        if len(merge) > 0:
            log.info(f'merge commits count: {len(merge)}')

        return merge

    def calculate_entropy(self, commit: Commit) -> float:
        modified_lines = []

        # Calculate modified lines in each file and collect them
        for file_path, stats in commit.stats.files.items():
            modified_lines.append(stats['insertions'] + stats['deletions'])

        total_modified_lines = sum(modified_lines)

        # Calculate entropy
        entropy = 0.0
        for lines in modified_lines:
            if lines > 0:
                entropy -= lines / total_modified_lines * math.log2(lines / total_modified_lines)

        return entropy

    def contains_defect_fix(self, commit):
        # Keywords to search for in the commit message
        keywords = ["bug", "fix", "defect", "patch"]

        # Search for keywords in the commit message
        message = commit.message.lower()
        for keyword in keywords:
            if keyword in message:
                return 1

        return 0

    def get_touched_files(self, commit):
        return [diff.a_path for diff in commit.diff(commit.parents[0])]

    def calculate_age(self, commit):
        """
        Calculate the age of modifications in the commit.
        """
        touched_files = self.get_touched_files(commit)
        commit_date = datetime.fromtimestamp(commit.committed_date)
        total_interval = timedelta(0)
        file_count = 0

        for file_path in touched_files:
            last_modified_date = self.get_latest_modified_date_before_commit(commit, file_path)
            if last_modified_date:
                time_interval = commit_date - last_modified_date
                total_interval += time_interval
                file_count += 1

        if file_count == 0:
            return 0

        average_age = total_interval.seconds / file_count
        return average_age

    def get_latest_modified_date_before_commit(self, commit, file_path):
        tree = commit.tree

        # Find the file in the commit tree
        for blob in tree.traverse():
            if blob.path == file_path:
                break
        else:
            print(f"File '{file_path}' not found in commit '{commit.hexsha}'")
            return None

        # Traverse the history of the file until the latest modification before the commit
        for parent_commit in commit.iter_parents():
            try:
                parent_blob = parent_commit.tree / blob.path
                # If the blob is found, return the modification date of the parent commit
                if parent_blob:
                    latest_modification_date = datetime.fromtimestamp(parent_commit.committed_date)
                    return latest_modification_date
            except IndexError or KeyError:
                continue

        # If no modification before the commit is found, return None
        return None

    def get_touched_files(self, commit):
        return commit.stats.files.keys()

    def calculate_ndevelopers(files, commit):
        touched_files = [diff.a_path for diff in commit.diff(commit.parents[0])]
        developers = set()
        for touched_file in touched_files:
            for parent_commit in commit.parents:
                try:
                    parent_blob = parent_commit.tree / touched_file
                    if parent_blob:
                        # Add the author of the parent commit to the set of developers
                        developers.add(parent_commit.author.email)
                except KeyError:
                    continue
        return len(developers)

    def get_touched_date(self, commit, file_path):
        # Get the commit time for the file
        commit_time = commit.committed_date
        # Get the file modification time
        file_mod_time = commit.tree[file_path].committed_date
        return datetime.fromtimestamp(file_mod_time)

    def mine_all_commits(self):
        """
        Fetch all commits from the history of the repository.
        """
        log.info("Mining all commits from the repository.")
        if not self._repository:
            log.error("Repository not initialized.")
            return []

        commits = list(self._repository.iter_commits())
        log.info(f"Total number of commits found: {len(commits)}")
        return commits


if __name__ == "__main__":

    current_file_path = os.path.abspath(__file__)
    repo_list_path = "/".join(current_file_path.split("/")[:-1] + ['data', 'unique_repo_names.csv'])
    df = pd.read_csv(repo_list_path)
    parser = argparse.ArgumentParser(description='Process an input file and output its contents to a specified file.')
    parser.add_argument('-c', '--commits_file_path', type=str, required=False,
                        help='The path to the historical commits file.')
    parser.add_argument('-cdir', '--commit_file_dir', type=str, required=True,
                        help='The dir to the historical commits file.')
    parser.add_argument('-r', '--repo_path', type=str, required=True, help='The path to the repository.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('-rn', '--repo_name', type=str, required=False, help='The path to the output file.')
    args = parser.parse_args()
    commits_file_path = args.commits_file_path if args.commits_file_path else "/Users/andie/PycharmProjects/pyszz_andie/commit_analyzer/data/test_data/ad510_decoherence_commit_history_data.csv"
    commit_file_base_dir = args.commit_file_dir if args.commit_file_dir else "/Users/andie/PycharmProjects/pyszz_andie/commit_analyzer"
    repo_path = args.repo_path if args.repo_path else "/Users/andie/Andie/test_repo"
    output_path = args.output_path if args.output_path else f"/Users/andie/PycharmProjects/pyszz_andie/commit_analyzer/data/test_data/commit_features.csv"
    for idx, row in df.iterrows():
        repo_name = row['repo_name']
        print(f'''Currently Processing Project {repo_name}!''')
        repo_name = repo_name.strip()
        commit_file_dir = [commit_file_base_dir] + [f'''{repo_name.replace('/', '_')}_commit_history_data.csv''']
        print(f'''commit file dir : {commit_file_dir}''')
        commit_file_path = '/'.join(commit_file_dir)
        historical_commit_data = pd.read_csv(commit_file_path)
        if not os.path.exists(commit_file_path):
            print(f'''The commit data {historical_commit_data} for the project {repo_name} is not done yet!''')
            continue
        tmp_output_path = f'''{output_path}/{repo_name.replace('/', '_')}_commit_features.csv'''
        if os.path.exists(tmp_output_path):
            print(f'''The project {repo_name} already processed, go continue with the next project!!!''')
            continue
        miner = CodeRepoFeatureMiner(repo_full_name=repo_name, repos_dir=repo_path)
        features_list = list()
        for index, row in historical_commit_data.iterrows():
            print(f'''The output path is :{output_path}''')
            commit_hash = row['commit_hash']
            commit = miner.get_commit_object(commit_hash)
            res_dic = miner.calculate_author_metrics_optimized(commit=commit)
            add = commit.stats.total['insertions']
            print(add)
            deleted = commit.stats.total['deletions']
            print(deleted)
            num_files = commit.stats.total['files']
            print(num_files)
            lines = commit.stats.total['lines']
            print(lines)
            res_dic['lines_of_added'] = add
            res_dic['lines_of_deleted'] = deleted
            res_dic['lines_of_modified'] = lines
            res_dic['num_files'] = num_files
            res_dic['is_Friday'] = 1 if commit.committed_datetime.weekday() == 4 else 0
            features_list.append(res_dic)
        print(f'''>>>>> Writting results to {tmp_output_path} ''')
        features_df = pd.DataFrame(features_list)
        features_df.to_csv(tmp_output_path, index=False)
        print(f'''>>>>> Writting is DONE !!!!!!!!!!!!!!!!!!!!! ''')

    # commits_file_path = args.commits_file_path if args.commits_file_path else "/Users/andie/PycharmProjects/pyszz_andie/commit_analyzer/data/test_data/ad510_decoherence_commit_history_data.csv"
    # repo_path = args.repo_path if args.repo_path else "/Users/andie/Andie/test_repo"
    # output_path = args.output_path if args.output_path else f"/Users/andie/PycharmProjects/pyszz_andie/commit_analyzer/data/test_data/commit_features.csv"
    # repo_name = args.repo_name if args.repo_name else 'ad510/decoherence'
    # output_path = f'''{output_path}/{repo_name.replace('/', '_')}_commit_features.csv'''
    # historical_commit_data = pd.read_csv(commits_file_path)
    # miner = CodeRepoFeatureMiner(repo_full_name=repo_name, repos_dir=repo_path)
    print(f'''Commit Minner Work is Finished!!!!! ''')
