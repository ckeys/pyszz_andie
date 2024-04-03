import os
import logging as log
import time
import math
from tempfile import mkdtemp
from options import Options
from shutil import copytree
from datetime import datetime, timedelta
from typing import List, Set, Tuple, Dict
from git import Commit, Repo
from datetime import timezone
from pydriller import RepositoryMining, ModificationType


class CodeRepoFeatureMiner(object):
    def __init__(self, repo_full_name: str, repo_url: str, repos_dir: str = None):
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
        print(rev_list_command)
        commit_hashes = self.repository.git.execute(rev_list_command).strip().split('\n')
        hist_modified_files = set()
        developer_changes = []  # Store developer changes for REXP calculation
        exp_added_lines = 0
        exp_removed_lines = 0

        log.info(f'''---> Have to analyze {len(commit_hashes)} commits in total''')
        start_time = time.time()
        for i in range(0, len(commit_hashes)):
            try:
                commit_hash = commit_hashes[i]
                diff_stat = self.repository.git.diff("--shortstat", f"{commit_hash}^..{commit_hash}")
                diff_filenames = self.repository.git.diff("--name-only", f"{commit_hash}^..{commit_hash}").splitlines()
                hist_modified_files.update(diff_filenames)
                changes = diff_stat.split(',')
                exp_added_lines += sum(
                    [int(change.strip().split()[0]) if 'insertion' in change else 0 for change in changes])
                exp_removed_lines += sum(
                    [int(change.strip().split()[0]) if 'deletion' in change else 0 for change in changes])
                # Store developer changes for REXP calculation
                developer_changes.append(
                    {'date': self.repository.commit(commit_hash).committed_datetime, 'num_changes': len(changes),
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

        log.info(f"It takes {(end_time - start_time) / 60} minutes to run the historical commits analyse!")
        return {"commit": commit_num, "exp_of_files": len(hist_modified_files),
                "exp_of_codes": exp_changed_lines, "exp_of_commits": len(commit_hashes),
                "REXP": REXP, "SEXP": SEXP}

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
    pass
