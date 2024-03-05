import logging as log
import traceback
import math
from datetime import timezone

from datetime import datetime, timedelta
from typing import List, Set, Tuple, Dict
import time
from time import time as ts
from git import Commit
from pydriller import RepositoryMining, ModificationType
from szz.core.abstract_szz import AbstractSZZ, ImpactedFile
from szz.ag_szz import AGSZZ
from szz.core.abstract_szz import ImpactedFile, DetectLineMoved
from szz.ma_szz import MASZZ


class MLSZZ(AGSZZ):

    def __init__(self, repo_full_name: str, repo_url: str, repos_dir: str = None):
        super().__init__(repo_full_name, repo_url, repos_dir)

    def calculate_REXP(self, developer_changes, commit):
        total_weighted_changes = 0
        total_weight = 0
        commit_date_aware = commit.committed_datetime.replace(tzinfo=timezone.utc)
        # Iterate over developer's changes
        for change in developer_changes:
            # Calculate age of the change (in years)
            age = (commit_date_aware - change['date']).days / 365

            # Weighted change is the number of changes divided by the age
            weighted_change = change['num_changes'] / (age+1)

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

    def _is_git_mode_change(self, git_show_output: List[str], current_file: str):
        return any(line.strip().startswith('mode change') and current_file in line for line in git_show_output)

    def select_meta_changes(self, commit_hash: str, current_file: str, filter_revert: bool = False) -> Set[str]:
        meta_changes = set()
        repo_mining = RepositoryMining(path_to_repo=self.repository_path, single=commit_hash).traverse_commits()
        for commit in repo_mining:
            # ignore revert commits
            if filter_revert and (commit.msg.startswith("Revert") or "This reverts commit" in commit.msg):
                log.info(f'exclude meta-change (Revert commit): {current_file} {commit.hash}')
                meta_changes.add(commit.hash)
                continue

            show_str = self.repository.git.show(commit.hash, '--summary').splitlines()
            if show_str and self._is_git_mode_change(show_str, current_file):
                log.info(f'exclude meta-change (file mode change): {current_file} {commit.hash}')
                meta_changes.add(commit.hash)
            else:
                try:
                    for m in commit.modifications:
                        if (current_file == m.new_path or current_file == m.old_path) and (
                                m.change_type in self.change_types_to_ignore):
                            log.info(f'exclude meta-change ({m.change_type}): {current_file} {commit.hash}')
                            meta_changes.add(commit.hash)
                except Exception as e:
                    log.error(f'unable to analyze commit: {self.repository_path} {commit.hash}')

        return meta_changes

    def find_bic(self, fix_commit_hash: str, impacted_files: List['ImpactedFile'], **kwargs) -> Tuple[
        Set[Commit], List[Dict]]:
        """
                Find bug introducing commits candidates.

                :param str fix_commit_hash: hash of fix commit to scan for buggy commits
                :param List[ImpactedFile] impacted_files: list of impacted files in fix commit
                :key ignore_revs_file_path (str): specify ignore revs file for git blame to ignore specific commits.
                :returns Set[Commit] a set of bug introducing commits candidates, represented by Commit object
                """
        blame_detailes = dict()
        can_feas = list()
        log.info(f"find_bic() kwargs: {kwargs}")

        ignore_revs_file_path = kwargs.get('ignore_revs_file_path', None)
        self._set_working_tree_to_commit(fix_commit_hash)

        bug_introd_commits = set()
        commits_to_ignore = set()
        res_dic = dict()
        for imp_file in impacted_files:
            commits_to_ignore_current_file = commits_to_ignore.copy()
            try:
                blame_data = self._blame(
                    rev='HEAD^',
                    file_path=imp_file.file_path,
                    modified_lines=imp_file.modified_lines,
                    ignore_revs_file_path=ignore_revs_file_path,
                    ignore_whitespaces=False,
                    skip_comments=False
                )
                bug_introd_commits.update([entry.commit for entry in blame_data])
                for commit in bug_introd_commits:
                    add = commit.stats.total['insertions']
                    print(add)
                    deleted = commit.stats.total['deletions']
                    print(deleted)
                    num_files = commit.stats.total['files']
                    print(num_files)
                    lines = commit.stats.total['lines']
                    print(lines)
                    res_dic = self.calculate_author_metrics_optimized(commit)
                    commit_modified_files = list(commit.stats.files.keys())
                    res_dic['lines_of_added'] = add
                    res_dic['lines_of_deleted'] = deleted
                    res_dic['lines_of_modified'] = lines
                    res_dic['num_files'] = num_files
                    res_dic['modified_files'] = commit_modified_files
                    res_dic['is_Friday'] = 1 if commit.committed_datetime.weekday() == 4 else 0
                    print(res_dic)
                    can_feas.append(res_dic)
            #  {'commit': <git.Commit "2574243a39d90a2673cf56647c524e268d7f169e">, 'num_files_changed': 4956, 'lines_of_code_changed': 548857, 'num_of_commits': 1566}
            except:
                print(traceback.format_exc())

        if 'issue_date_filter' in kwargs and kwargs['issue_date_filter']:
            before = len(bug_introd_commits)
            bug_introd_commits = [c for c in bug_introd_commits if c.authored_date <= kwargs['issue_date']]
            log.info(f'Filtering by issue date returned {len(bug_introd_commits)} out of {before}')
        else:
            log.info("Not filtering by issue date.")
        return (bug_introd_commits, can_feas)

    def calculate_entropy(self, commit):
        modified_lines = []

        # Calculate modified lines in each file and collect them
        for file_path, stats in commit.stats.files.items():
            modified_lines.append(stats['insertions'] + stats['deletions'])

        total_modified_lines = sum(modified_lines)

        # Calculate entropy
        entropy = 0
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

    def get_touched_date(self, commit, file_path):
        # Get the commit time for the file
        commit_time = commit.committed_date
        # Get the file modification time
        file_mod_time = commit.tree[file_path].committed_date
        return datetime.fromtimestamp(file_mod_time)

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

    def get_touched_files(self, commit):
        return commit.stats.files.keys()

    def calculate_metrics(self, commit):
        modified_directories = set()
        subsystems = set()

        for file_path in self.get_touched_files(commit):
            # Extract root directory name
            root_directory = file_path.split('/')[0]
            modified_directories.add(file_path.split('/')[0])

            # Extract subsystem (root directory name)
            subsystems.add(root_directory)

        num_subsystems = len(subsystems)
        num_modified_directories = len(modified_directories)

        entropy = self.calculate_entropy(commit)

        return num_subsystems, num_modified_directories, entropy

    def get_latest_modified_date_before_commit(self, commit, file_path):
        tree = commit.tree

        # Find the file in the commit tree
        for blob in tree.traverse():
            if blob.path == file_path:
                break
        else:
            print(f"File '{file_path}' not found in commit '{commit.hash}'")
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

    def get_last_modified_date_before_commit_tmp(self, file_path, commit):
        """
        Get the last modification date of the given file before the commit date.
        """
        last_modified_date = None
        for commit in self.repository.iter_commits(commit):
            file = commit.tree[file_path]
            if file:
                last_modified_date = datetime.fromtimestamp(file.committed_date)
                break
        return last_modified_date

    def find_bic_v2(self, fix_commit_hash: str, impacted_files: List['ImpactedFile'], **kwargs) -> Tuple[
        Set[Commit], List[Dict]]:
        """
                Find bug introducing commits candidates.

                :param str fix_commit_hash: hash of fix commit to scan for buggy commits
                :param List[ImpactedFile] impacted_files: list of impacted files in fix commit
                :key ignore_revs_file_path (str): specify ignore revs file for git blame to ignore specific commits.
                :returns Set[Commit] a set of bug introducing commits candidates, represented by Commit object
                """
        blame_detailes = dict()
        can_feas = list()
        log.info(f"find_bic() kwargs: {kwargs}")

        ignore_revs_file_path = kwargs.get('ignore_revs_file_path', None)
        self._set_working_tree_to_commit(fix_commit_hash)

        params = dict()
        max_change_size = kwargs.get('max_change_size', MASZZ.DEFAULT_MAX_CHANGE_SIZE)
        filter_revert = kwargs.get('filter_revert_commits', False)
        params['ignore_revs_file_path'] = kwargs.get('ignore_revs_file_path', None)
        params['detect_move_within_file'] = kwargs.get('detect_move_within_file', True)
        params['detect_move_from_other_files'] = kwargs.get('detect_move_from_other_files', DetectLineMoved.SAME_COMMIT)
        params['ignore_revs_list'] = list()
        if kwargs.get('blame_rev_pointer', None):
            params['rev_pointer'] = kwargs['blame_rev_pointer']

        bug_introd_commits = set()
        commits_to_ignore = set()
        res_dic = dict()
        start = ts()
        for imp_file in impacted_files:
            commits_to_ignore_current_file = commits_to_ignore.copy()
            try:
                blame_data = self._blame(
                    rev='HEAD^',
                    file_path=imp_file.file_path,
                    modified_lines=imp_file.modified_lines,
                    ignore_revs_file_path=ignore_revs_file_path,
                    ignore_whitespaces=False,
                    skip_comments=False
                )
                to_blame = True
                while to_blame:
                    log.info(f"excluding commits: {params['ignore_revs_list']}")
                    blame_data = self._ag_annotate([imp_file], **params)

                    new_commits_to_ignore = set()
                    new_commits_to_ignore_current_file = set()
                    for bd in blame_data:
                        if bd.commit.hexsha not in new_commits_to_ignore and bd.commit.hexsha not in new_commits_to_ignore_current_file:
                            if bd.commit.hexsha not in commits_to_ignore_current_file:
                                new_commits_to_ignore.update(self._exclude_commits_by_change_size(bd.commit.hexsha,
                                                                                                  max_change_size=max_change_size))
                                new_commits_to_ignore.update(self.get_merge_commits(bd.commit.hexsha))
                                new_commits_to_ignore_current_file.update(
                                    self.select_meta_changes(bd.commit.hexsha, bd.file_path, filter_revert))

                    if len(new_commits_to_ignore) == 0 and len(new_commits_to_ignore_current_file) == 0:
                        to_blame = False
                    elif ts() - start > (60 * 60 * 0.1):  # 1 hour max time
                        log.error(f"blame timeout for {self.repository_path}")
                        to_blame = False

                bug_introd_commits.update([entry.commit for entry in blame_data])
                for commit in bug_introd_commits:
                    num_subsystems, num_modified_directories, entropy = self.calculate_metrics(commit)
                    res_dic['num_subsystems'] = num_subsystems
                    res_dic['num_modified_directories'] = num_modified_directories
                    res_dic['entropy'] = entropy
                    fix_commit = self.repository.commit(fix_commit_hash)
                    age = self.calculate_age(commit)
                    res_dic['age'] = age
                    ndev = self.calculate_ndevelopers(commit)
                    res_dic['ndev'] = ndev
                    add = commit.stats.total['insertions']
                    print(add)
                    deleted = commit.stats.total['deletions']
                    print(deleted)
                    num_files = commit.stats.total['files']
                    print(num_files)
                    lines = commit.stats.total['lines']
                    print(lines)
                    res_dic = self.calculate_author_metrics_optimized(commit)
                    commit_modified_files = list(commit.stats.files.keys())
                    res_dic['lines_of_added'] = add
                    res_dic['lines_of_deleted'] = deleted
                    res_dic['lines_of_modified'] = lines
                    res_dic['num_files'] = num_files
                    res_dic['modified_files'] = commit_modified_files
                    res_dic['is_Friday'] = 1 if commit.committed_datetime.weekday() == 4 else 0
                    print(res_dic)
                    can_feas.append(res_dic)
            #  {'commit': <git.Commit "2574243a39d90a2673cf56647c524e268d7f169e">, 'num_files_changed': 4956, 'lines_of_code_changed': 548857, 'num_of_commits': 1566}
            except:
                print(traceback.format_exc())

        if 'issue_date_filter' in kwargs and kwargs['issue_date_filter']:
            before = len(bug_introd_commits)
            bug_introd_commits = [c for c in bug_introd_commits if c.authored_date <= kwargs['issue_date']]
            log.info(f'Filtering by issue date returned {len(bug_introd_commits)} out of {before}')
        else:
            log.info("Not filtering by issue date.")
        return (bug_introd_commits, can_feas)
