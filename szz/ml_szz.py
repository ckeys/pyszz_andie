import logging as log
import traceback
import math
import re
import subprocess
from datetime import timezone
from collections import defaultdict
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
from collections import Counter
from operator import attrgetter


class MLSZZ(AGSZZ):

    def __init__(self, repo_full_name: str, repo_url: str, repos_dir: str = None):
        super().__init__(repo_full_name, repo_url, repos_dir)

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
        cwd = self.repository.working_dir
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
        commit = self.repository.commit(commit_hash)
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
        cwd = self.repository.working_dir
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
        hist_commit_hashes = self.repository.git.execute(rev_list_command).strip().split('\n')[1:]
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
            commit_obj = self.repository.commit(commit_hash)
            diff_stat_tmp = commit_obj.stats
            diff_filenames_tmp = commit_obj.stats.files
            diff_filenames = list(diff_filenames_tmp.keys())
            hist_modified_files.update(diff_filenames)
            # changes = diff_stat.split(',')
            exp_added_lines += diff_stat_tmp.total['insertions']
            exp_removed_lines += diff_stat_tmp.total['deletions']
            developer_changes.append(
                {'date': self.repository.commit(commit_hash).committed_datetime, 'num_changes': 1,
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
                    deleted = commit.stats.total['deletions']
                    num_files = commit.stats.total['files']
                    lines = commit.stats.total['lines']
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

    def get_commit_changes(self, commit: Commit):
        # Get the parent commit
        if not commit.parents:
            raise Exception("The commit has no parents, it's the initial commit.")
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
        # #
        # # modified_lines = []
        # #
        # # # Calculate modified lines in each file and collect them
        # # for file_path, stats in commit.stats.files.items():
        # #     modified_lines.append(stats['insertions'] + stats['deletions'])
        # #
        # # total_modified_lines = sum(modified_lines)
        # #
        # # # Calculate entropy
        # # entropy = 0
        # # for lines in modified_lines:
        # #     if lines > 0:
        # #         entropy -= lines / total_modified_lines * math.log2(lines / total_modified_lines)
        #
        # return entropy

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
        return [diff.a_path if diff.a_path else diff.b_path for diff in commit.diff(commit.parents[0])]

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
            list_of_developers = self.repository.git.execute(git_command).strip().split('\n')[1:]
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

    def get_touched_files(self, commit):
        return commit.stats.files.keys()

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

    def get_lines_of_code_before_change(self, commit: Commit):

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
                file_content = parent_commit.tree[file_path].data_stream.read().decode('utf-8')
                # TODO： need to remove annotations and remove files that are not source codes
                lines_of_code = len(file_content.splitlines())
            except KeyError:
                # File did not exist in the parent commit
                lines_of_code = 0

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
            list_of_commits = self.repository.git.execute(git_command).strip().split('\n')[1:]
            for c in list_of_commits:
                unique_last_changes.add(c)
        return len(unique_last_changes)

    def find_bic_v2(self, fix_commit_hash: str, impacted_files: List['ImpactedFile'], **kwargs) -> Tuple[
        Set[Commit], List[Dict]]:
        """
                Find bug introducing commits candidates.

                :param str fix_commit_hash: hash of fix commit to scan for buggy commits
                :param List[ImpactedFile] impacted_files: list of impacted files in fix commit
                :key ignore_revs_file_path (str): specify ignore revs file for git blame to ignore specific commits.
                :returns Set[Commit] a set of bug introducing commits candidates, represented by Commit object
                """
        bug_introd_commits = set()
        commits_to_ignore = set()
        try:
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
            params['detect_move_from_other_files'] = kwargs.get('detect_move_from_other_files',
                                                                DetectLineMoved.SAME_COMMIT)
            params['ignore_revs_list'] = list()
            if kwargs.get('blame_rev_pointer', None):
                params['rev_pointer'] = kwargs['blame_rev_pointer']
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
                #  {'commit': <git.Commit "2574243a39d90a2673cf56647c524e268d7f169e">, 'num_files_changed': 4956, 'lines_of_code_changed': 548857, 'num_of_commits': 1566}
                except:
                    print(traceback.format_exc())

            latest_bic = None

            if len(bug_introd_commits) > 0:
                latest_bic = max(bug_introd_commits, key=attrgetter('committed_date'))
                max_modified_lines = max(commit.stats.total['lines'] for commit in bug_introd_commits)
                earliest_bic = min(bug_introd_commits, key=attrgetter('committed_date'))

            for commit in bug_introd_commits:
                res_dic = dict()
                num_subsystems, num_modified_directories, entropy = self.calculate_diffusion_metrics(commit)
                res_dic['num_subsystems'] = num_subsystems # SCRIPT
                res_dic['num_modified_directories'] = num_modified_directories # SCRIPT
                res_dic['entropy'] = entropy # SCRIPT
                res_dic['LT'] = self.get_lines_of_code_before_change(commit) # SCRIPT
                res_dic['FIX'] = self.purpose_of_change(commit) # SCRIPT
                age = self.calculate_age(commit) # SCRIPT
                res_dic['age'] = age # SCRIPT
                ndev = self.calcualte_ndev(commit) # SCRIPT
                res_dic['ndev'] = ndev # SCRIPT
                res_dic['nuc'] = self.calculate_nuc(commit) # SCRIPT
                add = commit.stats.total['insertions']
                deleted = commit.stats.total['deletions']
                num_files = commit.stats.total['files']
                lines = commit.stats.total['lines']
                experience_dict = self.calculate_author_metrics_optimized(commit) #exp, rexp, sexp
                # This creates a new dictionary
                res_dic.update(experience_dict)
                commit_modified_files = list(commit.stats.files.keys())
                res_dic['lines_of_added'] = add
                res_dic['lines_of_deleted'] = deleted
                res_dic['lines_of_modified'] = lines
                res_dic['num_files'] = num_files
                res_dic['modified_files'] = commit_modified_files
                res_dic['candidate_commit_to_fix'] = abs(
                    (self.get_commit_time(fix_commit_hash) - self.get_commit_time(commit.hexsha)).seconds)
                res_dic['is_Friday'] = 1 if commit.committed_datetime.weekday() == 4 else 0
                res_dic[
                    'is_latest_bic'] = 1 if latest_bic is not None and latest_bic.hexsha == commit.hexsha else 0
                res_dic['is_largest_mod'] = 1 if lines == max_modified_lines else 0
                res_dic[
                    'is_earliest_bic'] = 1 if earliest_bic is not None and earliest_bic.hexsha == commit.hexsha else 0
                print(res_dic)
                can_feas.append(res_dic)

            if 'issue_date_filter' in kwargs and kwargs['issue_date_filter']:
                before = len(bug_introd_commits)
                bug_introd_commits = [c for c in bug_introd_commits if c.authored_date <= kwargs['issue_date']]
                log.info(f'Filtering by issue date returned {len(bug_introd_commits)} out of {before}')
            else:
                log.info("Not filtering by issue date.")
        except Exception as e:
            log.error("Error:", e)
        return (bug_introd_commits, can_feas)
