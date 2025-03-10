import logging as log
import ntpath
import os
import math
from abc import ABC, abstractmethod
from enum import Enum
from shutil import copytree
from shutil import rmtree
from tempfile import mkdtemp
from typing import List, Set
import time
from datetime import timezone
from git import Commit, Repo
from pydriller import ModificationType, GitRepository as PyDrillerGitRepo

from options import Options
from szz.core.comment_parser import parse_comments
from datetime import datetime, timedelta


class AbstractSZZ(ABC):
    """
    AbstractSZZ is the base class for SZZ implementations. It has core methods for SZZ
    like blame and a diff parsing for impacted files. GitPython is used for base Git
    commands and PyDriller to parse commit modifications.
    """
    szz_variant_name = None

    def __init__(self, repo_full_name: str, repo_url: str, repos_dir: str = None, auto_clean_repo: bool = True):
        """
        Init an abstract SZZ to use as base class for SZZ implementations.
        AbstractSZZ uses a temp folder to clone and interact with the given git repo, where
        the name of the repo folder will be the full name having '/' replaced with '_'.
        The init method also set the default_ignore_regex for modified lines.

        :param str repo_full_name: full name of the Git repository to clone and interact with
        :param str repo_url: url of the Git repository to clone
        :param str repos_dir: temp folder where to clone the given repo
        """
        log.info(f'''Currently working on Task Index: {os.getenv('SLURM_ARRAY_TASK_ID')}''')
        self._repository = None
        self.auto_clean_repo = auto_clean_repo
        if self.auto_clean_repo:
            os.makedirs(Options.TEMP_WORKING_DIR, exist_ok=True)
            self.__temp_dir = mkdtemp(dir=os.path.join(os.getcwd(), Options.TEMP_WORKING_DIR))
        else:
            os.makedirs(Options.TEMP_WORKING_DIR, exist_ok=True)
            tmp_path = os.path.join(os.getcwd(), Options.TEMP_WORKING_DIR)
            self.__temp_dir = os.path.join(tmp_path,
                                           f"tmp_{repo_full_name.split('/')[-1]}_{self.szz_variant_name}_{os.getenv('SLURM_ARRAY_TASK_ID', 0)}")

        log.info(f"Create a temp directory : {self.__temp_dir}")
        self._repository_path = os.path.join(self.__temp_dir, repo_full_name.replace('/', '_'))
        if not os.path.isdir(self._repository_path):
            if repos_dir:
                log.info(f'''[Repo Directory]: {repo_full_name} exists in {repos_dir}''')
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

    def __del__(self):
        if self.auto_clean_repo:
            log.info("cleanup objects...")
            self.__cleanup_repo()
            self.__clear_gitpython()

    @property
    def repository(self) -> Repo:
        """
         Getter of current GitPython Repo object.

         :returns git.Repo repository
        """
        return self._repository

    @property
    def repository_path(self) -> str:
        """
         Getter of current repository local path.

         :returns str repository_path
        """
        return self._repository_path

    @abstractmethod
    def find_bic(self, fix_commit_hash: str, impacted_files: List['ImpactedFile'], **kwargs) -> Set[Commit]:
        """
         Abstract main method to find bug introducing commits. To be implemented by the specific SZZ implementation.

        :param str fix_commit_hash: hash of fix commit to scan for buggy commits
        :param List[ImpactedFile] impacted_files: list of impacted files in fix commit
        :param **kwargs: optional parameters specific for each SZZ implementation
        :returns Set[Commit] a set of bug introducing commits candidates, represented by Commit object
        """
        pass

    def get_impacted_files(self, fix_commit_hash: str,
                           file_ext_to_parse: List[str] = None,
                           only_deleted_lines: bool = True) -> List['ImpactedFile']:
        """
         Parse the diff of given fix commit using PyDriller to obtain a list of ImpactedFile with
         impacted file path and modified line ranges. As default behaviour, all deleted lines in the diff which
         are also added are treated as modified lines.

        :param List[str] file_ext_to_parse: parse only the given file extensions
        :param only_deleted_lines: considers as modified lines only the line numbers that are deleted and added.
            By default, only deleted lines are considered
        :param str fix_commit_hash: hash of fix commit to parse
        :returns List[ImpactedFile] impacted_files
        """
        impacted_files = list()
        try:
            fix_commit = PyDrillerGitRepo(self.repository_path).get_commit(fix_commit_hash)
            for mod in fix_commit.modifications:
                # skip newly added files
                if not mod.old_path:
                    continue

                # filter files by extension
                if file_ext_to_parse:
                    ext = mod.filename.split('.')
                    if len(ext) < 2 or (len(ext) > 1 and ext[1].lower() not in file_ext_to_parse):
                        log.info(f"skip file: {mod.filename}")
                        continue

                file_path = mod.new_path
                if mod.change_type == ModificationType.DELETE or mod.change_type == ModificationType.RENAME:
                    file_path = mod.old_path

                lines_deleted = [deleted[0] for deleted in mod.diff_parsed['deleted']]
                if len(lines_deleted) > 0:
                    impacted_files.append(ImpactedFile(file_path, lines_deleted, LineChangeType.DELETE))

                if not only_deleted_lines:
                    lines_added = [added[0] for added in mod.diff_parsed['added']]
                    if len(lines_added) > 0:
                        impacted_files.append(ImpactedFile(file_path, lines_added, LineChangeType.ADD))

            log.info(impacted_files)
        except ValueError as e:
            log.error(f"{e}")
            return None

        return impacted_files

    def _blame(self, rev: str,
               file_path: str,
               modified_lines: List[int],
               skip_comments: bool = False,
               ignore_revs_list: List[str] = None,
               ignore_revs_file_path: str = None,
               ignore_whitespaces: bool = False,
               detect_move_within_file: bool = False,
               detect_move_from_other_files: 'DetectLineMoved' = None
               ) -> Set['BlameData']:
        """
         Wrapper for Git blame command.

        :param str rev: commit revision
        :param str file_path: path of file to blame
        :param bool modified_lines: list of modified lines that will be converted in line ranges to be used with the param '-L' of git blame
        :param bool ignore_whitespaces: add param '-w' to git blame
        :param bool skip_comments: use a comment parser to identify and exclude line comments and block comments
        :param List[str] ignore_revs_list: specify a list of commits to ignore during blame
        :param bool detect_move_within_file: Detect moved or copied lines within a file
            (-M param of git blame, https://git-scm.com/docs/git-blame#Documentation/git-blame.txt--Mltnumgt)
        :param DetectLineMoved detect_move_from_other_files: Detect lines moved or copied from other files that were modified in the same commit
            (-C param of git blame, https://git-scm.com/docs/git-blame#Documentation/git-blame.txt--Cltnumgt)
        :param str ignore_revs_file_path: specify ignore revs file for git blame to ignore specific commits. The
            file must be in the same format as an fsck.skipList (https://git-scm.com/docs/git-blame)
        :returns Set[BlameData] a set of bug introducing commits candidates, represented by BlameData object
        """

        kwargs = dict()
        if ignore_whitespaces:
            kwargs['w'] = True
        if ignore_revs_file_path:
            kwargs['ignore-revs-file'] = ignore_revs_file_path
        if ignore_revs_list:
            kwargs['ignore-rev'] = list(ignore_revs_list)
        if detect_move_within_file:
            kwargs['M'] = True
        if detect_move_from_other_files and detect_move_from_other_files == DetectLineMoved.SAME_COMMIT:
            kwargs['C'] = True
        if detect_move_from_other_files and detect_move_from_other_files == DetectLineMoved.PARENT_COMMIT:
            kwargs['C'] = [True, True]
        if detect_move_from_other_files and detect_move_from_other_files == DetectLineMoved.ANY_COMMIT:
            kwargs['C'] = [True, True, True]

        bug_introd_commits = set()
        mod_line_ranges = self._parse_line_ranges(modified_lines)
        log.info(f"processing file: {file_path}")
        for entry in self.repository.blame_incremental(**kwargs, rev=rev, L=mod_line_ranges, file=file_path):
            # entry.linenos = input lines to blame (current lines)
            # entry.orig_lineno = output line numbers from blame (previous commit lines from blame)
            for line_num in entry.orig_linenos:
                source_file_content = self.repository.git.show(f"{entry.commit.hexsha}:{entry.orig_path}")
                line_str = source_file_content.split('\n')[line_num - 1].strip()
                b_data = BlameData(entry.commit, line_num, line_str, entry.orig_path)

                if skip_comments and self._is_comment(line_num, source_file_content, ntpath.basename(b_data.file_path)):
                    log.info(f"skip comment line ({line_num}): {line_str}")
                    continue

                # log.info(b_data)
                bug_introd_commits.add(b_data)

        return bug_introd_commits

    def _parse_line_ranges(self, modified_lines: List) -> List[str]:
        """
        Convert impacted lines list to list of modified lines range. In case of single line,
        the range will be the same line as start and end - ['line_num, line_num', 'start, end', ...]

        :param str modified_lines: list of modified lines
        :returns List[str] impacted_lines_ranges
        """
        mod_line_ranges = list()

        if len(modified_lines) > 0:
            start = int(modified_lines[0])
            end = int(modified_lines[0])

            if len(modified_lines) == 1:
                return [f'{start},{end}']

            for i in range(1, len(modified_lines)):
                line = int(modified_lines[i])
                if line - end == 1:
                    end = line
                else:
                    mod_line_ranges.append(f'{start},{end}')
                    start = line
                    end = line

                if i == len(modified_lines) - 1:
                    mod_line_ranges.append(f'{start},{end}')

        return mod_line_ranges

    def _is_comment(self, line_num: int, source_file_content: str, source_file_name: str) -> bool:
        """
        Check if the given line is a comment. It uses a specific comment parser which returns the interval of line
        numbers containing comments - CommentRange(start, end)

        :param int line_num: line number
        :param str source_file_content: The content of the file to parse
        :param str source_file_name: The name of the file to parse
        :returns bool
        """

        comment_ranges = parse_comments(source_file_content, source_file_name, self.__temp_dir)

        for comment_range in comment_ranges:
            if comment_range.start <= line_num <= comment_range.end:
                return True
        return False

    def _calculate_metrics(self, commit):
        modified_directories = set()
        subsystems = set()

        for file_path in self._get_touched_files(commit):
            # Extract root directory name
            root_directory = file_path.split('/')[0]
            modified_directories.add(file_path.split('/')[0])

            # Extract subsystem (root directory name)
            subsystems.add(root_directory)

        num_subsystems = len(subsystems)
        num_modified_directories = len(modified_directories)

        entropy = self._calculate_entropy(commit)
        return num_subsystems, num_modified_directories, entropy

    def _get_touched_files(self, commit):
        return [diff.a_path for diff in commit.diff(commit.parents[0])]

    def _get_latest_modified_date_before_commit(self, commit, file_path):
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

    def _calculate_age(self, commit):
        """
        Calculate the age of modifications in the commit.
        """
        touched_files = self._get_touched_files(commit)
        commit_date = datetime.fromtimestamp(commit.committed_date)
        total_interval = timedelta(0)
        file_count = 0

        for file_path in touched_files:
            last_modified_date = self._get_latest_modified_date_before_commit(commit, file_path)
            if last_modified_date:
                time_interval = commit_date - last_modified_date
                total_interval += time_interval
                file_count += 1

        if file_count == 0:
            return 0

        average_age = total_interval.seconds / file_count
        return average_age

    def _calculate_ndevelopers(files, commit):
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

    def _calculate_REXP(self, developer_changes, commit):
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

    def _calculate_SEXP(self, developer_changes, subsystem_changes):
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

    def _calculate_entropy(self, commit):
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

    def _calculate_author_metrics_optimized(self, commit):
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
        REXP = self._calculate_REXP(developer_changes, commit)
        # Calculate SEXP
        SEXP = self._calculate_SEXP(developer_changes, subsystem_changes)

        log.info(f"It takes {(end_time - start_time) / 60} minutes to run the historical commits analyse!")
        return {"commit": commit_num, "exp_of_files": len(hist_modified_files),
                "exp_of_codes": exp_changed_lines, "exp_of_commits": len(commit_hashes),
                "REXP": REXP, "SEXP": SEXP}

    def _set_working_tree_to_commit(self, commit: str):
        # self.repository.head.reference = self.repository.commit(fix_commit_hash)
        # reset the index and working tree to match the pointed-to commit
        try:
            # Reset the index and working tree to match the pointed-to commit
            self.repository.head.reset(commit=commit, index=True, working_tree=True)
            assert not self.repository.head.is_detached
        except Exception as e:
            # Log the error and skip processing for this commit
            print(f"Error resetting to commit {commit}: {e}")

    def _get_impacted_file_content(self, fix_commit_hash: str, impacted_file: 'ImpactedFile') -> str:
        return self.repository.git.show(f"{fix_commit_hash}:{impacted_file.file_path}")

    def get_commit(self, hash: str) -> Commit:
        """ return the Commit object for the given hash """
        return self.repository.commit(hash)

    def __cleanup_repo(self):
        """ Cleanup of local repository used by SZZ """
        log.info(f"Clean up {self.__temp_dir}")
        if os.path.isdir(self.__temp_dir):
            rmtree(self.__temp_dir)

    def __clear_gitpython(self):
        """ Cleanup of GitPython due to memory problems """
        if self._repository:
            self._repository.close()
            self._repository.__del__()


class DetectLineMoved(Enum):
    """
    DetectLineMoved represents the -C param of git blame (https://git-scm.com/docs/git-blame#Documentation/git-blame.txt--Cltnumgt),
    which detect lines moved or copied from other files that were modified in the same commit. The default [<num>] param
    of alphanumeric characters to detect is used (i.e. 40).

    * SAME_COMMIT = -C
    * PARENT_COMMIT = -C -C
    * ANY_COMMIT = -C -C -C
    """
    SAME_COMMIT = 1
    PARENT_COMMIT = 2
    ANY_COMMIT = 3


class LineChangeType(Enum):
    """
    Git line change type

    * ADD = the line is only added
    * MODIFY = the line is deleted and then added
    * DELETE = the line is deleted, but also can be modified
    """
    ADD = 1
    MODIFY = 2
    DELETE = 3


class ImpactedFile:
    """ Data class to represent impacted files """

    def __init__(self, file_path: str, modified_lines: List[int], line_change_type: 'LineChangeType'):
        """
        :param str file_path: previous path of the current impacted file
        :param List[int] modified_lines: list of modified lines
        :param 'LineChangeType' line_change_type: the type of change performed in the modified lines
        :returns ImpactedFile
        """
        self.file_path = file_path
        self.modified_lines = modified_lines
        self.line_change_type = line_change_type

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(file_path="{self.file_path}",modified_lines={self.modified_lines},line_change_type={self.line_change_type})'


class BlameData:
    """ Data class to represent blame data """

    def __init__(self, commit: Commit, line_num: int, line_str: str, file_path: str):
        """
        :param Commit commit: commit detected by git blame
        :param int line_num: number of the blamed line
        :param str line_str: content of the blamed line
        :param str file_path: path of the blamed file
        :returns BlameData
        """
        self.commit = commit
        self.line_num = line_num
        self.line_str = line_str
        self.file_path = file_path

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(commit={self.commit.hexsha},line_num={self.line_num},file_path="{self.file_path}",line_str="{self.line_str}")'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.file_path == other.file_path and self.line_num == other.line_num

    def __hash__(self) -> int:
        return 31 * hash(self.line_num) + hash(self.file_path)
