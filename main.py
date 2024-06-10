import json
import logging as log
import os
import sys
import dateparser
from time import time as ts

import yaml

from szz.ag_szz import AGSZZ
from szz.b_szz import BaseSZZ
from szz.l_szz import LSZZ
from szz.ml_szz import MLSZZ
from szz.ma_szz import MASZZ, DetectLineMoved
from szz.r_szz import RSZZ
from szz.ra_szz import RASZZ

log.basicConfig(level=log.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
log.getLogger('pydriller').setLevel(log.WARNING)


def main(input_json: str, out_json: str, conf: dict(), repos_dir: str, start_index: int, end_index: int = None):
    szz_name = conf['szz_name']
    with open(input_json, 'r') as in_file:
        bugfix_commits = json.loads(in_file.read())
    tot = len(bugfix_commits)
    if end_index is None:
        end_index = len(bugfix_commits)

    write_file_index = end_index // (end_index - start_index)
    write_file_name = f"output_{szz_name}/output_write_{write_file_index}.log"
    if not os.path.exists(f'''output_{szz_name}'''):
        os.makedirs(f'''output_{szz_name}''')
    # Check if file exists, and clear its content if it does
    if os.path.exists(write_file_name):
        with open(write_file_name, 'w'):
            pass  # This clears the content of the file

    w_file = open(write_file_name, 'a')

    with open(out_json, 'w') as file:
        log.info(f'''Start at : {start_index} and end at : {end_index}, total length : {len(bugfix_commits[start_index:end_index])}''')
        bic_dict = None
        for i, commit in enumerate(bugfix_commits[start_index:end_index], start=start_index):
            log.info(f'''Repo Directory is {repos_dir} and Repo Name is {commit['repo_name']}''')

            # if not os.path.exists(f'''{repos_dir}/{commit['repo_name']}'''):
            #     log.info(f'''The path is not existing {repos_dir}/{commit['repo_name']}''')
            #     log.info(f'Skipping {commit["repo_name"]}')
            #     continue

            bug_introducing_commits = set()
            repo_name = commit['repo_name']
            repo_url = f'https://test:test@github.com/{repo_name}.git'  # using test:test as git login to skip private repos during clone
            # username = 'ckeys'
            # st = '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'
            # repo_url = f'''https://{username}:{st}@github.com/{repo_name}.git'''
            fix_commit = commit['fix_commit_hash']

            log.info(f'{i + 1} of {tot}: {repo_name} {fix_commit}')

            commit_issue_date = None
            if conf.get('issue_date_filter', None):
                commit_issue_date = (
                        commit.get('earliest_issue_date', None) or commit.get('best_scenario_issue_date', None))
                commit_issue_date = dateparser.parse(commit_issue_date).timestamp()
            if szz_name == 'b':
                b_szz = BaseSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = b_szz.get_impacted_files(fix_commit_hash=fix_commit,
                                                     file_ext_to_parse=conf.get('file_ext_to_parse'),
                                                     only_deleted_lines=conf.get('only_deleted_lines', True))
                (bug_introducing_commits, bic_dict) = b_szz.find_bic_v2(fix_commit_hash=fix_commit,
                                                                        impacted_files=imp_files,
                                                                        ignore_revs_file_path=conf.get(
                                                                            'ignore_revs_file_path'),
                                                                        issue_date_filter=conf.get('issue_date_filter'),
                                                                        issue_date=commit_issue_date)
                # print(bic_dict)
            elif szz_name == 'ag':
                ag_szz = AGSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = ag_szz.get_impacted_files(fix_commit_hash=fix_commit,
                                                      file_ext_to_parse=conf.get('file_ext_to_parse'),
                                                      only_deleted_lines=conf.get('only_deleted_lines', True))
                bug_introducing_commits = ag_szz.find_bic(fix_commit_hash=fix_commit,
                                                          impacted_files=imp_files,
                                                          ignore_revs_file_path=conf.get('ignore_revs_file_path'),
                                                          max_change_size=conf.get('max_change_size'),
                                                          issue_date_filter=conf.get('issue_date_filter'),
                                                          issue_date=commit_issue_date)

            elif szz_name == 'ma':
                ma_szz = MASZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = ma_szz.get_impacted_files(fix_commit_hash=fix_commit,
                                                      file_ext_to_parse=conf.get('file_ext_to_parse'),
                                                      only_deleted_lines=conf.get('only_deleted_lines', True))
                bug_introducing_commits = ma_szz.find_bic(fix_commit_hash=fix_commit,
                                                          impacted_files=imp_files,
                                                          ignore_revs_file_path=conf.get('ignore_revs_file_path'),
                                                          max_change_size=conf.get('max_change_size'),
                                                          detect_move_from_other_files=DetectLineMoved(
                                                              conf.get('detect_move_from_other_files')),
                                                          issue_date_filter=conf.get('issue_date_filter'),
                                                          issue_date=commit_issue_date)

            elif szz_name == 'r':
                r_szz = RSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = r_szz.get_impacted_files(fix_commit_hash=fix_commit,
                                                     file_ext_to_parse=conf.get('file_ext_to_parse'),
                                                     only_deleted_lines=conf.get('only_deleted_lines', True))
                bug_introducing_commits = r_szz.find_bic(fix_commit_hash=fix_commit,
                                                         impacted_files=imp_files,
                                                         ignore_revs_file_path=conf.get('ignore_revs_file_path'),
                                                         max_change_size=conf.get('max_change_size'),
                                                         detect_move_from_other_files=DetectLineMoved(
                                                             conf.get('detect_move_from_other_files')),
                                                         issue_date_filter=conf.get('issue_date_filter'),
                                                         issue_date=commit_issue_date)

            elif szz_name == 'l':
                l_szz = LSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = l_szz.get_impacted_files(fix_commit_hash=fix_commit,
                                                     file_ext_to_parse=conf.get('file_ext_to_parse'),
                                                     only_deleted_lines=conf.get('only_deleted_lines', True))
                bug_introducing_commits = l_szz.find_bic(fix_commit_hash=fix_commit,
                                                         impacted_files=imp_files,
                                                         ignore_revs_file_path=conf.get('ignore_revs_file_path'),
                                                         max_change_size=conf.get('max_change_size'),
                                                         detect_move_from_other_files=DetectLineMoved(
                                                             conf.get('detect_move_from_other_files')),
                                                         issue_date_filter=conf.get('issue_date_filter'),
                                                         issue_date=commit_issue_date)
            elif szz_name == 'ra':
                ra_szz = RASZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = ra_szz.get_impacted_files(fix_commit_hash=fix_commit,
                                                      file_ext_to_parse=conf.get('file_ext_to_parse'),
                                                      only_deleted_lines=conf.get('only_deleted_lines', True))
                bug_introducing_commits = ra_szz.find_bic(fix_commit_hash=fix_commit,
                                                          impacted_files=imp_files,
                                                          ignore_revs_file_path=conf.get('ignore_revs_file_path'),
                                                          max_change_size=conf.get('max_change_size'),
                                                          detect_move_from_other_files=DetectLineMoved(
                                                              conf.get('detect_move_from_other_files')),
                                                          issue_date_filter=conf.get('issue_date_filter'),
                                                          issue_date=commit_issue_date)
            elif szz_name == 'ml':
                ml_szz = MLSZZ(repo_full_name=repo_name, repo_url=repo_url, repos_dir=repos_dir)
                imp_files = ml_szz.get_impacted_files(fix_commit_hash=fix_commit,
                                                      file_ext_to_parse=conf.get('file_ext_to_parse'),
                                                      only_deleted_lines=conf.get('only_deleted_lines', True))
                if imp_files is None:
                    continue
                (bug_introducing_commits, bic_dict) = ml_szz.find_bic_v2(fix_commit_hash=fix_commit,
                                                                         impacted_files=imp_files,
                                                                         ignore_revs_file_path=conf.get(
                                                                             'ignore_revs_file_path'),
                                                                         issue_date_filter=conf.get(
                                                                             'issue_date_filter'),
                                                                         issue_date=commit_issue_date)
            else:
                log.info(f'SZZ implementation not found: {szz_name}')
                exit(-3)

            log.info(f"result: {bug_introducing_commits}")
            bugfix_commits[i]["inducing_commit_hash"] = [bic.hexsha for bic in bug_introducing_commits if bic]
            if bic_dict is not None:
                bugfix_commits[i]["candidate_features"] = bic_dict
            # log.info(f'''Write {i}, {bugfix_commits[i]['id']}: {bugfix_commits[i]} ''')
            w_file.write(f'''Write {i}, {bugfix_commits[i]['id']}: {bugfix_commits[i]} ''')
            file.write(json.dumps(bugfix_commits[i]) + '\n')
    out2_json = os.path.join('out', f'bic_{szz_name}_{int(ts())}.json')
    with open(out2_json, 'w') as out:
        log.info(f'''Writing Results to {out2_json}!''')
        json.dump(bugfix_commits, out)
    w_file.close()
    log_files = [f for f in os.listdir(f'''output_{szz_name}''') if f.startswith("output_write_") and f.endswith(".log")]
    # Combine content of individual log files into summary file
    with open(f"output_{szz_name}/output_write_summary.log", "w") as summary_file:
        for log_file in log_files:
            with open(f"output_{szz_name}/{log_file}", "r") as file:
                summary_file.write(file.read()+"\n")
    # Remove individual log files
    for log_file in log_files:
        os.remove(f"output_{szz_name}/{log_file}")
    log.info(f"+++ DONE +++")



if __name__ == "__main__":
    if (len(sys.argv) > 0 and '--help' in sys.argv[1]) or len(sys.argv) < 3:
        print('USAGE: python main.py <bugfix_commits.json> <conf_file path> <repos_directory(optional)>')
        print('If repos_directory is not set, pyszz will download each repository')
        exit(-1)
    input_json = sys.argv[1]
    conf_file = sys.argv[2]
    repos_dir = sys.argv[3] if len(sys.argv) > 3 else None
    repos_dir = None
    start_index = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    end_index = int(sys.argv[5]) if len(sys.argv) > 5 else None

    if not os.path.isfile(input_json):
        log.error('invalid input json')
        exit(-2)
    if not os.path.isfile(conf_file):
        log.error('invalid conf file')
        exit(-2)

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    log.info(f"parsed conf yml: {conf}")
    szz_name = conf['szz_name']

    out_dir = 'out'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_json = os.path.join(out_dir, f'bic_{szz_name}_{int(ts())}.json')

    if not szz_name:
        log.error('The configuration file does not define the SZZ name. Please, fix.')
        exit(-3)

    log.info(f'Launching {szz_name}-szz')

    main(input_json, out_json, conf, repos_dir, start_index, end_index)
