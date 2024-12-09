import pandas as pd


def process_data(data):
    # List to store flattened rows
    rows = []

    # Iterate through each entry in the data
    for entry in data:
        print(f'''[DEBUG]: {entry}''')
        # Extract metadata
        id_ = entry['id']
        repo_name = entry['repo_name']
        fix_commit_hash = entry['fix_commit_hash']
        bug_commit_hash = ','.join(entry['bug_commit_hash'])  # Convert list to string
        best_scenario_issue_date = entry.get('best_scenario_issue_date', None)
        language = ','.join(entry.get('language', []))  # Convert list to string
        inducing_commit_hash = ','.join(entry['inducing_commit_hash'])  # Convert list to string

        # Iterate through candidate features
        for feature in entry['candidate_features']:
            # Create a row combining metadata and feature data
            row = {
                'id': id_,
                'repo_name': repo_name,
                'fix_commit_hash': fix_commit_hash,
                'bug_commit_hash': bug_commit_hash,
                'best_scenario_issue_date': best_scenario_issue_date,
                'language': language,
                'inducing_commit_hash': inducing_commit_hash,
                **feature  # Merge feature dictionary
            }
            rows.append(row)

    # Convert rows to a pandas DataFrame
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Example data
    # data = [
    #     {'id': 258, 'repo_name': 'wmanley/stb-tester', 'fix_commit_hash': '0984374d44cac0ab9fd3418c124a6c33483d4801',
    #      'bug_commit_hash': ['b8d39d434bd374f46cef4c5970a486e6c1b60de6'],
    #      'best_scenario_issue_date': '2014-04-02T11:33:24', 'language': ['py'],
    #      'inducing_commit_hash': ['2ee7b00324e6347f5f31101be3a9945b4f6b1afc',
    #                               'a189e0c962ae07b72726ac5a5998eb949883f193',
    #                               '587b70e683d74a26cd5dad5098b33f1abfa6a4e1',
    #                               'cf57a4c2c0c2711520106a1a97f96b3a9e419f90',
    #                               'b8d39d434bd374f46cef4c5970a486e6c1b60de6'], 'candidate_features': [
    #         {'num_subsystems': 8, 'num_modified_directories': 3, 'entropy': 3.169925001442312, 'LT': 299.22222222222223,
    #          'FIX': 0, 'age': 1.735659722222222, 'ndev': 10, 'nuc': 285,
    #          'commit': '2ee7b00324e6347f5f31101be3a9945b4f6b1afc', 'exp_of_files': 62, 'exp_of_codes': 20840,
    #          'EXP': 341, 'REXP': 264.0, 'SEXP': 265.0, 'lines_of_added': 60, 'lines_of_deleted': 5,
    #          'lines_of_modified': 65, 'num_files': 9,
    #          'modified_files': ['.travis.yml', 'README.rst', 'extra/stb-tester.spec', 'stbt-completion', 'stbt-run',
    #                             'stbt.conf', 'stbt.py', 'tests/stbt.conf', 'tests/test-stbt-py.sh'],
    #          'candidate_commit_to_fix': 16642, 'is_Friday': 0, 'is_latest_bic': 0, 'is_largest_mod': 0},
    #         {'num_subsystems': 3, 'num_modified_directories': 2, 'entropy': 1.584962500721156, 'LT': 507.3333333333333,
    #          'FIX': 1, 'age': 0.066875, 'ndev': 10, 'nuc': 186, 'commit': 'a189e0c962ae07b72726ac5a5998eb949883f193',
    #          'exp_of_files': 62, 'exp_of_codes': 21428, 'EXP': 355, 'REXP': 278.0, 'SEXP': 192.0, 'lines_of_added': 42,
    #          'lines_of_deleted': 58, 'lines_of_modified': 100, 'num_files': 3,
    #          'modified_files': ['stbt.conf', 'stbt.py', 'tests/stbt.conf'], 'candidate_commit_to_fix': 75970,
    #          'is_Friday': 0, 'is_latest_bic': 0, 'is_largest_mod': 0},
    #         {'num_subsystems': 1, 'num_modified_directories': 1, 'entropy': 0.0, 'LT': 1481.0, 'FIX': 0,
    #          'age': 6.944444444444444e-05, 'ndev': 10, 'nuc': 185, 'commit': '587b70e683d74a26cd5dad5098b33f1abfa6a4e1',
    #          'exp_of_files': 62, 'exp_of_codes': 21591, 'EXP': 364, 'REXP': 287.0, 'SEXP': 140.0, 'lines_of_added': 86,
    #          'lines_of_deleted': 89, 'lines_of_modified': 175, 'num_files': 1, 'modified_files': ['stbt.py'],
    #          'candidate_commit_to_fix': 427, 'is_Friday': 1, 'is_latest_bic': 0, 'is_largest_mod': 1},
    #         {'num_subsystems': 1, 'num_modified_directories': 1, 'entropy': 0.0, 'LT': 1042.0, 'FIX': 1,
    #          'age': 2.1687962962962963, 'ndev': 5, 'nuc': 94, 'commit': 'cf57a4c2c0c2711520106a1a97f96b3a9e419f90',
    #          'exp_of_files': 37, 'exp_of_codes': 8074, 'EXP': 172, 'REXP': 95.0, 'SEXP': 63.0, 'lines_of_added': 5,
    #          'lines_of_deleted': 4, 'lines_of_modified': 9, 'num_files': 1, 'modified_files': ['stbt.py'],
    #          'candidate_commit_to_fix': 73676, 'is_Friday': 0, 'is_latest_bic': 0, 'is_largest_mod': 0},
    #         {'num_subsystems': 6, 'num_modified_directories': 4, 'entropy': 3.169925001442312, 'LT': 441.77777777777777,
    #          'FIX': 1, 'age': 1.0081134259259257, 'ndev': 13, 'nuc': 411,
    #          'commit': 'b8d39d434bd374f46cef4c5970a486e6c1b60de6', 'exp_of_files': 86, 'exp_of_codes': 11167, 'EXP': 90,
    #          'REXP': 82.5, 'SEXP': 86.0, 'lines_of_added': 40, 'lines_of_deleted': 17, 'lines_of_modified': 57,
    #          'num_files': 9, 'modified_files': ['README.rst', 'docs/backwards-compatibility.md', 'extra/vm/README.rst',
    #                                             'extra/vm/setup-vagrant-user.sh', 'stbt-tv', 'stbt.py',
    #                                             'tests/run-performance-test.sh', 'tests/stbt.conf',
    #                                             'tests/test-stbt-py.sh'], 'candidate_commit_to_fix': 13348,
    #          'is_Friday': 0, 'is_latest_bic': 1, 'is_largest_mod': 0}]}]
    #
    # # Process data
    # df = process_data(data)
    #
    # # Display the DataFrame
    # print(df)
    #
    # # Save to CSV (optional)
    # df.to_csv('processed_data.csv', index=False)
    #
    # # data2 = [{'id': 4, 'repo_name': 'ahobson/ruby-pcap', 'fix_commit_hash': '0ad41d0684c2ec4c2a6b604f7aafbaf9f0459dcc',
    # #           'bug_commit_hash': ['272f03ff3b5bf79829f80c2febd004904d64006e'],
    # #           'best_scenario_issue_date': '2011-06-01T04:05:04', 'language': ['rb']}]
    # #
    # # # Process data
    # # df = process_data(data2)
    #
    # # Display the DataFrame
    # print(df)

    data3 = [{'id': 54, 'repo_name': 'jwang36/edk2', 'fix_commit_hash': '9bf86b12ce04fcb7d997eb1fc8c55fd43d18ab79',
              'bug_commit_hash': ['bf9e636605188e291d33ab694ff1c5926b6f0800'],
              'earliest_issue_date': '2018-12-09T13:34:00', 'language': ['py'],
              'inducing_commit_hash': ['bf9e636605188e291d33ab694ff1c5926b6f0800'], 'candidate_features': [
            {'num_subsystems': 1, 'num_modified_directories': 3, 'entropy': 2.321928094887362, 'LT': 1393.0, 'FIX': 1,
             'age': 0.0004976851851851852, 'ndev': 28, 'nuc': 252, 'commit': 'bf9e636605188e291d33ab694ff1c5926b6f0800',
             'exp_of_files': 11, 'exp_of_codes': 1354, 'EXP': 18, 'REXP': 17.5, 'SEXP': 0.0, 'lines_of_added': 88,
             'lines_of_deleted': 5, 'lines_of_modified': 93, 'num_files': 5,
             'modified_files': ['BaseTools/Source/Python/Common/Expression.py',
                                'BaseTools/Source/Python/Common/Misc.py',
                                'BaseTools/Source/Python/CommonDataClass/CommonClass.py',
                                'BaseTools/Source/Python/Workspace/BuildClassObject.py',
                                'BaseTools/Source/Python/Workspace/DscBuildData.py'], 'candidate_commit_to_fix': 34948,
             'is_Friday': 1, 'is_latest_bic': 1, 'is_largest_mod': 1}]}]

    # Process data
    df = process_data(data3)

    # Display the DataFrame
    print(df)
