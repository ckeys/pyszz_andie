import re
import sys

import pandas as pd


class BugLabeler:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def label_commits(self):
        self.df['bug_inducing'] = self.df['commit_message'].apply(self._is_bug_fix)
        return self.df

    def _is_bug_fix(self, message):
        # Additional keywords for bug fixing commits
        keywords = ['fix', 'bug']

        # Regular expression patterns for matching commit hashes and BUG-**
        hash_pattern = r'[a-f0-9]{7,}'
        bug_pattern = r'BUG-\d+'

        # Compile regular expressions
        hash_regex = re.compile(hash_pattern, re.IGNORECASE)
        bug_regex = re.compile(bug_pattern, re.IGNORECASE)

        # Check for keywords and patterns in the message
        if any(keyword in message.lower() for keyword in keywords) or \
                re.search(hash_regex, message) or \
                re.search(bug_regex, message):
            return True
        else:
            return False


if __name__ == '__main__':
    csv_file = f'/Users/andie/PycharmProjects/pyszz_andie/commit_analyzer/data/test_data/ad510_decoherence_commit_history_data.csv'
    labeler = BugLabeler(csv_file)
    labeled_commits = labeler.label_commits()
    df = labeled_commits[labeled_commits['bug_inducing'] == True]
    df.to_csv('bug_fix_idendifier.csv')
    bug_fixing_commits_count = labeled_commits[labeled_commits['bug_inducing'] == True].shape[0]
    print("Number of bug fixing commits:", bug_fixing_commits_count)
