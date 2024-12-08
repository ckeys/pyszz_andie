import re
import sys
import os

import pandas as pd
import argparse


class BugLabeler:
    def __init__(self, csv_file):
        if csv_file.endswith('.csv'):
            self.df = pd.read_csv(csv_file)
        elif csv_file.endswith('.json'):
            self.df = pd.read_json(csv_file)

    def label_commits(self):
        self.df['is_bug_fixing'] = self.df['commit_message'].apply(self._is_bug_fix)
        return self.df

    def _is_bug_fix(self, message):
        # Expanded keywords and phrases for bug-fixing commits
        keywords = [
            'fix', 'bug', 'patch', 'issue', 'resolve', 'resolved',
            'repair', 'correct', 'error', 'defect', 'hotfix'
        ]

        # Regular expression patterns for commit hashes, BUG-**, and common formats
        hash_pattern = r'\b[a-f0-9]{7,}\b'
        bug_pattern = r'\bBUG-\d+\b'
        issue_pattern = r'\b(issue|ticket|case|problem)-?\d+\b'
        resolve_phrases = r'\b(fixes|resolves|closes|addresses|repairs)\b'

        # Compile regex
        hash_regex = re.compile(hash_pattern, re.IGNORECASE)
        bug_regex = re.compile(bug_pattern, re.IGNORECASE)
        issue_regex = re.compile(issue_pattern, re.IGNORECASE)
        resolve_regex = re.compile(resolve_phrases, re.IGNORECASE)

        # Normalize the message
        message = message.lower()

        # Check for keywords
        keyword_found = any(keyword in message for keyword in keywords)

        # Check for regex patterns
        pattern_found = bool(
            hash_regex.search(message) or
            bug_regex.search(message) or
            issue_regex.search(message) or
            resolve_regex.search(message)
        )

        # Strengthen the detection by combining keyword and pattern checks
        return keyword_found or pattern_found


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze commit history and label bug-inducing commits.')
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to the input CSV file containing commit history data.'
    )
    args = parser.parse_args()

    # Extract project name from input file
    input_file = args.input
    base_name = os.path.basename(input_file)
    project_name = base_name.rsplit('_commit_history_data.csv', 1)[0]


    # Generate output file name
    output_file = f'{project_name}_bug_fix_identifier.csv'
    # Create output folder if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate output file path
    output_file = os.path.join(output_dir, f'{project_name}_bug_fix_identifier.json')

    # Analyze commits
    labeler = BugLabeler(input_file)
    labeled_commits = labeler.label_commits()
    df = labeled_commits
    # Filter bug-inducing commits and save the results
    bug_fixing_df = labeled_commits[labeled_commits['is_bug_fixing'] == True]
    df.to_json(output_file, orient='records', lines=True)

    # Print the count of bug-fixing commits
    total_commits = df.shape[0]
    print(f'''Total commits: {total_commits}''')
    bug_fixing_commits_count = bug_fixing_df.shape[0]
    print(f"Number of bug fixing commits: {bug_fixing_commits_count}")
    print(f"Results saved to: {output_file}")