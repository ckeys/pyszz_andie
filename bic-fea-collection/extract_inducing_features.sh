#!/bin/bash

# =============================================================================
# Script: extract_inducing_features.sh
# Description: Processes a JSON lines file to extract features from inducing_commit_hash list.
#              For each inducing_commit_hash, determines:
#                - is_Friday: Whether the commit was made on Friday.
#                - is_latest_bic: Whether the commit is the latest among the inducing_commit_hash list.
#                - is_largest_mod: Whether the commit has the largest modification lines.
#                - is_earliest_bic: Whether the commit is the earliest among the inducing_commit_hash list.
#
# Configuration:
#   - INPUT_FILE: Path to the input JSON lines file.
#   - REPO_PATH: Path to the Git repository.
#   - OUTPUT_DIR: Directory where the output CSV will be saved.
#
# Usage:
#   1. Set the INPUT_FILE, REPO_PATH, and OUTPUT_DIR variables below.
#   2. Make the script executable:
#        chmod +x extract_inducing_features.sh
#   3. Run the script:
#        ./extract_inducing_features.sh
# =============================================================================

# ----------------------------- Configuration ------------------------------

LOCAL_PROJECT_NAME="bitcoin"
# Path to the input JSON lines file
INPUT_FILE="/home/$(whoami)/andie/pyszz_andie/shjobs/out/bic_b_$(LOCAL_PROJECT_NAME).json"

# Path to the Git repository
REPO_PATH="/home/$(whoami)/andie/repo/$(LOCAL_PROJECT_NAME)"

# Directory to save the output CSV file
OUTPUT_DIR="/home/$(whoami)/andie/features"

# Name of the output CSV file
OUTPUT_FILE="$(LOCAL_PROJECT_NAME)_inducing_commits_features.csv"

# ---------------------------- Helper Functions ----------------------------

# Function to display usage (if needed in future enhancements)
usage() {
    echo "Usage: $0"
    echo "Ensure that INPUT_FILE, REPO_PATH, and OUTPUT_DIR are set in the script."
    exit 1
}

# Function to extract commit date in ISO format
get_commit_date() {
    local commit_hash="$1"
    git show -s --format=%cd --date=iso "$commit_hash"
}

# Function to get total lines added and deleted in a commit
get_commit_modifications() {
    local commit_hash="$1"
    # Sum lines added and deleted, ignoring binary files ('-')
    local added=$(git show --numstat --pretty="" "$commit_hash" | awk '$1 ~ /^[0-9]+$/ {sum += $1} END {print sum}')
    local deleted=$(git show --numstat --pretty="" "$commit_hash" | awk '$2 ~ /^[0-9]+$/ {sum += $2} END {print sum}')
    # Handle cases where added or deleted is empty
    added=${added:-0}
    deleted=${deleted:-0}
    echo "$added" "$deleted"
}

# Function to calculate candidate_commit_to_fix
calculate_candidate_commit_to_fix() {
    local inducing_hash="$1"
    local fix_hash="$2"

    # Check if inducing_hash is an ancestor of fix_hash
    if git merge-base --is-ancestor "$inducing_hash" "$fix_hash"; then
        # Calculate number of commits from inducing_hash to fix_hash (excluding fix_hash)
        local count=$(git rev-list "$inducing_hash".."$fix_hash" --count)
        echo "$count"
    else
        # If inducing_hash is not an ancestor, set as N/A
        echo "N/A"
    fi
}

# ----------------------------- Main Logic ----------------------------------
echo "Main Logic Starts"
# Ensure that the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

# Ensure that the Git repository path exists
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "Error: Git repository not found at '$REPO_PATH'."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Full path to the output CSV file
FULL_OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"
echo "Initialize output CSV with headers"
# Initialize output CSV with headers
echo "fix_commit_hash,inducing_commit_hash,is_Friday,is_latest_bic,is_earliest_bic,is_largest_mod,candidate_commit_to_fix,lines_of_modified_code" > "$FULL_OUTPUT_PATH"

# Change to the Git repository directory
cd "$REPO_PATH" || { echo "Error: Cannot change directory to '$REPO_PATH'."; exit 1; }
echo "Start to read file line"
# Read the input file line by line
#while IFS= read -r line; do
jq -c '.[]' "$INPUT_FILE" | while IFS= read -r line; do
    # Extract fix_commit_hash and inducing_commit_hash list using jq
    echo "Extract fix_commit_hash and inducing_commit_hash list"
    fix_commit_hash=$(echo "$line" | jq -r '.fix_commit_hash')
    inducing_commit_hashes=$(echo "$line" | jq -r '.inducing_commit_hash[]?')
    echo "Convert inducing_commit_hashes to an array"
    # Convert inducing_commit_hashes to an array
    readarray -t inducing_hash_array <<< "$inducing_commit_hashes"
   
    # If there are inducing commits
    if [ ${#inducing_hash_array[@]} -gt 0 ]; then
        # Initialize associative arrays to hold commit data
        declare -A commit_weekdays
        declare -A commit_lines
        declare -A commit_timestamps

        # Variables to track latest, earliest, and largest modification
        latest_time=0
        latest_commit=""
        earliest_time=9999999999
        earliest_commit=""
        largest_mod=0
        largest_mod_commit=""

        # Loop through inducing commits to collect data
        echo "Loop through inducing commits to collect data"
	for inducing_hash in "${inducing_hash_array[@]}"; do
            # Trim whitespace
            inducing_hash=$(echo "$inducing_hash" | xargs)

            # Check if inducing_hash is not empty
            if [ -z "$inducing_hash" ]; then
                echo "WARNING: Encountered an empty inducing_commit_hash for fix_commit_hash '$fix_commit_hash'. Skipping this entry." >&2
                continue
            fi

            # Validate commit hash
            if ! git cat-file -e "${inducing_hash}^{commit}" 2>/dev/null; then
                echo "WARNING: Commit hash '$inducing_hash' does not exist in the repository. Skipping." >&2
                continue
            fi

            # Get commit date
            commit_date=$(get_commit_date "$inducing_hash")
            # Convert commit date to timestamp
            commit_timestamp=$(date -d "$commit_date" +%s)
            commit_timestamps["$inducing_hash"]=$commit_timestamp

            # Get the weekday (1=Monday, ..., 7=Sunday)
            commit_weekday=$(date -d "$commit_date" +%u)  # 5=Friday

            # Set is_Friday flag
            if [ "$commit_weekday" -eq 5 ]; then
                commit_weekdays["$inducing_hash"]=1
            else
                commit_weekdays["$inducing_hash"]=0
            fi

            # Get total lines added and deleted
            read added deleted <<< "$(get_commit_modifications "$inducing_hash")"
            commit_lines["$inducing_hash"]=$((added + deleted))

            # Update latest_commit
            if [ "$commit_timestamp" -gt "$latest_time" ]; then
                latest_time=$commit_timestamp
                latest_commit="$inducing_hash"
            fi

            # Update earliest_commit
            if [ "$commit_timestamp" -lt "$earliest_time" ]; then
                earliest_time=$commit_timestamp
                earliest_commit="$inducing_hash"
            fi

            # Update largest_mod_commit
            if [ "${commit_lines["$inducing_hash"]}" -gt "$largest_mod" ]; then
                largest_mod="${commit_lines["$inducing_hash"]}"
                largest_mod_commit="$inducing_hash"
            fi
        done

        # Now, loop again through inducing_hash_array to set flags
        for inducing_hash in "${inducing_hash_array[@]}"; do
            # Trim whitespace
            inducing_hash=$(echo "$inducing_hash" | xargs)

            # Check if commit was processed
            if [ -z "${commit_timestamps["$inducing_hash"]:-}" ]; then
                # Commit was skipped due to missing or empty hash
                continue
            fi

            # Get is_Friday flag
            is_Friday=${commit_weekdays["$inducing_hash"]}

            # Determine is_latest_bic
            if [ "$inducing_hash" == "$latest_commit" ]; then
                is_latest_bic=1
            else
                is_latest_bic=0
            fi

            # Determine is_earliest_bic
            if [ "$inducing_hash" == "$earliest_commit" ]; then
                is_earliest_bic=1
            else
                is_earliest_bic=0
            fi

            # Determine is_largest_mod
            if [ "$inducing_hash" == "$largest_mod_commit" ]; then
                is_largest_mod=1
            else
                is_largest_mod=0
            fi

            # Calculate candidate_commit_to_fix
            candidate_commit_to_fix=$(calculate_candidate_commit_to_fix "$inducing_hash" "$fix_commit_hash")

            # Get lines_of_modified_code
            lines_of_modified_code=${commit_lines["$inducing_hash"]}

            # Append the data to the output CSV
            echo "$fix_commit_hash,$inducing_hash,$is_Friday,$is_latest_bic,$is_earliest_bic,$is_largest_mod,$candidate_commit_to_fix,$lines_of_modified_code" >> "$FULL_OUTPUT_PATH"
        done

        # Unset associative arrays for the next iteration to avoid data leakage
        unset commit_weekdays
        unset commit_lines
        unset commit_timestamps
    fi

#done < "$INPUT_FILE"
done
echo "Extraction completed. Results saved to '$FULL_OUTPUT_PATH'."
