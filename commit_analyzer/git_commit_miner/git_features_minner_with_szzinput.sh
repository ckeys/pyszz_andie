#!/bin/bash

# =============================================================================
# Script: mine_git_features.sh
# Description: Extracts comprehensive commit features from Git repositories and
#              saves them into individual CSV and JSON files for each project.
# =============================================================================

# ----------------------------- Configuration ------------------------------

# Base directories
BASE_PROJECT_DIR="/Users/andie/PycharmProjects"                  # Directory containing Git repositories
FEATURES_DIR="/Users/andie/features"                 # Directory to store feature CSV and JSON files

# Ensure the FEATURES_DIR exists
mkdir -p "$FEATURES_DIR"

# List of project names (repositories)
# PROJECTS=("react" "spring-boot" "spring-framework" "cpython" "bitcoin")  # Add your project names here
PROJECTS=("pig")
# Define required file extensions for nf (Number of Files)
REQUIRED_EXTENSIONS=("py" "java" "c" "cpp" "js" "cxx" "hpp" "h" "sh" "bash" "kt" "kts")  # Modify as needed

# Define keywords for PurposeFeatures (fix detection)
FIX_KEYWORDS=("bug" "fix" "wrong" "error" "fail" "problem" "patch")

# ---------------------------- Helper Functions ----------------------------

# Function to compute entropy using awk
compute_entropy() {
    local counts=("$@")
    echo "${counts[@]}" | awk '
    {
        n = NF
        total = 0
        for(i=1;i<=n;i++) {
            total += $i
        }
        if(total == 0) {
            print 0
            exit
        }
        entropy = 0
        for(i=1;i<=n;i++) {
            if($i > 0) {
                p = $i / total
                entropy -= p * log(p)
            }
        }
        print entropy
    }'
}

# Function to compute rexp using awk
compute_rexp() {
    local current_time=$1
    shift
    local past_times=("$@")
    echo "${past_times[@]}" | awk -v current_time="$current_time" '
    {
        rexp = 0
        for(i=1;i<=NF;i++) {
            denominator = (current_time - $i) / 86400 / 365 + 1
            if(denominator > 0) {
                rexp += 1 / denominator
            }
        }
        print rexp
    }'
}

# Function to display progress bar
# Arguments: current_step, total_steps
show_progress() {
    local current=$1
    local total=$2
    local progress=$(( current * 100 / total ))
    local bar_length=50
    local filled_length=$(( progress * bar_length / 100 ))
    local empty_length=$(( bar_length - filled_length ))
    # Build the bar string
    local bar=""
    for ((i=0; i<filled_length; i++)); do
        bar="${bar}#"
    done
    for ((i=0; i<empty_length; i++)); do
        bar="${bar}-"
    done
    # Print the progress bar
    printf "\rProgress: |%s| %d%% (%d/%d)" "$bar" "$progress" "$current" "$total"
}

# Function to check if a file is in required extensions
in_our_extensions() {
    local file="$1"
    for ext in "${REQUIRED_EXTENSIONS[@]}"; do
        if [[ "$file" == *.$ext ]]; then
            echo "$ext"
        fi
    done
}

# Function to extract features from a single commit
extract_features() {
    local commit_id=$1
    local project=$2
    local author_email=$3
    local time_stamp=$4
    local modified_files="$5"
    local commit_message="$6"
    local modified_numstat="$7"

    # --------------------- Feature Calculations --------------------------

    # 1. Number of Subsystems (ns)
    # Assuming subsystem is the top-level directory
    ns=$(echo "$modified_files" | awk -F/ '{print $1}' | sort | uniq | wc -l | tr -d ' ')

    # 2. Number of Directories (nd)
    # Extract directories by removing the file names
    nd=$(echo "$modified_files" | awk -F/ 'NF>1 {OFS="/"; NF--; print $0}' | sort | uniq | wc -l | tr -d ' ')

    # 3. Number of Files with Required Extensions (nf)
    nf=0
    ext_counts=()
    for ext in "${REQUIRED_EXTENSIONS[@]}"; do
        count=$(echo "$modified_files" | grep -E "\.${ext}$" | wc -l | tr -d ' ')
        ext_counts+=("$count")
        nf=$((nf + count))
    done

    # 4. Entropy Calculation based on file extension distribution
    entropy=$(compute_entropy "${ext_counts[@]}")

    # 5. Experience Features (exp, rexp, sexp)

    # Initialize associative arrays if not already
    if [[ -z "${exp_dict[$author_email]}" ]]; then
        exp_dict["$author_email"]=0
    fi
    if [[ -z "${rexp_dict[$author_email]}" ]]; then
        rexp_dict["$author_email"]=0
    fi

    # exp: Number of previous commits by the author
    exp=${exp_dict["$author_email"]}

    # rexp: Cumulative experience based on past commit timestamps
    if [[ -z "${changes_dict["$author_email"]}" ]]; then
        changes_dict["$author_email"]=""
    fi
    # Convert past timestamps to an array
    IFS=' ' read -r -a past_times <<< "${changes_dict["$author_email"]}"
    # Compute rexp using awk
    if [ ${#past_times[@]} -gt 0 ]; then
        rexp=$(compute_rexp "$time_stamp" "${past_times[@]}")
    else
        rexp=0
    fi

    # sexp: Sum of previous modifications of subsystems by the author
    sexp=0
    for subsys in $modified_subsystems; do
        key="${author_email}:${subsys}"
        if [[ -z "${sexp_dict["$key"]}" ]]; then
            sexp_dict["$key"]=0
        fi
        sexp=$((sexp + sexp_dict["$key"]))
    done

    # Update experience dictionaries
    # Update exp
    exp_dict["$author_email"]=$((exp_dict["$author_email"] + 1))

    # Update rexp
    rexp_dict["$author_email"]=$(awk -v a="${rexp_dict["$author_email"]}" -v b="$rexp" 'BEGIN { printf "%.6f", a + b }')

    # Update changes_dict with current timestamp
    if [ -z "${changes_dict["$author_email"]}" ]; then
        changes_dict["$author_email"]="$time_stamp"
    else
        changes_dict["$author_email"]="${changes_dict["$author_email"]} $time_stamp"
    fi

    # Update sexp_dict for each subsystem
    for subsys in $modified_subsystems; do
        key="${author_email}:${subsys}"
        sexp_dict["$key"]=$((sexp_dict["$key"] + 1))
    done

    # 6. History Features (ndev, age, nuc)
    # Initialize History Features
    ndev=0
    age=0
    nuc=0

    # Initialize temporary arrays for developers and changes
    declare -A temp_dev_set
    declare -A temp_change_set

    # Iterate over modified files to gather developers and changes
    while IFS= read -r line; do
        # Each line has: added deleted file_path
        added=$(echo "$line" | awk '{print $1}')
        deleted=$(echo "$line" | awk '{print $2}')
        file_path=$(echo "$line" | awk '{print $3}')

        # Extract developers and changes from file_stats
        developers=${file_stats["$file_path,developers"]}
        changes=${file_stats["$file_path,changes"]}

        if [[ -n "$developers" ]]; then
            IFS=',' read -r -a devs <<< "$developers"
            for dev in "${devs[@]}"; do
                temp_dev_set["$dev"]=1
            done
        fi

        if [[ -n "$changes" ]]; then
            IFS=',' read -r -a chgs <<< "$changes"
            for chg in "${chgs[@]}"; do
                temp_change_set["$chg"]=1
            done
        fi

        # Calculate age (in days)
        last_age=${file_stats["$file_path,last_age"]}
        if [[ -n "$last_age" ]]; then
            file_age=$(( (time_stamp - last_age) / 86400 ))  # Age in days
            age=$((age + file_age))
        fi
    done <<< "$modified_numstat"

    # Calculate ndev and nuc
    ndev=$(echo "${!temp_dev_set[@]}" | tr ' ' '\n' | sort | uniq | wc -l | tr -d ' ')
    nuc_count=$(echo "${!temp_change_set[@]}" | tr ' ' '\n' | sort | uniq | wc -l | tr -d ' ')
    if [ "$nf" -gt 0 ]; then
        nuc=$(awk "BEGIN {printf \"%.6f\", $nuc_count / $nf}")
        age=$(awk "BEGIN {printf \"%.6f\", $age / $nf}")
    else
        nuc=0
        age=0
    fi

    # 7. Purpose Features (fix)
    fix=0
    commit_message_lower=$(echo "$commit_message" | tr '[:upper:]' '[:lower:]')
    for keyword in "${FIX_KEYWORDS[@]}"; do
        if [[ "$commit_message_lower" == *"$keyword"* ]]; then
            fix=1
            break
        fi
    done

    # 8. Size Features (la, ld, lt)
    # 8. Size Features (la, ld, lt)
    la=0
    ld=0
    lt=0

    while IFS= read -r line; do
        # Each line has: added deleted file_path
        added=$(echo "$line" | awk '{print $1}')
        deleted=$(echo "$line" | awk '{print $2}')
        file_path=$(echo "$line" | awk '{print $3}')
        file_ext=$(in_our_extensions "$file_path")
        if [[ -n "$file_ext" ]]; then
            # Handle '-' by converting to 0
            if [[ "$added" == "-" ]]; then
                added=0
            fi
            if [[ "$deleted" == "-" ]]; then
                deleted=0
            fi

             # Ensure 'added' and 'la' are numeric
            if [[ "$added" =~ ^[0-9]+$ ]] && [[ "$la" =~ ^[0-9]+$ ]]; then
                 la=$(( la + added ))
            else
                 echo "WARNING: Non-numeric value encountered for 'added' or 'la'. Setting to 0." >&2
                 la=0
             fi

             # Similarly handle 'deleted' and 'ld'
             if [[ "$deleted" =~ ^[0-9]+$ ]] && [[ "$ld" =~ ^[0-9]+$ ]]; then
                ld=$(( ld + deleted ))
             else
                 echo "WARNING: Non-numeric value encountered for 'deleted' or 'ld'. Setting to 0." >&2
                 ld=0
             fi
         fi
     done <<< "$modified_numstat"

    if [ "$nf" -gt 0 ]; then
        lt=$(awk "BEGIN {printf \"%.6f\", ($la + $ld) / $nf}")
    else
        lt=0
    fi

    # ----------------------- Output Features ------------------------------

    # Prepare the CSV line
    csv_line="$project,$commit_id,$ns,$nd,$nf,$entropy,$exp,$rexp,$sexp,$ndev,$age,$nuc,$fix,$la,$ld,$lt"

    # Output to CSV
    echo "$csv_line" >> "$OUTPUT_FILE"

    # ----------------------- Output JSON for SZZ ----------------------------

    if [ "$fix" -eq 1 ]; then
        if [ "$is_first_fix" -eq 1 ]; then
            # First fix commit, no comma
            echo "    {" >> "$JSON_FILE"
            is_first_fix=0
        else
            # Subsequent fix commits, add comma before the object
            echo "    }," >> "$JSON_FILE"
            echo "    {" >> "$JSON_FILE"
        fi

        # Extract unique languages from modified files
        language_array=$(echo "$modified_files" | awk -F. '/\./ {print tolower($NF)}' | grep -E '^(py|java|c|cpp|js)$' | sort | uniq | awk '{printf "\"%s\",", $0}' | sed 's/,$//')

        # Write JSON object
        echo "        \"id\": $json_id," >> "$JSON_FILE"
        echo "        \"repo_name\": \"$project\"," >> "$JSON_FILE"
        echo "        \"fix_commit_hash\": \"$commit_id\"," >> "$JSON_FILE"
        echo "        \"language\": [${language_array}]" >> "$JSON_FILE"

        # Increment JSON ID
        json_id=$((json_id + 1))
    fi
}

# ----------------------------- Main Logic ----------------------------------

# Declare associative arrays for tracking experience and history
declare -A exp_dict        # author_email -> exp
declare -A rexp_dict       # author_email -> rexp
declare -A sexp_dict       # "author_email:subsys" -> count
declare -A changes_dict    # author_email -> "timestamp1 timestamp2 ..."
declare -A file_stats      # "file_path,attribute" -> value (for HistoryFeatures)

# Iterate over each project
for project in "${PROJECTS[@]}"; do
    echo "Processing project: $project"

    # Define the output CSV and JSON files for the current project
    OUTPUT_FILE="${FEATURES_DIR}/${project}_git_features.csv"
    JSON_FILE="${FEATURES_DIR}/${project}_szz_input.json"

    # Initialize JSON file with opening bracket
    echo "[" > "$JSON_FILE"
    is_first_fix=1
    json_id=1

    # Navigate to the project's Git repository
    cd "$BASE_PROJECT_DIR/$project" || {
        echo "Error: Invalid project path for $project"
        exit 1
    }

    # Retrieve all commit IDs in chronological order along with author_email, timestamp, and commit message
    # Format: commit_id|author_email|timestamp|commit_message
    commit_info=$(git log --pretty=format:"%H|%ae|%at|%s" --reverse)

    # Convert commit_info to an array
    IFS=$'\n' read -rd '' -a commit_array <<< "$commit_info"

    # Total number of commits
    total_commits=${#commit_array[@]}
    echo "Total commits found: $total_commits"

    # Write CSV headers for the current project
    echo "project,commit_id,ns,nd,nf,entropy,exp,rexp,sexp,ndev,age,nuc,fix,la,ld,lt" > "$OUTPUT_FILE"

    # Initialize progress
    current_commit=0

    # Process each commit
    for commit_entry in "${commit_array[@]}"; do
        # Increment commit counter
        current_commit=$((current_commit + 1))

        # Split commit_entry into commit_id, author_email, timestamp, and commit_message
        IFS='|' read -r commit_id author_email time_stamp commit_message <<< "$commit_entry"

        # Get list of modified files in the commit along with lines added and deleted
        # Format per line: added deleted file_path
        modified_numstat=$(git show --pretty="" --numstat "$commit_id")

        # Get list of modified files for ns, nd, nf
        modified_files=$(echo "$modified_numstat" | awk '{print $3}')

        # Skip commits with no changes
        if [ -z "$modified_files" ]; then
            continue
        fi

        # Extract modified subsystems (top-level directories)
        modified_subsystems=$(echo "$modified_files" | awk -F/ '{print $1}' | sort | uniq)

        # Display which commit is being processed
        echo "Processing commit $current_commit/$total_commits: $commit_id"

        # Extract features from the commit
        extract_features "$commit_id" "$project" "$author_email" "$time_stamp" "$modified_files" "$commit_message" "$modified_numstat"

        # Update the progress bar
        show_progress "$current_commit" "$total_commits"
    done

    # After processing all commits, close the JSON array
    if [ "$is_first_fix" -eq 0 ]; then
        echo "    }" >> "$JSON_FILE"
    fi
    echo "]" >> "$JSON_FILE"

    # Move to a new line after the progress bar completes
    echo

    echo "Completed processing project: $project"
done

echo "Feature extraction completed. Results saved to $FEATURES_DIR"