#!/bin/bash

# Directory containing the log files
LOG_DIR="/home/huayo708/projects/pyszz_andie/slurm_jobs"

# Find the highest index from existing log files
LAST_INDEX=$(ls $LOG_DIR/mlszz_p*.log 2>/dev/null | sed -n 's/.*mlszz_p\([0-9]\+\)\.log/\1/p' | sort -n | tail -1)

# Initialize index if no log files are found
if [[ -z $LAST_INDEX ]]; then
  LAST_INDEX=2
fi

# Increment the index
NEXT_INDEX=$((LAST_INDEX + 1))

# Set job name, log file names, and error file names with the incremented index
JOB_NAME="mlszz_p${NEXT_INDEX}"
LOG_FILE="$LOG_DIR/mlszz_p${NEXT_INDEX}.log"
ERR_FILE="$LOG_DIR/mlszz_p${NEXT_INDEX}.err"

#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_FILE
#SBATCH --error=$ERR_FILE

# Command to run
/usr/bin/python /home/huayo708/projects/pyszz_andie/main.py /home/huayo708/projects/pyszz_andie/in/bugfix_commits_all.json /home/huayo708/projects/pyszz_andie/conf/mlszz.yml /home/huayo708/projects/repo 5