#!/bin/bash

# Directory containing the log files
LOG_DIR="/home/huayo708/projects/pyszz_andie/slurm_jobs"

# Find the highest index from existing log files
LAST_INDEX=$(ls $LOG_DIR/mlszz_p*.log 2>/dev/null | sed -n 's/.*mlszz_p\([0-9]\+\)\.log/\1/p' | sort -n | tail -1)

# Initialize index if no log files are found
if [[ -z $LAST_INDEX ]]; then
  LAST_INDEX=2
fi

# Determine the last value written from the latest error file
if [[ $LAST_INDEX -ge 2 ]]; then
  LAST_ERR_FILE="$LOG_DIR/mlszz_p${LAST_INDEX}.err"
  PARAM_VALUE=$(grep "Write " $LAST_ERR_FILE | tail -1 | sed -n 's/.*Write \([0-9]\+\),.*/\1/p')
else
  PARAM_VALUE=5  # Default value if no previous error file exists
fi

# Increment the index
NEXT_INDEX=$((LAST_INDEX + 1))

# Set job name, log file names, and error file names with the incremented index
JOB_NAME="mlszz_p${NEXT_INDEX}"
LOG_FILE="$LOG_DIR/mlszz_p${NEXT_INDEX}.log"
ERR_FILE="$LOG_DIR/mlszz_p${NEXT_INDEX}.err"

echo $JOB_NAME
echo $LOG_FILE
echo $ERR_FILE

#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_FILE
#SBATCH --error=$ERR_FILE

# Command to run
/usr/bin/python /home/huayo708/projects/pyszz_andie/main.py /home/huayo708/projects/pyszz_andie/in/bugfix_commits_all.json /home/huayo708/projects/pyszz_andie/conf/mlszz.yml /home/huayo708/projects/repo $PARAM_VALUE