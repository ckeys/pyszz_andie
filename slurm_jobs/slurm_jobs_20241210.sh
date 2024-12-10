#!/bin/bash
#SBATCH --job-name=srcml_job            # Base job name
#SBATCH --mem=8G                        # Memory allocation
#SBATCH --output=srcml_output.%j.log     # Base output log file name with job ID
#SBATCH --error=srcml_error.%j.log       # Base error log file name with job ID
#SBATCH --partition=aoraki              # Partition name (adjust as needed)
#SBATCH --time=03:00:00                 # Maximum time allocation

# Load required modules
module purge                             # Purge any loaded modules to avoid conflicts
module load srcml                        # Load the srcml module

# Load the Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate otagophd

# Directory containing the log files
LOG_DIR="/home/huayo708/projects/pyszz_andie/slurm_jobs/logs"

# Remove the log directory if it exists, then recreate it
if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

# Find the highest index from existing log files
LAST_INDEX=$(ls $LOG_DIR/mlszz_p*.log 2>/dev/null | sed -n 's/.*mlszz_p\([0-9]\+\)\.log/\1/p' | sort -n | tail -1)

# Initialize index if no log files are found
if [[ -z $LAST_INDEX ]]; then
  LAST_INDEX=0
fi

# Set default PARAM_VALUE in case there is no previous error file
PARAM_VALUE=5

# Determine the last value written from the latest error file if it exists
LAST_ERR_FILE="$LOG_DIR/mlszz_p${LAST_INDEX}.err"
if [[ -f "$LAST_ERR_FILE" ]]; then
  PARAM_VALUE=$(grep "Write " "$LAST_ERR_FILE" | tail -1 | sed -n 's/.*Write \([0-9]\+\),.*/\1/p')
fi

# Set the number of jobs to submit
START_AT=2
NUM_JOBS=10  # Change this to the desired number of jobs
BATCH_SIZE=50  # Size of each batch (i.e., 0-999 for the first job, 1000-1999 for the second job, etc.)

JOB_INDEX=0
# Loop to create and submit jobs
for ((i=2; i<START_AT+NUM_JOBS; i++)); do
  # Calculate start_index and end_index for each job
  start_index=$((i * BATCH_SIZE))
  end_index=$(((i + 1) * BATCH_SIZE - 1))

  # Set job name, log file names, and error file names with the incremented index
  JOB_NAME="mlszz_p${JOB_INDEX}"
  LOG_FILE="$LOG_DIR/mlszz_p${JOB_INDEX}.log"
  ERR_FILE="$LOG_DIR/mlszz_p${JOB_INDEX}.err"

  # Command to run the Python script with dynamic start_index and end_index
  CMD="/home/huayo708/miniforge3/envs/otagophd/bin/python /home/huayo708/projects/pyszz_andie/main.py /home/huayo708/projects/pyszz_andie/in/bugfix_commits_all.json /home/huayo708/projects/pyszz_andie/conf/mlszz.yml /home/huayo708/projects/repo $start_index $end_index"

  # Submit the job with dynamic job name, log file, and error file
  sbatch --job-name="$JOB_NAME" --output="$LOG_FILE" --error="$ERR_FILE" --wrap="$CMD"

  # Optionally, you can print information about the job that was submitted
  echo "Submitted job: $JOB_NAME with start_index=$start_index, end_index=$end_index, log: $LOG_FILE and error: $ERR_FILE"
  # Find the highest index from existing log files
  JOB_INDEX=$((JOB_INDEX+1))
done