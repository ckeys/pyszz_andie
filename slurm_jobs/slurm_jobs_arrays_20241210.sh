#!/bin/bash
#SBATCH --job-name=srcml_job            # Base job name
#SBATCH --mem=32G                        # Memory allocation
#SBATCH --output=./slurmlogs/srcml_output.%A_%a.log # Output log file name with job array ID
#SBATCH --error=./slurmlogs/srcml_error.%A_%a.log   # Error log file name with job array ID
#SBATCH --partition=aoraki              # Partition name (adjust as needed)
#SBATCH --array=38-100                    # Define an array with 50 tasks (adjust as needed)
#SBATCH --nodes=1                       # Require exactly 1 node per task
#SBATCH --ntasks=1                      # One task per job
#SBATCH --time=1-12:00:00               # 1 day, 12 hours

# Check if the $HOME directory is available and writable
while [ ! -d "$HOME" ] || [ ! -w "$HOME" ]; do
  echo "Home directory $HOME not available or writable, waiting..."
  sleep 10
done
echo "Home directory $HOME is now available."

# Load required modules
module purge                             # Purge any loaded modules to avoid conflicts
module load srcml                        # Load the srcml module

# Load the Conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate otagophd

# Directory containing the log files
LOG_DIR="/home/huayo708/projects/pyszz_andie/slurm_jobs/logs"
mkdir -p "$LOG_DIR"                     # Ensure the log directory exists

# Batch configuration
BATCH_SIZE=10                           # Size of each batch

# Set job-specific parameters based on SLURM_ARRAY_TASK_ID
JOB_INDEX=$SLURM_ARRAY_TASK_ID          # Use SLURM array ID for unique job indexing
start_index=$((JOB_INDEX * BATCH_SIZE)) # Calculate start index
end_index=$(((JOB_INDEX + 1) * BATCH_SIZE - 1)) # Calculate end index

# Job-specific log files
JOB_NAME="mlszz_p_${JOB_INDEX}"
LOG_FILE="$LOG_DIR/mlszz_p_${start_index}_${end_index}.log"
ERR_FILE="$LOG_DIR/mlszz_p_${start_index}_${end_index}.err"

# Command to run the Python script with dynamic start_index and end_index
CMD="/home/huayo708/miniforge3/envs/otagophd/bin/python /home/huayo708/projects/pyszz_andie/main.py \
     /home/huayo708/projects/pyszz_andie/in/valid_project.json \
     /home/huayo708/projects/pyszz_andie/conf/mlszz.yml \
     /home/huayo708/projects/repo \
     $start_index $end_index"

# Print job details (optional)
echo "Starting job: $JOB_NAME"
echo "Indices: start_index=$start_index, end_index=$end_index"
echo "Log: $LOG_FILE, Error log: $ERR_FILE"
echo "Command: $CMD"

# Execute the command
$CMD