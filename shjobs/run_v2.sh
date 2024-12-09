# Check if MLSZZ_PATH is set
if [ -z "$MLSZZ_PATH" ]; then
  echo "MLSZZ_PATH environment variable is not set"
  exit 1
fi

# Ensure the logs directory exists
logs_dir="logs"
if [ ! -d "$logs_dir" ]; then
  echo "Creating logs directory..."
  mkdir -p "$logs_dir"
fi

# Base command
base_command="nohup python $MLSZZ_PATH/main.py $MLSZZ_PATH/in/valid_project.json $MLSZZ_PATH/conf/mlszz.yml None"

# Job parameters
start=0
increment=100
num_jobs=20  # Total number of jobs

# Create jobs
for ((i=0; i<num_jobs; i++)); do
  # Calculate start and end for the job
  end=$((start + increment - 1))

  # Log file name inside the logs directory
  log_file="$logs_dir/output_${start}_${end}.log"

  # Construct the command
  full_command="$base_command $start $end > $log_file 2>&1 &"

  # Print and execute the command
  echo "Running job $i: $full_command"
  eval $full_command

  # Update the start for the next job
  start=$((end + 1))
done
