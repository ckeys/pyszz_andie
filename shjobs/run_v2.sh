# Step 1: setup MLSZZ_PATH
# Step 2: Setup SZZ_V
# Step 3: Setup input.json
# Step 4: Setup repo directory
# Check if MLSZZ_PATH is set
#  conda activate andieenv
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=MCZbase && ./run_v2.sh
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=Metazone && ./run_v2.sh
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=odoo && ./run_v2.sh
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=operations-puppet && ./run_v2.sh
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=Parrot && ./run_v2.sh
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=systemd && ./run_v2.sh
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=tor_1 && ./run_v2.sh
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=tor && ./run_v2.sh
## export MLSZZ_PATH=/home/huayo708/andie/pyszz_andie SZZ_V=all PROJECT_NAME=linux && ./run_v2.sh
# conda activate andieenv
if [ -z "$MLSZZ_PATH" ]; then
  echo "MLSZZ_PATH environment variable is not set"
  exit 1
fi

if [ -z "$SZZ_V" ]; then
  echo "SZZ_V environment variable is not set"
  exit 1
fi

if [ -z "$PROJECT_NAME" ]; then
  echo "PROJECT_NAME environment variable is not set"
  exit 1
fi

# Ensure the logs directory exists
logs_dir="logs/$PROJECT_NAME"
if [ ! -d "$logs_dir" ]; then
  echo "Creating logs directory..."
  mkdir -p "$logs_dir"
fi

# Base command
# base_command="nohup python $MLSZZ_PATH/main.py $MLSZZ_PATH/in/react_szz_input.json $MLSZZ_PATH/conf/$SZZ_V.yml $HOME/andie/repo"
szz_input="$MLSZZ_PATH/in/${PROJECT_NAME}_szz_input.json"
base_command="nohup python $MLSZZ_PATH/main.py $MLSZZ_PATH/in/${PROJECT_NAME}_szz_input.json"



# Job parameters
start=0
# increment=7000
num_jobs=1  # Total number of jobs
method="$SZZ_V"

# Get the length of the JSON array (assuming the array is at the root of the JSON)

array_length=$(jq '. | length' "$szz_input")
echo "The length of the JSON array is ${array_length}"
# Dynamically set the increment (you can adjust this as needed)
increment=$((array_length / 1))  # Example: divide the array length by X for job distribution
# If increment is 0, set it to 1 to avoid issues
if [ "$increment" -le 0 ]; then
  increment=1
fi
echo "The increment is ${increment}"
# Create jobs
#for ((i=0; i<num_jobs; i++)); do
#  # Calculate start and end for the job
#  end=$((start + increment - 1))
#
#  # Log file name inside the logs directory
#  log_file="$logs_dir/${method}_output_${start}_${end}.log"
#
#  # Construct the command
#  full_command="$base_command $start $end > $log_file 2>&1 &"
#
#  # Print and execute the command
#  echo "Running job $i: $full_command"
#  eval $full_command
#
#  # Update the start for the next job
#  start=$((end + 1))
# done
#

# Function to run the job for a specific method
run_job() {
  method=$1
  start=$2
  end=$3
  log_file=$4

  # Construct the command
  full_command="$base_command $MLSZZ_PATH/conf/${method}.yml $HOME/andie/repo $start $end > $log_file 2>&1 &"

  # Print and execute the command
  echo "Running job for method $method: $full_command"
  eval $full_command
}

if [ "$SZZ_V" == "all" ]; then
  # List of methods to execute if SZZ_V is 'all'
  methods=("bszz" "maszz" "rszz" "lszz")

  # Run jobs concurrently for each method
  for method in "${methods[@]}"; do
    end=$((start + increment - 1))
    # Log file name inside the logs directory
    log_file="$logs_dir/${PROJECT_NAME}_${method}_output_${start}_${end}.log"

    # Run the job for each method in the background
    run_job $method $start $end $log_file

    # Update the start for the next job
    #start=$((end + 1))
    #done
  done

  # Wait for all background jobs to complete
  #wait
else
  # Execute for the specified SZZ_V
  method="$SZZ_V"
  for ((i=0; i<num_jobs; i++)); do
    # Calculate start and end for the job
    end=$((start + increment - 1))

    # Log file name inside the logs directory
    log_file="$logs_dir/${PROJECT_NAME}_${method}_output_${start}_${end}.log"

    # Run the job for the specified method in the background
    run_job $method $start $end $log_file

    # Update the start for the next job
    start=$((end + 1))
  done

  # Wait for all background jobs to complete
  #wait
fi
