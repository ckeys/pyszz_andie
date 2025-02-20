#!/bin/bash

# Check if MLSZZ_PATH is set
if [ -z "$MLSZZ_PATH" ]; then
  echo "MLSZZ_PATH environment variable is not set"
  exit 1
fi

if [ -z "$SZZ_V" ]; then
  echo "SZZ_V environment variable is not set"
  exit 1
fi

# Base command
base_command="nohup python $MLSZZ_PATH/main.py $MLSZZ_PATH/in/valid_project.json $MLSZZ_PATH/conf/$SZZ_V.yml None"

# Job parameters
start=0
increment=2000
method = $SZZ_V

# Create 10 jobs
for i in {0..2}; do
  end=$((start + (i + 1) * increment - 1))
  log_file="${method}_output_${start}_${end}.log"
  full_command="$base_command $start $end > $log_file 2>&1 &"
  echo "Running job $i: $full_command"
  eval $full_command
  start=$((end + 1))
done
