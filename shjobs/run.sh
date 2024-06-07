#!/bin/bash

# Check if MLSZZ_PATH is set
if [ -z "$MLSZZ_PATH" ]; then
  echo "MLSZZ_PATH environment variable is not set"
  exit 1
fi

# Base command
base_command="nohup python $MLSZZ_PATH/main.py $MLSZZ_PATH/in/valid_project.json $MLSZZ_PATH/conf/mlszz.yml None"

# Job parameters
start=0
increment=50

# Create 10 jobs
for i in {0..9}; do
  end=$((start + increment))
  log_file="output_${i}.log"
  full_command="$base_command $start $end > $log_file 2>&1 &"
  echo "Running job $i: $full_command"
  eval $full_command
  start=$end
done