#!/bin/bash

# Get current timestamp
timestamp=$(date +"%Y%m%d%H%M%S")

# Generate job name, output file, and error file based on timestamp
job_name="feature_miner_$timestamp"
output_file="feature_miner_$timestamp.log"
error_file="feature_miner_$timestamp.err"

#SBATCH --job-name=$job_name              # Job name
#SBATCH --output=$output_file             # Output file for stdout
#SBATCH --error=$error_file               # Output file for stderr
#SBATCH --nodes=1                        # Request 10 nodes
#SBATCH --time=INFINITE                   # Adjust the time limit as needed

# Command to run
/usr/bin/python /home/huayo708/projects/pyszz_andie/commit_analyzer/feature_miner.py -cdir /home/huayo708/projects/data/historical_commit_data -r '' -o /home/huayo708/projects/data/commit_minner_data