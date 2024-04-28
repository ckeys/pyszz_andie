#!/bin/bash
#SBATCH --job-name=feature_miner_v3
#SBATCH --output=feature_miner_v3.log
#SBATCH --error=feature_miner_v3.err
#SBATCH --nodes=10              # Request 2 nodes
#SBATCH --time=INFINITE  # Adjust the time limit as needed


# Command to run
/usr/bin/python /home/huayo708/projects/pyszz_andie/commit_analyzer/feature_miner.py -cdir /home/huayo708/projects/data/historical_commit_data -r /home/huayo708/projects/repo -o /home/huayo708/projects/data/commit_minner_data
