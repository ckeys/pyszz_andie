#!/bin/bash
#SBATCH --job-name=feature_miner_v2
#SBATCH --output=feature_miner_v2.log
#SBATCH --error=feature_miner_v2.err
#SBATCH --ntasks-per-node=1   # Run 1 task per node
#SBATCH --nodes=10              # Request 2 nodes
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00   # Adjust the time limit as needed


# Command to run
/usr/bin/python /home/huayo708/projects/pyszz_andie/commit_analyzer/feature_miner.py -cdir /home/huayo708/projects/data/historical_commit_data -r /home/huayo708/projects/repo -o /home/huayo708/projects/data/commit_minner_data
