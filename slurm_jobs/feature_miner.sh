#!/bin/bash
#SBATCH --job-name=feature_miner
#SBATCH --output=feature_miner.log
#SBATCH --error=feature_miner.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00   # Adjust the time limit as needed


# Command to run
/usr/bin/python /home/huayo708/projects/pyszz_andie/commit_analyzer/feature_miner.py -cdir /home/huayo708/projects/data/historical_commit_data -r /home/huayo708/projects/repo -o /home/huayo708/projects/data/commit_miner_data
