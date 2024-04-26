#!/bin/bash
#SBATCH --job-name=git_commit_miner_v2
#SBATCH --output=git_commit_miner_v2.log
#SBATCH --error=git_commit_miner_v2.err
#SBATCH --nodes=10              # Request 10 nodes

python /home/huayo708/projects/pyszz_andie/commit_analyzer/git_commit_miner/git_commit_miner.py -repo_dir /home/huayo708/projects/repo -output_dir /home/huayo708/projects/data/historical_commit_data