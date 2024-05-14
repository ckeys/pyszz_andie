#!/bin/bash
#SBATCH --job-name=mlszz
#SBATCH --output=mlszz.log
#SBATCH --error=mlszz.err


# Command to run
/usr/bin/python /home/huayo708/projects/pyszz_andie/main.py /home/huayo708/projects/pyszz_andie/in/bugfix_commits_all.json /home/huayo708/projects/pyszz_andie/conf/mlszz.yml /home/huayo708/projects/repo 5