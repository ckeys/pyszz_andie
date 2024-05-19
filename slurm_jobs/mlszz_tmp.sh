#!/bin/bash
#SBATCH --job-name=mlszz_p3
#SBATCH --output=mlszz_p3.log
#SBATCH --error=mlszz_p3.err


# Command to run
/usr/bin/python /home/huayo708/projects/pyszz_andie/main.py /home/huayo708/projects/pyszz_andie/in/bugfix_commits_all.json /home/huayo708/projects/pyszz_andie/conf/mlszz.yml /home/huayo708/projects/repo 78