#!/bin/bash

#SBATCH -J make
#SBATCH -p gpu
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node v100:1
#SBATCH --time=01:00:00

#Run your program
srun python setup.py install