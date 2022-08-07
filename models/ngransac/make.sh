#!/bin/bash

#SBATCH -J rigidmask0
#SBATCH -p gpu
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zpetroff@iu.edu
#SBATCH --nodes=1
#SBATCH --gpus-per-node v100:1
#SBATCH --time=01:00:00

#Run your program
srun python setup.py install