#!/bin/bash
#SBATCH -c 1 
#SBATCH -t 0:10:00
#SBATCH -p gpu
#SBATCH --gres=gpu:teslaM40:1 
#SBATCH --mem=16G

module load gcc/9.2.0
module load cuda/11.7

python helper_code.py