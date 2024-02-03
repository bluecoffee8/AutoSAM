#!/bin/bash
#SBATCH -c 1 
#SBATCH -t 17:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

module load gcc/9.2.0
module load cuda/11.7

python scripts/main_autosam_replicate.py