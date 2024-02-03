#!/bin/bash
#SBATCH -c 1 
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1,vram:24G
#SBATCH --mem=4G

module load gcc/9.2.0
module load cuda/11.7

python scripts/main_novel_autosam.py