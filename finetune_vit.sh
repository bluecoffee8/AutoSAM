#!/bin/bash
#SBATCH -c 1 
#SBATCH -t 6:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1,vram:48G
#SBATCH --mem=48G

module load gcc/9.2.0
module load cuda/11.7

python scripts/main_autosam_seg.py