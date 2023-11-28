#!/bin/bash
#SBATCH -c 1 
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:teslaM40:1 
#SBATCH --mem=24G

module load gcc/9.2.0
module load cuda/11.7

python scripts/main_feat_seg.py 