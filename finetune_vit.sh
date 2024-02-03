#!/bin/bash
#SBATCH -c 1 
#SBATCH -t 24:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1,vram:48G
#SBATCH --mem=8G

module load gcc/9.2.0
module load cuda/11.7

python scripts/main_autosam_seg.py --epochs 120 -b 10 --tr_size 2211 --save_dir synapse10 --synapse_save_path ./output_test/synapse10 --model_type vit_b_original