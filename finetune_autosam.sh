#!/bin/bash
#SBATCH -c 1 
#SBATCH -t 4:00:00
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1,vram:75G
#SBATCH --mem=8G

module load gcc/9.2.0
module load cuda/11.7

python scripts/main_autosam_seg2.py --epochs 120 -b 10 --tr_size 200 --save_dir synapse19 --synapse_save_path ./output_test/synapse19 --model_type vit_b_original