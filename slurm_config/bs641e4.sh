#!/bin/bash
#SBATCH --job-name=bs641e4
#SBATCH --output=/mnt/storage/qisheng/github/PhonGen2025/slurm_config/out/bs641e4.out
#SBATCH --error=/mnt/storage/qisheng/github/PhonGen2025/slurm_config/out/bs641e4.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/PhonGen2025/src;python3 train.py --config ./configs/BS64_LR1e4.py"
