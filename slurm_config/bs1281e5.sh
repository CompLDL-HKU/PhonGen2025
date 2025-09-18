#!/bin/bash
#SBATCH --job-name=bs1281e5
#SBATCH --output=/mnt/storage/qisheng/github/PhonGen2025/slurm_config/out/bs1281e5.out
#SBATCH --error=/mnt/storage/qisheng/github/PhonGen2025/slurm_config/out/bs1281e5.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/qisheng/cuda12.8_sandbox bash -c "cd /mnt/storage/qisheng/github/PhonGen2025/src;python3 train.py --config ./configs/BS128_LR1e5.py"
