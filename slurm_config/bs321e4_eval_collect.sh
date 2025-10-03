#!/bin/bash
#SBATCH --job-name=bs321e4ec
#SBATCH --output=/mnt/storage/franklhtan/projects/PhonGen2025/slurm_config/out/bs321e4ec.out
#SBATCH --error=/mnt/storage/franklhtan/projects/PhonGen2025/slurm_config/out/bs321e4ec.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

singularity exec --fakeroot --nv --writable --bind /mnt/storage/:/mnt/storage/ /mnt/storage/franklhtan/ubuntu bash -c "cd /mnt/storage/franklhtan/projects/PhonGen2025/src;python3 eval_collect.py --config ./configs/BS32_LR1e4_eval_collect.py"