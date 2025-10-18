#!/bin/bash
#SBATCH --job-name=bbs321e4v2ec
#SBATCH --chdir=/mnt/storage/franklhtan/projects/PhonGen2025
#SBATCH --output=/mnt/storage/franklhtan/projects/PhonGen2025/slurm_config/out/bbs321e4v2ec.out
#SBATCH --error=/mnt/storage/franklhtan/projects/PhonGen2025/slurm_config/out/bbs321e4v2ec.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

singularity exec --bind /mnt/storage/:/mnt/storage/ --nv --fakeroot /mnt/storage/franklhtan/ubuntu bash -c "cd /mnt/storage/franklhtan/projects/PhonGen2025/src;python3 eval_collect.py --config ./configs/b_BS32_LR1e4v2_FEAT_4_collect.py"