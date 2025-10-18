#!/bin/bash
#SBATCH --job-name=Cabs321e4v2EU
#SBATCH --chdir=/mnt/storage/franklhtan/projects/PhonGen2025
#SBATCH --output=/mnt/storage/franklhtan/projects/PhonGen2025/slurm_config/out/Cabs321e4v2EU.out
#SBATCH --error=/mnt/storage/franklhtan/projects/PhonGen2025/slurm_config/out/Cabs321e4v2EU.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

singularity exec --bind /mnt/storage/:/mnt/storage/ --nv --fakeroot /mnt/storage/franklhtan/ubuntu bash -c "cd /mnt/storage/franklhtan/projects/PhonGen2025/src;python3 eval_collect_twophase.py --config ./configs/C_a_BS32_LR1e4v2_FEAT_4_EU.py"