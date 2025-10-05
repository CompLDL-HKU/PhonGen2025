#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --chdir=/mnt/storage/franklhtan/projects/PhonGen2025
#SBATCH --output=/mnt/storage/franklhtan/projects/PhonGen2025/gpu_check.out
#SBATCH --error=/mnt/storage/franklhtan/projects/PhonGen2025/gpu_check.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

singularity exec -B /mnt/storage/franklhtan/:/mnt/storage/franklhtan --nv --fakeroot /mnt/storage/franklhtan/ubuntu/ bash -c "cd /mnt/storage/franklhtan;python3 /mnt/storage/franklhtan/projects/PhonGen2025/src/test_mnist.py"