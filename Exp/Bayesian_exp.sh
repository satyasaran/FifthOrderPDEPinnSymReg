#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1
#SBATCH --mem=30GB
#SBATCH --job-name=RadiMax_GPU_GB
#SBATCH --error=%J.err_
#SBATCH --output=%J.out_
#SBATCH -p gpu --gres=gpu:a100
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=5:20:00
hostname
echo $CUDA_VISIBLE_DEVICES
python3 Bayesian_exp.py
