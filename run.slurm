#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=UNet-SMP-L-2GPU-10EPOCHS
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1

/home/hasimoto/tools/anaconda3/envs/py3.9-pt/bin/python main.py results 10
