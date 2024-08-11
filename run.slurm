#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=CIMAT-ODS17-SMP-L-Unet+ResNet34
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-30%1

srun /home/hasimoto/tools/anaconda3/envs/py3.9-pt/bin/python main.py results logs unet resnet34 10
