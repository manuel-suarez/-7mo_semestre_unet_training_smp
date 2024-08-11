#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=CIMAT-ODS17-SMP-L-FPN+ResNet34
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1-30%1

srun /home/$(whoami)/tools/anaconda3/envs/py3.9-pt/bin/python main.py results logs fpn resnet34 10
