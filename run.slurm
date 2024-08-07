#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=SMPT-Torch-UNet+ResNet34-OSDC
#SBATCH --time=0
#SBATCH --mem=0
#SBATCH --array=1

/home/est_posgrado_manuel.suarez/tools/anaconda3/envs/py3.9-pt/bin/python main_torch.py results 10
