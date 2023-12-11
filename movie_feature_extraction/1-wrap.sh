#!/bin/bash
# v3.2

# activate venv
eval "$(conda shell.bash hook)" # this line is necessary to conda activate via bash script
conda activate topfeval

python3.9 1-pliers_feature_extraction.py $1 $2 $3 $4

# Deactivate environment 
conda deactivate
