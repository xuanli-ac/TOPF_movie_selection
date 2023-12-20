#!/bin/bash
# v3.2

# activate venv
eval "$(conda shell.bash hook)" # this line is necessary to conda activate via bash script
conda activate topfeval

python3.9 4-cal_feature_importance.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}

# Deactivate environment 
conda deactivate
