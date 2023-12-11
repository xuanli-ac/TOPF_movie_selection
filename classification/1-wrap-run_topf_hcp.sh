#!/bin/bash
# v3.2

# activate venv
eval "$(conda shell.bash hook)" # this line is necessary to conda activate via bash script
conda activate topfeval

python3.9 1-run_topf_hcp.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19}

# Deactivate environment 
conda deactivate
