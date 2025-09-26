#!/bin/bash
#SBATCH --job-name=preprocessing_pdms
#SBATCH --partition=compute # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=50gb
#SBATCH --output=logs/preprocessing%a_%j.log # %j is job id
#SBATCH --time=1:00:00

# This script preprocesses PDMS data
source /etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate cass_pipeline

export DATA_FORMAT="it" # Set the data format to "it"

# Run the preprocessing scripts
python ISHMED_preprocessing.py  --db-format $DATA_FORMAT
python COPRA_6_preprocessing.py --db-format $DATA_FORMAT
python RedCAP_preprocessing.py --db-format $DATA_FORMAT
