#!/bin/bash
#SBATCH --job-name=extract_ppg_features
#SBATCH --output=logs/extract_ppg_features_%j.out
#SBATCH --partition=compute # -p
#SBATCH --cpus-per-task=12 # -c
#SBATCH --mem=128gb
#SBATCH --time=2-00:00:00

# uncomment to receive email when job fails
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=firstname.lastname@student.hpi.de

# Initialize conda:
eval "$(conda shell.bash hook)"

# some command to activate a specific conda environment or whatever:
conda activate cass_pipeline_latest

python -m preprocessing.preprocess_wearable.feature_extraction.ppg_feature_extraction --kubios_fs $@
