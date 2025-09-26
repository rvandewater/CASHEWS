#!/bin/bash
#SBATCH --job-name=extract_ppg_minirocket
#SBATCH --output=logs/extract_ppg_minirocket_%j.out
#SBATCH --partition=pgpu
#SBATCH --cpus-per-task=32 # -c
#SBATCH --gpus-per-task=16
#SBATCH --mem=512gb
#SBATCH --time=2-00:00:00

# uncomment to receive email when job fails
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=firstname.lastname@student.hpi.de

# Initialize conda:
eval "$(conda shell.bash hook)"

# some command to activate a specific conda environment or whatever:
conda activate cass_pipeline

python -m preprocessing.preprocess_wearable.embedding.extract_minirocket_features --clean_data_dir 2025-03-12_gap-60s_corsanov2-infrared-light --gpu $@ 
