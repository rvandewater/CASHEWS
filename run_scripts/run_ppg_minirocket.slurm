#!/bin/bash
#SBATCH --job-name=ppg_minirocket
#SBATCH --output=logs/extract_ppg_minirocket_%j.out
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16 #32 # -c
#SBATCH --mem=256gb #512gb
#SBATCH --time=1-00:00:00

# uncomment to receive email when job fails
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=firstname.lastname@student.hpi.de

# Initialize conda:
eval "$(conda shell.bash hook)"

# some command to activate a specific conda environment or whatever:
conda activate cass_pipeline_embeddings
# Get the current date in YYYY-MM-DD format
current_date=$(date +%Y-%m-%d)
# Define the base directory
base_dir="/sc-projects/sc-proj-cc08-cassandra/Prospective_Preprocessed/ppg_embeddings"

# Create the output directory using the current date
output_dir="${base_dir}/${current_date}_MINIROCKET"

python -m preprocessing.preprocess_wearable.embedding.extract_minirocket_features --output_dir=$output_dir $@
