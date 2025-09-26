#!/bin/bash
#SBATCH --job-name=note_embeddings
#SBATCH --output=logs/text_embedding_%j.log
#SBATCH --error=logs/text_embedding_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --partition=compute # -p
#SBATCH --time=24:00:00
#SBATCH -o logs/data_segmentation.out

source /etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate note_embeddings
export TQDM_DISABLE=1

# Run the Python script
python process_text_embeddings.py --data_type="medications"
python process_text_embeddings.py --data_type="clinical_notes" --dims=64
