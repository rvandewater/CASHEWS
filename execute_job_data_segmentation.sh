#!/bin/bash
#SBATCH --job-name=data_segmentation
#SBATCH --output=logs/data_segmentation_%j.log
#SBATCH --error=logs/data_segmentation_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --partition=compute # -p
#SBATCH --time=18:00:00
#SBATCH -o logs/data_segmentation.out

source /etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate cass_pipeline_latest
export TQDM_DISABLE=1

# Run the Python script
 python execute_data_segmentation.py \
    --cohort="ICU" \
    --cohort_name="real_life_set" \
    --complication_cohort="real_life_set" \
    --complication_types="SSI-3,POPF,Galleleck/BDA" \
    --segment_length_hours=0.5
# python execute_data_segmentation.py \
#    --cohort="ICU_and_normal_ward" \
#    --cohort_name="lab_set" \
#    --complication_cohort="lab_set" \
#    --complication_types="SSI-3,POPF,Galleleck/BDA" \
#    --segment_length_hours=0.125

#python execute_data_segmentation.py \
#    --cohort="ICU" \
#    --cohort_name="real_life_set" \
#    --complication_cohort="real_life_set" \
#    --complication_types="SSI-3,POPF,Galleleck/BDA"\
#--cohort="ICU_and_normal_ward"#--cohort="ICU_and_normal_ward"
#--cohort="normal_ward"