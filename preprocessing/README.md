# Preprocessing Package

The preprocessing package contains the scripts and functions to prepare the data for segmentation. The data is loaded
from the shared HPC folder and preprocessed before segmentation. The preprocessing steps include data cleaning, data
filtering, data embedding, and data segmentation. The data is then saved in the shared HPC folder for segmentation.

## Commands
Activate environment:
```bash
conda activate cass_pipeline
```
### Extract patient summaries
To extract patient summaries from the wearable data, run the following command from the root directory of the repository:
```bash
python -m preprocessing.preprocess_wearable.extract_patient_summaries 
```

### Clean PPG
To clean the PPG data, run the following command from the root directory of the repository:
```bash
sbatch run_scripts/run_ppg_cleaning.slurm
## Description

- `data_checks`: check the data for missing values and other issues.
- `data_utils`: utility functions for data processing.
- `dataloader`: helper functions/package to load the data from the shared HPC folder.
- `embeddings_clinical_notes`: extract embeddings from the clinical notes.
- `embeddings_medications`: extract embeddings from the medication data.
- `filters`: helper functions to filter the data.
- `post_processing`: post-processing steps after data segmentation.
- `preprocess_pdms`: COPRA6, ISHMED and RedCAP data preprocessing after loading from HDL.
- `preprocess_wearable`: extract embeddings from the wearable data.
- `prepare_segmentation.py`: script to prepare data for segmentation.
- `segment.py`: main script to segment the data.
