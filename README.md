![CASHEWS Logo](assets/cashews_logo.svg)
# CASHEWS

This repository contains code for processing the CASSANDRA project's data which aims to create a real-time alarm system
for post-operative complications from clinical data, wearable data, and notes.

## Setup instructions

```bash
git clone https://git.bihealth.org/vandewrp/hpc-cassandra-analysis.git
cd hpc-cassandra-analysis
```

Use conda to install the environment we use:

`conda env create -f environment.yml`

Then use:

`conda activate cass_pipeline`

Or choose the environment as kernel in your notebook. You can install the kernel on the charite cluster using:

`python -m ipykernel install --user --name cass_pipeline `

## File Description

- `baselines`: Contains scripts and functions to extract baseline datasets.
- `data_exploration`: Contains notebooks and scripts for exploring the data.
- `deprecated`: Contains old or unused scripts and functions.
- `paper_statistics_visualization`: Contains scripts and notebooks for visualizing statistics for the paper.
- `preprocessing`: Contains scripts and functions to prepare the data for segmentation.
- `retrospective`: Contains scripts and functions for retrospective data analysis.
- `utils`: Contains utility functions used across the project.
- `config.yaml`: Configuration file for the data segmentation script.
- `debug_data_segmentation.ipynb`: Jupyter notebook for debugging the data segmentation process.
- `environment.yml`: Conda environment configuration file.
- `execute_data_segmentation.py`: Script to execute the data segmentation process.
- `execute_job_data_segmentation.sh`: Shell script to run the data segmentation job on HPC.

## Wearable data
The wearable data is stored in the shared HPC folder and is processed using the `preprocess_wearable` module. 
The data is cleaned, filtered, and segmented before being used for further analysis.

### PPG feature extraciton
To extract PPG features from the wearable data, run the following command from the root directory of the repository:
```bash
sbatch run_scripts/run_ppg_feature_extraction.slurm --clean_data_dir 2025-06-06_gap-60s_corsanov2-infrared-light -a --output_dir /sc-projects/sc-proj-cc08-cassandra/Prospective_Preprocessed/ppg_features/2025-06-17_minor-segment-5m_corsanov2-infrared-light_kubios-fs_aligned
```
### Minirocket
To extract MiniRocket features from the wearable data, run the following command from the root directory of the repository:
```bash
sbatch run_scripts/run_ppg_minirocket.slurm --clean_data_dir 2025-06-06_gap-60s_corsanov2-infrared-light -v
```

