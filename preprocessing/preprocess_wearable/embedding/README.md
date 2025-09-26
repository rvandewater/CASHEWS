# Generate Feature Embeddings

## MINIROCKET

run `extract_minirocket_features.py` specifying the considered segment lenghts to encode and the corresponding number of features.
Embdding features are calculated using the CPU-based [MINIROCKET implementation](https://github.com/angus924/minirocket) (as of 26 March 2025).
Technically, GPU functionality is implemented using the [tsai implementation](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb) leveraging PyTorch for convolutions. However, the extracted features are currently all-zeros due to unknown reasons.

## Time Series Models

1. Train model using `train_embedding_model.py`
    1. Potential model types are [TCN-AE](https://github.com/MarkusThill/bioma-tcn-ae) (preferred) (adapted from TF to PyTorch, using [PyTorch-TCN](https://github.com/paul-krug/pytorch-tcn)) and [LAAE](https://arxiv.org/pdf/2201.09172) with their respoective configs being stored in `./config`
    2. Model checkpoints are stored in `./models/{model_type}/{YYYY-MM-DD}`
    3. Hyperparameter optimisation enabled using WandB sweeps
2. Extract embeddings using `extract_embedding_features.py` referencing model checkpoint
    1. Uses non-overlapping windows of length `cfg.data.window_size` within segments of length `--minor_segment_m` to create a batch and extract embeddings for the minor segment.
    2. All concatentated embeddings for the minor segment are averaged within larger major segments of length `--major_segment_m`

Best Model is shown on [WandB](https://wandb.ai/cassandra_hpi/ppg_embedding/runs/8um1gzmt/workspace?nw=nwuserbjarnepfitzner) and stored at `/sc-projects/sc-proj-cc08-cassandra/Prospective_Preprocessed/ppg_embeddings/models/TCNAE/2025-02-17/model_5.pth`
