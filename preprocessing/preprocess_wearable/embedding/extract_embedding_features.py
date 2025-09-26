import argparse
import os
import sys
import time
from datetime import datetime
from functools import partial

import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import multiprocessing as mp

import preprocessing.text_embeddings.clinical_notes_utils
from preprocessing.preprocess_wearable.embedding.CassandraDataset import (
    get_windows,
    split_df_by_time_gaps,
)
from preprocessing.preprocess_wearable.ppg_paths import (
    CLEANED_PPG_BASE_PATH,
    PPG_EMBEDDINGS_BASE_PATH,
)
from preprocessing.preprocess_wearable.wearable_utils import (
    add_time_diff_col,
    get_segments,
    load_cleaned_ppg_for_patient,
    resample_df,
)

SAMPLING_FREQ = 32


def main(args):
    """
    Main function to extract embeddings from PPG signals for a given patient ID.

    Parameters:
    args (argparse.Namespace): The command-line arguments.
    """
    # Get model
    cfg = get_config(os.path.dirname(args.model_path))
    if args.clean_data_dir is None:
        args.clean_data_dir = cfg.ppg_folder

    if args.output_dir == "":
        out_dir = (
            f"{args.output_base_dir}/"
            + datetime.now().strftime("%Y-%m-%d")
            + f"_{cfg.model._target_.split('.')[-1]}_segments-minor-{args.minor_segment_m}m-major-{args.major_segment_m}m"
        )
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/clean_data_origin.txt", "a") as f:
        f.write(f"Clean PPG data loaded from {os.path.join(CLEANED_PPG_BASE_PATH, args.clean_data_dir)}\n")

    queue = mp.Queue()
    if args.patient_id is not None:
        queue.put(0)
        init_worker(queue)
        # Somehow cannot pass model anymore, so loading it for each patient
        # process_single_patient(args, cfg, out_dir, model, args.patient_id)
        process_single_patient(args, cfg, out_dir, args.patient_id)
    else:
        n_gpus = torch.cuda.device_count()
        for gpu_ids in range(n_gpus):
            queue.put(gpu_ids)
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 16
        with mp.Pool(processes=n_cpus, initializer=init_worker, initargs=(queue,)) as p:
            p.map(
                partial(process_single_patient, args, cfg, out_dir),
                os.listdir(os.path.join(CLEANED_PPG_BASE_PATH, args.clean_data_dir)),
            )


def init_worker(queue):
    global gpu_queue
    gpu_queue = queue


def process_single_patient(args, cfg, out_dir, pat_id):
    try:
        # request GPU resources
        gpu_id = gpu_queue.get()
        device = torch.device(f"cuda:{gpu_id}")
        if args.verbose:
            print(f"Processing patient {pat_id}...")
            sys.stdout.flush()

        if "parquet" in pat_id:
            pat_id = pat_id.split(".")[0]
        out_file_name = f"{out_dir}/{pat_id}.parquet"
        if os.path.exists(out_file_name) and not args.overwrite:
            print(f"Output file for patient {pat_id} already exists, pass --overwrite to overwrite.")
            return
        else:
            print(f"Writing to {out_file_name}")

        start_time = time.time()
        ppg_signal, sampling_freq = load_cleaned_ppg_for_patient(
            pat_id, os.path.join(CLEANED_PPG_BASE_PATH, args.clean_data_dir), args.verbose
        )

        # Upsample to SAMPLING_FREQ if necessary
        if sampling_freq != SAMPLING_FREQ:
            ppg_signal = add_time_diff_col(ppg_signal)
            split_dfs = split_df_by_time_gaps(
                ppg_signal, cfg.data.window_size // sampling_freq, verbose=False
            )

            resampled_split_dfs = []
            for split_df in split_dfs:
                resampled_split_dfs.append(resample_df(split_df))
            ppg_signal = pl.concat(resampled_split_dfs)

        ppg_mean = ppg_signal["ppg"].mean()
        ppg_std = ppg_signal["ppg"].std()

        major_segment_list = get_segments(ppg_signal, f"{args.major_segment_m}m", args.verbose)

        model = load_model(os.path.join(args.model_path), cfg.model)
        feature_generator = model.encoder
        feature_generator.to(device)
        feature_generator.eval()

        major_embeddings = []

        for major_segment in major_segment_list:
            if major_segment.is_empty():
                continue

            major_embedding = [
                major_segment.select(
                    pl.col("segment_group").first().dt.cast_time_unit("ns").alias("datetime")
                )
            ]
            segment_list = get_segments(major_segment, f"{args.minor_segment_m}m", verbose=False)
            samples_per_segment = args.minor_segment_m * 60 * SAMPLING_FREQ

            # Build one batch per segment
            batches = []
            for segment in segment_list:
                # todo what about segments shorter than 5 mins?
                # todo fill up segment with nans
                # only use full segments
                if segment.is_empty() or segment.shape[0] < samples_per_segment:
                    continue

                segment_windows, n_windows, _ = get_windows(
                    segment.select((pl.col("ppg") - ppg_mean) / ppg_std).to_numpy(),
                    cfg.data.window_size,
                    stride=cfg.data.window_size,
                    skip_zero_stddev=False,
                )
                # segment_windows length: cfg.data.window_size
                if len(segment_windows) > 0:
                    batches.append(torch.from_numpy(np.stack(segment_windows, axis=0)).float())

            # batches is list of (args.major_segment_m/args.minor_segment_m) segments of shape [cfg.data.window_size, 1]
            embeddings = []
            completeness = []

            for batch in batches:
                completeness.append(1 - np.mean(torch.isnan(batch).numpy()))
                batch = batch.to(device)
                preds = preprocessing.embeddings_clinical_notes.clinical_notes_utils.get_embedding(batch)
                # preds is tensor of shape [batchsize, embedding_size]
                preds = preds.detach().cpu()
                embeddings.append(preds.view(1, -1))

            if len(embeddings) > 0:
                embeddings = torch.mean(torch.stack(embeddings), dim=0).numpy()
                major_embedding.append(
                    pl.DataFrame(data=embeddings, schema=[f"emb_{i}" for i in range(embeddings.shape[1])])
                )
                total_completeness = np.mean(completeness)
                major_embedding.append(
                    pl.DataFrame(data=[total_completeness], schema=["signal_completeness"])
                )
                major_embeddings.append(pl.concat(major_embedding, how="horizontal"))

        if len(major_embeddings) > 0:
            all_features = pl.concat(major_embeddings, how="diagonal").sort("datetime")
            all_features.write_parquet(out_file_name)

            if args.verbose:
                print(f"Done processing patient {pat_id} in {(time.time() - start_time) / 60:.2f} minutes")
                sys.stdout.flush()
        else:
            if args.verbose:
                print(f"No embeddings generated for patient {pat_id}")
                sys.stdout.flush()
    except Exception as e:
        print(f"Error processing patient {pat_id}:", sys.exc_info()[0])
        print(e)
        sys.stdout.flush()
    finally:
        gpu_queue.put(gpu_id)


def load_model(model_path, model_config):
    model = instantiate(model_config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    print("Model loaded from:", model_path)

    return model


def get_config(model_path):
    model_args = OmegaConf.load(os.path.join(model_path, "config.yaml"))
    print("Loaded config:")
    print(OmegaConf.to_container(model_args, resolve=True))
    return model_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean_data_dir",
        help='Path to the directory containing the clean PPG data or subfolder in "/sc-projects/sc-proj-cc08-cassandra/Prospective_Preprocessed/ppg_features"',
    )
    parser.add_argument("--output_base_dir", default=PPG_EMBEDDINGS_BASE_PATH)
    parser.add_argument(
        "--output_dir",
        default="",
        help="Precise path to the output directory. Otherwise the directory name will be inferred.",
    )
    parser.add_argument(
        "--patient_id", "-p", type=str, default=None, help="If provided, only process patient with this ID"
    )
    parser.add_argument(
        "--minor_segment_m", type=int, default=1, help="Minor segments to calculate embeddings from"
    )
    parser.add_argument(
        "--major_segment_m", type=int, default=5, help="Major segments to aggregate embeddings in"
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing files")
    parser.add_argument("--model_path", type=str, help="Path to load the model and model config from.")

    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Print verbose output")

    args = parser.parse_args()
    print(args)

    # add clean data base path to clean data dir
    if args.clean_data_dir.find("/") == -1:
        args.clean_data_dir = f"{CLEANED_PPG_BASE_PATH}/{args.clean_data_dir}"

    mp.set_start_method("spawn")
    main(args)
