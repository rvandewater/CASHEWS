# file extract_minirocket

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from functools import partial
import numpy as np
import polars as pl
from omegaconf import OmegaConf
from torch import multiprocessing as mp

from preprocessing.preprocess_wearable.embedding.minirocket_implementation import (
    fit,
    transform,
)
from preprocessing.preprocess_wearable.ppg_paths import (
    CLEANED_PPG_BASE_PATH,
    PPG_EMBEDDINGS_BASE_PATH,
    CONFIG_PATH
)
from preprocessing.preprocess_wearable.wearable_utils import (
    clip_ppg_signal_to_stay_duration,
    get_stay_aligned_segments,
    load_cleaned_ppg_for_patient,
    resample_df,
)

# Import for GPU

try:
    import torch
    from tsai.models.MINIROCKET_Pytorch import MiniRocketFeatures

    tsai_available = True
    print("TSAI MiniRocketFeatures is available.")
except ImportError:
    tsai_available = False
    print("TSAI MiniRocketFeatures is not available (will use CPU instead).")


def main(args):
    """
    Main function to extract MiniRocket features from PPG signals.

    Parameters:
    args (argparse.Namespace): The command-line arguments.
    """

    if args.output_dir == "":
        actual_n_features = [int(num_features / 84) * 84 for num_features in args.num_features]
        n_features_str = (
            "-".join([str(n) for n in actual_n_features])
            if isinstance(actual_n_features, list)
            else str(actual_n_features)
        )
        gpu_str = "_GPU" if args.gpu and tsai_available else ""
        segment_lengths_str = (
            "h-".join([str(s) for s in args.segment_length])
            if isinstance(args.segment_length, list)
            else str(args.segment_length)
        )
        out_dir = (
            f"{args.output_base_dir}/{datetime.now().strftime('%Y-%m-%d')}_MINIROCKET{gpu_str}"
            + f"_{segment_lengths_str}h"
            + f"_{n_features_str}-features"
            + f"_corsanov2-{args.corsano_v2_light}-light"
        )
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/clean_data_origin.txt", "a") as f:
        f.write(f"Clean PPG data loaded from {os.path.join(CLEANED_PPG_BASE_PATH, args.clean_data_dir)}\n")

    n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 16
    if args.gpu and tsai_available:
        queue = mp.Queue()
        if args.patient_id is not None:
            queue.put(0)
            init_worker(queue)
            process_single_patient(args, out_dir, args.patient_id)
        else:
            n_gpus = torch.cuda.device_count()
            for gpu_ids in range(n_gpus):
                queue.put(gpu_ids)
            with mp.Pool(processes=n_cpus, initializer=init_worker, initargs=(queue,)) as p:
                p.map(
                    partial(process_single_patient, args, out_dir),
                    os.listdir(os.path.join(CLEANED_PPG_BASE_PATH, args.clean_data_dir)),
                )

    else:
        if args.patient_id is not None:
            process_single_patient(args, out_dir, args.patient_id)
        else:
            with mp.Pool(processes=n_cpus) as p:
                p.map(
                    partial(process_single_patient, args, out_dir),
                    os.listdir(os.path.join(CLEANED_PPG_BASE_PATH, args.clean_data_dir)),
                )


def init_worker(queue):
    global gpu_queue
    gpu_queue = queue


def process_single_patient(args, out_dir, pat_id):
    try:
        if args.gpu and tsai_available:
            gpu_id = gpu_queue.get()
            gpu_device = torch.device(f"cuda:{gpu_id}")
        if args.verbose:
            print(f"Processing patient {pat_id}...")
            sys.stdout.flush()

        if "parquet" in pat_id:
            pat_id = pat_id.split(".")[0]

        out_file_name = f"{out_dir}/{pat_id}.parquet"

        if os.path.exists(out_file_name) and not args.overwrite:
            print(f"Output file for patient {pat_id} already exists, pass --overwrite to overwrite.")
            return

        start_time = time.time()

        # Load PPG signal
        ppg_signal, sampling_freq = load_cleaned_ppg_for_patient(
            pat_id, os.path.join(CLEANED_PPG_BASE_PATH, args.clean_data_dir), args.verbose
        )
        if sampling_freq < 32:
            if args.verbose:
                print(f"Resampling PPG signal from {sampling_freq}Hz to 32Hz")
            ppg_signal = resample_df(ppg_signal, ["ppg"], 32)

        # Clip signal to stay between surgery start and end time
        ppg_signal, stay_start, stay_end = clip_ppg_signal_to_stay_duration(
            ppg_signal,
            pat_id,
            OmegaConf.load(CONFIG_PATH),
            args.verbose,
        )
        if len(ppg_signal) == 0:
            print(f"No data found for patient {pat_id} after filtering")
            return

        # Ensure segment_length is a list
        segment_lengths = (
            args.segment_length if isinstance(args.segment_length, list) else [args.segment_length]
        )

        # Convert num_features to a dictionary
        num_features = args.num_features if isinstance(args.num_features, list) else [args.num_features]
        num_features = {
            segment_length: n_features for segment_length, n_features in zip(segment_lengths, num_features)
        }

        segment_hours = []

        for segment_length in segment_lengths:
            segment_hours.append(float(segment_length))

        segment_hours.sort()
        min_segment_hours = segment_hours[0]

        if args.verbose:
            print(f"Using segment lengths (hours): {segment_hours}")
            print(f"Using minimum segment length for iteration: {min_segment_hours}h")

        minirocket_models = {}

        segments = get_stay_aligned_segments(ppg_signal, min_segment_hours, stay_start, args.verbose)
        max_segment_samples = max([len(segment) for segment in segments])
        minirocket_input_samples = {
            segment_length: max_segment_samples * int(segment_length / min_segment_hours)
            for segment_length in segment_hours
        }
        print(f"Minirocket input samples: {minirocket_input_samples}")

        if args.verbose:
            print(f"Found {len(segments)} segments of {min_segment_hours}h")

        if len(segments) == 0:
            print(f"No segments found for patient {pat_id}")
            return

        all_results = []

        # ppg_segments = [segment['ppg'].to_numpy() for segment in segments]
        # ppg_segments = np.stack(ppg_segments, axis=0)
        # print(ppg_segments.shape)
        # exit()

        for i, segment in enumerate(segments):
            if args.verbose and i % 10 == 0:
                print(f"Processing segment {i+1}/{len(segments)}")

            # Get timestamp for this segment
            segment_timestamp = segment["datetime"][0]

            segment_features = {
                "datetime": [segment_timestamp],
            }

            for segment_length in segment_hours:
                if segment_length == min_segment_hours:
                    current_data = segment["ppg"].to_numpy()
                else:
                    end_time = segment["datetime"].max()
                    start_time = end_time - timedelta(hours=segment_length)

                    long_segment = ppg_signal.filter(
                        (pl.col("datetime") > start_time) & (pl.col("datetime") <= end_time)
                    )
                    current_data = long_segment["ppg"].to_numpy()

                X = np.full((1, minirocket_input_samples[segment_length]), 0, dtype=np.float32)
                X[0, -len(current_data) :] = current_data

                # Create MiniRocket model if not already created
                if segment_length not in minirocket_models:
                    if args.gpu and tsai_available:
                        # GPU implementation
                        X_reshaped = np.expand_dims(X, axis=1)
                        X_tensor = torch.tensor(X_reshaped).to(gpu_device)

                        rocket = MiniRocketFeatures(1, X.shape[1], num_features=num_features[segment_length])
                        rocket = rocket.to(gpu_device)
                        rocket.fit(X_tensor)

                        minirocket_models[segment_length] = ("gpu", rocket)
                    else:
                        # CPU implementation
                        parameters = fit(X, num_features=num_features[segment_length])
                        minirocket_models[segment_length] = ("cpu", parameters)

                    if args.verbose and i == 0:
                        print(f"Trained MiniRocket model for {segment_length}h segments")

                model_type, model = minirocket_models[segment_length]

                if model_type == "gpu":
                    X_reshaped = np.expand_dims(X, axis=1)
                    X_tensor = torch.tensor(X_reshaped).to(gpu_device)
                    features = model(X_tensor)
                    features = features.cpu().numpy()
                else:
                    features = transform(X, model)

                for feature_idx in range(features.shape[1]):
                    segment_features[f"emb_{feature_idx}_{int(segment_length)}h"] = [
                        float(features[0, feature_idx])
                    ]

            all_results.append(pl.DataFrame(segment_features))

        if all_results:
            result_df = pl.concat(all_results, how="diagonal").sort("datetime")

            result_df.write_parquet(out_file_name)

            if args.verbose:
                print(
                    f"Saved result with {len(result_df)} rows and {len(result_df.columns)} columns to {out_file_name}"
                )
        else:
            print(f"No valid segments for patient {pat_id}")

        if isinstance(start_time, float):
            print(f"Done processing patient {pat_id} in {(time.time() - start_time) / 60:.2f} minutes")
        else:
            print(f"Done processing patient {pat_id}")

    except Exception as e:
        print(f"Error processing patient {pat_id}: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if args.gpu and tsai_available:
            gpu_queue.put(gpu_id)
        sys.stdout.flush()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_data_dir",
        help='Path to the directory containing the clean PPG data or subfolder in "/sc-projects/sc-proj-cc08-cassandra/Prospective_Preprocessed/ppg_features"',
    )
    parser.add_argument("--output_base_dir", default=PPG_EMBEDDINGS_BASE_PATH)
    parser.add_argument("--output_dir", default="", help="Output directory path.")
    parser.add_argument(
        "--patient_id", "-p", type=str, default=None, help="If provided, only process patient with this ID"
    )
    parser.add_argument(
        "--segment_length",
        nargs="+",
        default=[1, 6, 12],
        help="Length of segments in hours (numeric values only, e.g., 1, 6, 12)",
    )
    parser.add_argument(
        "--num_features",
        nargs="+",
        default=[504, 336, 168],
        help="Number of MiniRocket features (minimum 84, will be rounded down to multiple of 84)",
    )
    parser.add_argument(
        "--corsano_v2_light", choices=["green", "red", "infrared"], help="The PPG light for Corsano v2"
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing files")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Print verbose output")
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="CURRENTLY BROKEN, GENERATES ALL-ZEROS FEATURES. Use GPU implementation from TSAI if available",
    )

    args = parser.parse_args()
    print(args)

    if args.corsano_v2_light is None:
        for substr in args.clean_data_dir.split("_"):
            if "corsanov2" in substr:
                args.corsano_v2_light = substr.split("-")[1]
                break
        if args.corsano_v2_light is None:
            args.corsano_v2_light = "unknown-light"

    # add clean data base path to clean data dir
    if args.clean_data_dir.find("/") == -1:
        args.clean_data_dir = f"{CLEANED_PPG_BASE_PATH}/{args.clean_data_dir}"

    main(args)
