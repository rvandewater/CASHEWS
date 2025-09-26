import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import neurokit2 as nk
import numpy as np
import polars as pl
import scipy
from omegaconf import OmegaConf

from preprocessing.preprocess_wearable.feature_extraction.ppg_extract_neurokit_features import (
    get_all_hrv_features,
)
# from preprocessing.preprocess_wearable.feature_extraction.ppg_extract_pyppg_features import (
#     get_all_pyppg_features,
# )
from preprocessing.preprocess_wearable.ppg_paths import (
    CLEANED_PPG_BASE_PATH,
    PPG_FEATURES_BASE_PATH,
    CONFIG_PATH,
)
from preprocessing.preprocess_wearable.wearable_utils import (
    clip_ppg_signal_to_stay_duration,
    get_segments,
    get_stay_aligned_segments,
    insert_missing_value_dts,
    load_cleaned_ppg_for_patient,
)


def get_peaks_quality(peak_indices):
    intervals = np.diff(peak_indices)
    # Maximum distance from interval to their median
    max_dist = np.max(np.abs(intervals - np.median(intervals)))
    # distance relative to mean interval
    peak_quality = max_dist / np.median(intervals)

    return peak_quality


def _synthetically_interpolate_peaks(peak_indices, median_multiplier=1.5, verbose=False):
    if verbose:
        print("Attempting synthetic interpolation of peaks.")

    peak_distances = np.diff(peak_indices)
    median_distance = int(np.median(peak_distances))
    interpolate_peaks_at = np.argwhere(peak_distances > median_multiplier * median_distance).squeeze(-1)

    if len(interpolate_peaks_at) == 0:
        if verbose:
            print("No gaps to interpolate. Exiting without futher feature calculation.")
            return None

    added_vals = 0
    for index in interpolate_peaks_at:
        dist = peak_distances[index]
        multiplier = int(np.ceil(dist / median_distance))
        # insert indices in between
        peak_indices = np.insert(
            peak_indices,
            index + 1 + added_vals,
            np.linspace(
                peak_indices[index + added_vals], peak_indices[index + 1 + added_vals], multiplier, dtype=int
            )[1:-1],
        )
        added_vals += multiplier - 2

    if verbose:
        print(f"Added {added_vals} interpolated peaks")

    return peak_indices


def get_peaks(df, sampling_freq, method="bishop", use_fft=False, verbose=False):
    ppg_signal = df.select(pl.col("ppg")).to_numpy().squeeze()
    nan_mask = np.isnan(ppg_signal)
    try:
        if use_fft and nan_mask.sum() != 0:
            if verbose:
                print("Interpolating missing data with FFT...")
            # using FFT to interpolate missing values
            ppg_signal[nan_mask] = 0  # Replace NaN with 0 for FFT

            # Perform FFT
            fft_signal = scipy.fft.fft(ppg_signal)
            fft_filtered = np.zeros_like(fft_signal)
            sorted_indices = np.argsort(-np.abs(fft_signal))  # Sort by magnitude

            good_signal = False
            num_frequencies = 5

            # Retain only the top `num_frequencies` coefficients (low-pass filtering)
            # Gradually increase retained frequencies
            while not good_signal:
                fft_filtered[sorted_indices[:num_frequencies]] = fft_signal[sorted_indices[:num_frequencies]]

                # Reconstruct signal (optional: filter high frequencies)
                reconstructed_signal = scipy.fft.ifft(fft_filtered).real
                ppg_signal[nan_mask] = reconstructed_signal[nan_mask]

                peaks_signal, info = nk.ppg_peaks(
                    ppg_signal, sampling_freq, correct_artifacts=True, method=method
                )
                ppg_peaks = info["PPG_Peaks"]
                if get_peaks_quality(ppg_peaks) < 1:
                    if verbose:
                        print(f"Found good signal quality with {num_frequencies} FFT frequencies.")
                    return ppg_peaks
                else:
                    num_frequencies += 5
                    if num_frequencies >= 30:
                        if verbose:
                            print("PPG signal without missing data has too poor signal quality.")
                        return _synthetically_interpolate_peaks(ppg_peaks, verbose)
        else:
            # forward fill nan values
            ppg_signal = df.select(pl.col("ppg").forward_fill()).to_numpy().squeeze()
            peaks_signal, info = nk.ppg_peaks(
                ppg_signal, sampling_freq, correct_artifacts=True, method=method
            )
            ppg_peaks = info["PPG_Peaks"]
            if get_peaks_quality(ppg_peaks) > 1:
                return _synthetically_interpolate_peaks(ppg_peaks, verbose)
    except Exception as e:
        if verbose:
            print(f"Error identifying peaks: {str(e)}")
        return None


def process_single_patient(args, out_dir, pat_id):
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
    ppg_signal, sampling_freq = load_cleaned_ppg_for_patient(pat_id, args.clean_data_dir, args.verbose)

    if ppg_signal.select(pl.col("ppg").is_null().sum()).item() == 0:
        # If clean signal was (likely) not saved with "full" datetime column to properly extract peaks, add it
        ppg_signal = insert_missing_value_dts(ppg_signal, sampling_freq)
    ppg_signal = ppg_signal.with_columns(pl.col("ppg").is_not_null().alias("ppg_present"))

    if args.align_with_stay:
        ppg_signal, stay_start, stay_end = clip_ppg_signal_to_stay_duration(
            ppg_signal,
            pat_id,
            OmegaConf.load(CONFIG_PATH),
            args.verbose,
        )
        if len(ppg_signal) == 0:
            print(f"No PPG data found for patient {pat_id} in the stay duration.")
            return
        segment_list = get_stay_aligned_segments(
            ppg_signal, args.minor_segment_m / 60, stay_start, args.verbose
        )
    else:
        # align with minutes on the clock
        segment_list = get_segments(ppg_signal, f"{args.minor_segment_m}m", args.verbose)

    if args.verbose:
        print(f"Found {len(segment_list)} {args.minor_segment_m}m segments.")
    segment_features = []
    segment_peaks = []
    for segment in segment_list:
        if segment.is_empty():
            continue
        if "segment_group" in segment.columns:
            # Use datetime stored in 'segment_group' to get truncated datetime
            extracted_features = [
                segment.select(
                    pl.col("segment_group").first().dt.cast_time_unit("ns").alias("datetime"),
                    (pl.col("ppg_present").sum() / pl.col("ppg_present").count()).alias(
                        "signal_completeness"
                    ),
                )
            ]
        else:
            extracted_features = [
                segment.select(
                    pl.col("datetime").first().dt.cast_time_unit("ns"),
                    (pl.col("ppg_present").sum() / pl.col("ppg_present").count()).alias(
                        "signal_completeness"
                    ),
                )
            ]

        # Find peaks for feature extraction
        ppg_peaks = get_peaks(
            segment, sampling_freq, args.peak_detection_method, args.use_fft, verbose=args.verbose
        )
        if ppg_peaks is None:
            continue
        segment_peaks.append(
            pl.DataFrame(data=segment.select(pl.col("datetime"))[ppg_peaks], schema=["peak_dt"])
        )

        if args.extract_pyppg_features:
            print("not extracting pyppg features, because it is not installed")
            # # todo probably too short of a segment for pyppg
            # pyppg_features, (ppg_peaks, upsampled_freq) = get_all_pyppg_features(
            #     segment["ppg"], sampling_freq, get_stat=False, verbose=True
            # )
            # flat_pyppg_feature_stats = None  # todo flatten pyppg_features
        else:
            upsampled_freq = sampling_freq

        if args.extract_hrv_features:
            hrv_features = get_all_hrv_features(
                upsampled_freq,
                peaks=ppg_peaks,
                ppg_signal=segment["ppg"].to_pandas().squeeze(),
                psd_method="welch",
                kubios_features_only=args.kubios_fs,
                show=False,
                verbose=args.verbose,
            )
            if hrv_features is not None:
                extracted_features.append(pl.DataFrame(data=hrv_features, nan_to_null=True))
        segment_features.append(pl.concat(extracted_features, how="horizontal"))
    all_features = pl.concat(segment_features, how="diagonal").sort("datetime")
    all_features = all_features.select(pl.col("datetime"), pl.all().exclude("datetime"))
    all_features.write_parquet(out_file_name)

    if args.write_peak_dts:
        all_peak_dts = pl.concat(segment_peaks, how="vertical").sort("peak_dt")
        all_peak_dts.write_parquet(f"{out_dir}/peak_dts/{pat_id}.parquet")

    if args.verbose:
        print(f"Done processing patient {pat_id} in {(time.time() - start_time)/60:.2f} minutes")
        sys.stdout.flush()


def main(args):
    """
    Main function to preprocess PPG signals for a given patient ID.

    Parameters:
    args (argparse.Namespace): The command-line arguments.
    """
    if args.output_dir == "":
        out_dir = (
            f"{args.output_base_dir}/"
            + datetime.now().strftime("%Y-%m-%d")
            + f"_minor-segment-{args.minor_segment_m}m_"
            + f"corsanov2-{args.corsano_v2_light}-light"
            + f"{'_kubios-fs' if args.kubios_fs else ''}"
            + f"{'_aligned' if args.align_with_stay else ''}"
        )
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Write the command-line arguments to a file
    with open(f"{out_dir}/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if args.write_peak_dts:
        os.makedirs(os.path.join(out_dir, "peak_dts"), exist_ok=True)

    with open(f"{out_dir}/clean_data_origin.txt", "w") as f:
        f.write(f"Clean PPG data loaded from {args.clean_data_dir}\n")

    if args.patient_id is not None:
        process_single_patient(args, out_dir, args.patient_id)
    else:
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 16
        with Pool(processes=n_cpus) as p:
            p.map(partial(process_single_patient, args, out_dir), os.listdir(args.clean_data_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_data_dir",
        required=True,
        help='Path to the directory containing the clean PPG data or subfolder in "/sc-projects/sc-proj-cc08-cassandra/Prospective_Preprocessed/ppg_features"',
    )
    parser.add_argument("--output_base_dir", default=PPG_FEATURES_BASE_PATH)
    parser.add_argument(
        "--output_dir",
        default="",
        help="Precise path to the output directory. Otherwise the directory name will be inferred.",
    )
    parser.add_argument(
        "--patient_id", "-p", type=str, default=None, help="If provided, only process patient with this ID"
    )
    parser.add_argument("--extract_hrv_features", type=bool, default=True)
    parser.add_argument("--extract_pyppg_features", type=bool, default=False)
    parser.add_argument(
        "--minor_segment_m", type=int, default=5, help="Minor segments to calculate features from"
    )
    parser.add_argument(
        "--corsano_v2_light", choices=["green", "red", "infrared"], help="The PPG light for Corsano v2"
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing files")
    parser.add_argument(
        "--kubios_fs", action="store_true", default=False, help="Use reduced Kubios feature set"
    )
    parser.add_argument(
        "--write_peak_dts", action="store_true", default=False, help="Write peak datetimes to file"
    )
    parser.add_argument(
        "--peak_detection_method",
        choices=["bishop", "elgendi"],
        default="elgendi",
        help="Peak detection method (bishop is slow)",
    )
    parser.add_argument(
        "--use_fft", type=bool, default=False, help="Use FFT to interpolate signal before extracting peaks"
    )
    parser.add_argument(
        "--align_with_stay",
        "-a",
        action="store_true",
        default=False,
        help="Align segments with stay duration",
    )

    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Print verbose output")

    args = parser.parse_args()

    if args.corsano_v2_light is None:
        for substr in args.clean_data_dir.split("_"):
            if "corsanov2" in substr:
                args.corsano_v2_light = substr.split("-")[1]
                break
        if args.corsano_v2_light is None:
            raise ValueError(
                "Please provide the Corsano v2 light color or a clean data directory containing it in the name"
            )

    # add clean data base path to clean data dir
    if args.clean_data_dir.find("/") == -1:
        args.clean_data_dir = f"{CLEANED_PPG_BASE_PATH}/{args.clean_data_dir}"
    main(args)
