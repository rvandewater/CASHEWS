import argparse
import os
import sys
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import neurokit2 as nk
import numpy as np
import pandas as pd
import polars as pl
import pypg

from preprocessing.preprocess_wearable.ppg_paths import (
    CLEANED_PPG_BASE_PATH,
    RAW_PPG_PATH,
)
from preprocessing.preprocess_wearable.wearable_utils import (
    add_time_diff_col,
    insert_missing_value_dts,
    load_raw_ppg_for_patient,
)


def split_df_by_time_gaps(df, gap_seconds=60, verbose=False):
    """
    Split the DataFrame by time gaps.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    gap_seconds (int): The gap in seconds to split the DataFrame.
    verbose (bool): If True, print the number of segments.

    Returns:
    list: A list of DataFrames split by the time gaps.
    """

    # Define the threshold for splitting in milliseconds (100 ms = 100_000 microseconds)
    threshold = pl.duration(seconds=gap_seconds)

    # Merge Corsano v1 and v2 PPG data for edge cases
    if (
        "acc" in df.columns
        and df["acc"].count() > 0
        and "ppg_ambient_value" in df.columns
        and df["ppg_ambient_value"].count() > 0
    ):
        # Create a boolean mask to detect the swap point
        swap_mask = df["acc"].is_null() & df["ppg_ambient_value"].is_not_null()
        v1_first = True
        if swap_mask.head(1).item():
            swap_mask = df["acc"].is_null() & df["ppg_ambient_value"].is_not_null()
            v1_first = False

        # Find the first index where the swap happens
        swap_idx = swap_mask.to_frame().select(pl.col("acc").arg_true().first()).item()

        # Split the DataFrame
        df_part1 = df[:swap_idx]
        df_part2 = df[swap_idx:]

        if v1_first:
            df_part2 = df_part2.drop_nulls("ppg_ambient_value")
            df_part1 = df_part1.drop_nulls("acc")
        else:
            df_part2 = df_part2.drop_nulls("acc")
            df_part1 = df_part1.drop_nulls("ppg_ambient_value")
        return split_df_by_time_gaps(df_part1, gap_seconds, verbose) + split_df_by_time_gaps(
            df_part2, gap_seconds, verbose
        )

    # Create a cumulative sum based on whether "diff" exceeds the threshold
    df_with_groups = df.with_columns(
        ((pl.col("diff") > threshold) | (pl.col("diff") < pl.duration(milliseconds=0)))
        .cum_sum()
        .alias("group_id")
    )

    # Split the DataFrame based on the unique values in "group_id"
    split_dfs = [
        (group_id[0], group_df.drop("group_id")) for group_id, group_df in df_with_groups.group_by("group_id")
    ]
    split_dfs = sorted(split_dfs, key=lambda x: x[0])

    if verbose:
        print(f"Split df into {len(split_dfs)} segments")
        for i in range(len(split_dfs)):
            print(f"\tSegment {i}: {split_dfs[i][1].select(pl.len()).item()} rows")

    # split_dfs will now contain a list of DataFrames, each split according to the condition.
    return [split_df[1] for split_df in split_dfs]


def compute_acc_cutoff(df, verbose=False):
    """
    Compute the accuracy cutoff for the accelerometer magnitude.

    Parameters:
    df (np.ndarray): The entire patient dataframe.
    verbose (bool): If True, print the accuracy cutoff and percent of clean data.

    Returns:
    float: The computed accuracy cutoff.
    """
    all_acc_data = pl.Series("accs", dtype=pl.Int64)
    if "ppg" in df.columns and len(df.filter(pl.col("ppg").is_not_null())) > 0:
        all_acc_data.append(df.select(pl.col("ppg").cast(pl.Int64)).to_series())
    if "ppg_ambient_value" in df.columns and len(df.filter(pl.col("ppg_ambient_value").is_not_null())) > 0:
        all_acc_data.append(df.select(pl.col("ppg_ambient_value").cast(pl.Int64)).to_series())
    cutoff = all_acc_data.mean() + all_acc_data.std()
    if verbose:
        print(
            f"Percent of data with movement artifacts: {np.sum(np.where(all_acc_data > cutoff, 1, 0)) / len(all_acc_data) * 100:.2f}"
        )
    print(f"Accuracy cutoff: {cutoff}")
    return float(cutoff)


def interpolate_at_acc_cutoff(
    ppg, acc_magnitude, acc_cutoff, interpolation_method="monotone_cubic", verbose=False
):
    """
    Interpolate the PPG signal when the accelerometer value is larger than mean + std.dev.

    Parameters:
    ppg (np.ndarray): The PPG signal.
    acc_magnitude (np.ndarray): The accelerometer magnitude signal.
    interpolation_method (str): The interpolation method to use.
    verbose (bool): If True, print the accuracy cutoff and percent of clean data.

    Returns:
    np.ndarray: The interpolated PPG signal.
    """
    # todo: potential for improvement by checking ppg quality before putting nan values
    ppg_ma_cleaned = np.where(acc_magnitude < acc_cutoff, ppg, np.nan)
    ppg_ma_cleaned = pd.DataFrame(ppg_ma_cleaned)

    if len(ppg_ma_cleaned.dropna()) < 2:
        return pd.DataFrame([])

    # monotone cubic spline interpolation of the filtered ppg signal
    # monotone prevents implausible "overshoots" or "undershoots" in the y-direction
    ppg_ma_cleaned = nk.signal_interpolate(ppg_ma_cleaned.iloc[:, 0], method=interpolation_method)
    if verbose:
        print(f"Interpolated PPG when accelerometer signal > {acc_cutoff}")
    return ppg_ma_cleaned


def filter_ppg_signal(raw_ppg, sampling_freq, filter_type="cheby", low_pass=0.5, high_pass=5, verbose=False):
    """
    Filter the PPG signal.

    Parameters:
    raw_ppg (np.ndarray): The raw PPG signal.
    sampling_freq (float): The sampling frequency.
    filter_type (str): The type of filter to use ('butter' or 'cheby').
    verbose (bool): If True, plot the raw signal.

    Returns:
    np.ndarray: The filtered PPG signal.
    """
    try:
        if filter_type == "butter":
            clean_ppg = pypg.filters.butterfy(
                raw_ppg,
                cutoff_frequencies=[low_pass, high_pass],
                sampling_frequency=sampling_freq,
                filter_type="bandpass",
                filter_order=4,
            )
        elif filter_type == "cheby":
            clean_ppg = pypg.filters.chebyfy(
                raw_ppg,
                cutoff_frequencies=[low_pass, high_pass],
                sampling_frequency=sampling_freq,
                filter_type="bandpass",
                filter_order=4,
            )
        else:
            raise NotImplementedError
    except ValueError as e:
        print(f"Error filtering PPG signal: {e}; skipping...")
        return None
    if verbose:
        print(f"Applied {filter_type} filter to PPG signal")
    return clean_ppg


def preprocess_ppg_for_patient(
    df,
    sampling_freq,
    corsano_v2_light="green",
    time_gaps_s=60,
    remove_high_movement_areas=False,
    calculate_quality=True,
    verbose=False,
):
    """
    Preprocess the PPG signal for a given patient ID.

    Parameters:
    pat_id (int): The patient ID.
    time_gaps_s (int): The time gap in seconds to split the data. Smaller gaps are interpolated

    Returns:
    tuple: A tuple containing the cleaned PPG signals and quality scores.
    """
    df = add_time_diff_col(df)
    split_dfs = split_df_by_time_gaps(df, time_gaps_s, verbose=verbose)
    acc_cutoff = compute_acc_cutoff(df)

    cleaned_dfs = []
    quality_scores = []
    for i, split_df in enumerate(split_dfs):
        if verbose:
            print(f"Processing split {i}...")

        # Test if split is corsano v1 or v2
        if "ppg" in split_df.columns and len(split_df.filter(pl.col("ppg").is_not_null())) > 0:
            ppg_key, acc_key = "ppg", "acc"
        else:
            ppg_key, acc_key = f"ppg_{corsano_v2_light}_value", "ppg_ambient_value"

        filtered_ppg = filter_ppg_signal(
            split_df[ppg_key].to_pandas(), sampling_freq, filter_type="cheby", verbose=False
        )
        if filtered_ppg is None:
            continue

        # todo add quality check for skewness > 0.2 and autocorrelation > 0.8?
        if remove_high_movement_areas:
            filtered_ppg = interpolate_at_acc_cutoff(
                filtered_ppg, split_df[acc_key].to_pandas(), acc_cutoff, verbose=verbose
            )
            if len(filtered_ppg) == 0:
                # too many movement artifacts
                continue
        filtered_df = pl.DataFrame(
            {
                "datetime": split_df.select(pl.col("datetime")),
                "acc": split_df.select(pl.col(acc_key)),
                "ppg": filtered_ppg,
            }
        )

        # Add nan values and create regular DT column
        filtered_df = insert_missing_value_dts(filtered_df, sampling_freq)
        cleaned_dfs.append(filtered_df)
        if calculate_quality:
            quality_scores.append(pypg.quality.skewness(filtered_ppg, int(sampling_freq)))

    return cleaned_dfs, quality_scores


def process_single_patient(args, out_dir, pat_id):
    if "parquet" in pat_id:
        pat_id = pat_id.split(".")[0]
    out_file_name = f"{out_dir}/{pat_id}.parquet"
    if os.path.exists(out_file_name) and not args.overwrite:
        print(f"Output file for patient {pat_id} already exists, pass --overwrite to overwrite.")
        return pd.DataFrame(columns=["patient_id", "split_id", "quality_score"])

    if args.verbose:
        print(f"Processing patient {pat_id}...")
        sys.stdout.flush()
    try:
        raw_df, sampling_freq = load_raw_ppg_for_patient(
            pat_id, base_path=args.raw_data_dir, verbose=args.verbose
        )
        if len(raw_df) == 0:
            print(f"No data for patient {pat_id}")
            return pd.DataFrame(columns=["patient_id", "split_id", "quality_score"])
        cleaned_ppgs, quality_scores = preprocess_ppg_for_patient(
            raw_df,
            sampling_freq,
            args.corsano_v2_light,
            args.time_gaps_s,
            remove_high_movement_areas=args.remove_high_movement_areas,
            calculate_quality=args.save_signal_qualities,
            verbose=args.verbose,
        )
        combined_df = pl.concat(
            [df.with_columns(pl.lit(i).alias("split_id")) for i, df in enumerate(cleaned_ppgs)],
            how="vertical",
        )
        # Round the acc and ppg columns to int64 to save storage space
        if args.round_to_int:
            combined_df = combined_df.select(
                pl.col("datetime"),
                pl.col("acc").round().cast(pl.Int64),
                pl.col("ppg").round().cast(pl.Int64),
                pl.col("split_id"),
            )
        combined_df.write_parquet(out_file_name)
        quality_scores_df = pd.DataFrame(
            {
                "patient_id": args.patient_id,
                "split_id": np.arange(len(quality_scores)),
                "quality_score": quality_scores,
            }
        )
        if args.verbose:
            print(f"Done with patient {pat_id}.\n\n===============\n\n")
            sys.stdout.flush()
    except Exception as e:
        print(f"Error processing patient {pat_id}: {e}")
        quality_scores_df = pd.DataFrame(columns=["patient_id", "split_id", "quality_score"])
    return quality_scores_df


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
            + f"_gap-{args.time_gaps_s}s_corsanov2-{args.corsano_v2_light}-light"
        )
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.patient_id is not None:
        all_quality_scores = process_single_patient(args, out_dir, args.patient_id)
    else:
        n_cpus = (
            int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else args.n_jobs
        )
        with Pool(n_cpus) as p:
            quality_scores_list = p.map(
                partial(process_single_patient, args, out_dir), os.listdir(args.raw_data_dir)
            )

        all_quality_scores = pd.concat(quality_scores_list, axis=0)
    if args.save_signal_qualities:
        all_quality_scores.to_csv(f"{out_dir}/quality_scores.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default=RAW_PPG_PATH)
    parser.add_argument("--output_base_dir", default=CLEANED_PPG_BASE_PATH)
    parser.add_argument(
        "--output_dir",
        default="",
        help="Precise path to the output directory. Otherwise the directory name will be inferred.",
    )
    parser.add_argument(
        "--patient_id", "-p", type=str, default=None, help="If provided, only process patient with this ID"
    )
    parser.add_argument("--time_gaps_s", type=int, default=60, help="Time gap in seconds to split the data")
    parser.add_argument(
        "--corsano_v2_light",
        default="infrared",
        choices=["green", "red", "infrared"],
        help="The PPG light for Corsano v2",
    )
    parser.add_argument(
        "--remove_high_movement_areas",
        action="store_true",
        default=False,
        help="Do not remove areas with large acceleration magnitude",
    )
    parser.add_argument(
        "--save_signal_qualities",
        "-ssq",
        action="store_true",
        default=False,
        help="Save the signal qualities",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing files")
    parser.add_argument(
        "--precise_values, -pv",
        dest="round_to_int",
        action="store_false",
        default=True,
        help="Don't round to int64",
    )
    parser.add_argument("--n_jobs", "-nj", default=None, type=int, help="Number of parallel jobs")

    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Print verbose output")

    args = parser.parse_args()
    main(args)
