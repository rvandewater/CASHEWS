import os
import sys
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from scipy.interpolate import CubicSpline
from pathlib import Path

# Add the parent directory of 'preprocessing' to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.prepare_segmentation import prepare_interval_dates
from preprocessing.preprocess_wearable.ppg_paths import RAW_PPG_PATH
from preprocessing.segment_utils import calculate_endpoint


# Helper class to suppress prints when reading data infos
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def load_raw_ppg_for_patient(pat_id, base_path=RAW_PPG_PATH, verbose=False):
    """
    Load the DataFrame for a given patient ID.

    Parameters:
    pat_id (int): The patient ID.
    base_path (str): The base path to the data.
    verbose (bool): If True, print progress.

    Returns:
    pl.DataFrame: The loaded DataFrame.
    """
    df = pl.read_parquet(f"{base_path}/{pat_id}.parquet")

    if "date_time" in df.columns:
        df = df.rename({"date_time": "datetime"})
    # Drop duplicates and sort by datetime
    df = df.unique(subset="datetime", keep="first").sort("datetime")

    if verbose:
        print("Loaded DataFrame.")

    # Drop duplicate datetimes with different values (Corsano V2)
    # df = df.groupby("datetime").agg(pl.all().sort_by("ppg", descending=True).first())
    # df = df.groupby("datetime").agg(pl.all().first())
    sampling_freq = compute_sampling_freq(df, verbose=False)
    if verbose:
        print(f"Loaded DataFrame. Sampling frequency: {sampling_freq} Hz")
    return df, sampling_freq


def load_cleaned_ppg_for_patient(pat_id, base_path, verbose=False):
    """
    Load the DataFrame for a given patient ID.

    Parameters:
    pat_id (int): The patient ID.
    base_path (str): The base path to the cleaned data.
    verbose (bool): If True, print progress.

    Returns:
    pl.DataFrame: The loaded DataFrame with clean ppg data.
    """
    df = pl.read_parquet(f"{base_path}/{pat_id}.parquet")
    sampling_freq = compute_sampling_freq(df, verbose=False)
    if verbose:
        print(f"Loaded DataFrame. Sampling frequency: {sampling_freq} Hz")
    return df, sampling_freq


def add_time_diff_col(df):
    """
    Add a time difference column to the DataFrame.

    Parameters:
    df (pl.DataFrame): The input DataFrame.

    Returns:
    pl.DataFrame: The DataFrame with the added time difference column.
    """
    df = df.with_columns(diff=df["datetime"].diff())
    df[0, "diff"] = df["diff"].median()
    return df


def compute_sampling_freq(df, verbose=False):
    """
    Compute the sampling frequency of the DataFrame.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    verbose (bool): If True, print the sampling frequency.

    Returns:
    float: The computed sampling frequency.
    """
    if "diff" in df.columns:
        sampling_freq = 1 / df["diff"].median().total_seconds()
    else:
        if "datetime" in df.columns:
            diff = df.select(pl.col("datetime").diff().median()).item()
        elif "date_time" in df.columns:
            diff = df.select(pl.col("date_time").diff().median()).item()
        else:
            raise ValueError("No datetime column found.")
        sampling_freq = int(1 / diff.total_seconds())
    if verbose:
        print(f"Sampling Frequency: {sampling_freq}")
    return sampling_freq


def insert_missing_value_dts(df, sampling_freq):
    """
    Adds full datetime column and inserts missing values.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    sampling_freq (float): The sampling frequency.

    Returns:
    pl.DataFrame: The DataFrame with missing value datetimes inserted.
    """
    full_dt_col = (
        pl.datetime_range(
            df.select(pl.col("datetime").min()),
            df.select(pl.col("datetime").max()),
            interval=f"{int(1 / sampling_freq * 1_000_000)}us",
            time_unit="ns",
            time_zone="UTC",
            eager=True,
        )
        .alias("datetime")
        .to_frame()
    )
    df = full_dt_col.join_asof(
        df,
        on="datetime",
        strategy="nearest",
        tolerance=f"{int(1/ sampling_freq*1_000_000 / 2) - 1}us",
        coalesce=True,
    )
    return df


def resample_df(df, resample_cols=None, sampling_freq=32):
    """
    Resample the DataFrame to an equal frequency using cubic spline interpolation.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    sampling_freq (float): The target sampling frequency.
    resample_cols (list): The columns to resample.

    Returns:
    pl.DataFrame: The resampled DataFrame.
    """
    if resample_cols is None:
        resample_cols = ["ppg", "acc"]
    new_cols = {}

    ms_diff = 1000 / sampling_freq
    elapsed_time_ms = df.select((pl.col("datetime") - pl.col("datetime").first()).dt.total_milliseconds())
    new_ts = np.arange(np.ceil(elapsed_time_ms[-1] / ms_diff)) * ms_diff

    # Handle NaNs in the data
    nan_mask = np.isnan(df[resample_cols[0]].to_numpy().squeeze())
    nan_indices_new = []
    if nan_mask.sum != 0:
        nan_idx = np.where(nan_mask)[0]
        idx_new = np.linspace(0, len(df) - 1, len(elapsed_time_ms))

        for idx in nan_idx:
            # Find the range in x_new that corresponds to the original NaN
            start = np.searchsorted(idx_new, idx, side="left")
            end = np.searchsorted(idx_new, idx + 1, side="right")

            # Store all indices in this range
            nan_indices_new.extend(range(start, end))

    for col in resample_cols:
        col_data = df.select(pl.col(col)).to_numpy().squeeze()[~nan_mask]
        assert np.isnan(col_data).sum() == 0, f"Found NaNs in {col} data"
        cs = CubicSpline(elapsed_time_ms.to_numpy().squeeze()[~nan_mask], col_data)
        new_col_data = cs(new_ts)
        new_col_data[nan_indices_new] = np.nan
        new_cols[col] = new_col_data

    return pl.DataFrame(
        {
            "datetime": pl.datetime_range(
                start=df.select(pl.col("datetime").first()),
                end=df.select(pl.col("datetime").last()),
                interval=f"{int(ms_diff * 1000)}us",
                closed="left",
                eager=True,
            ),
            **new_cols,
        }
    )


def get_segments(df, segment_str, verbose=False):
    new_df = df.with_columns(pl.col("datetime").dt.truncate(segment_str).alias("segment_group"))
    groups = new_df.group_by("segment_group", maintain_order=True).agg(pl.all())
    segments = [
        groups.filter(pl.col("segment_group") == the_group).select(pl.all().explode())
        for the_group in groups["segment_group"]
    ]

    if verbose:
        print(f"Found {len(segments)} segments of length {segment_str}.")
    return segments


def get_stay_aligned_segments(df, segment_length_hours: int, start_time, verbose=True):
    # segments aligned to the start of patient stay with the specified length in h.
    segment_duration = timedelta(hours=segment_length_hours)

    max_datetime = df["datetime"].max()
    if isinstance(start_time, datetime):
        start_time = start_time.to_pydatetime()

    start_time = start_time.replace(tzinfo=max_datetime.tzinfo)
    if max_datetime < start_time:
        if verbose:
            print(f"No data found after start_time {start_time}")
        return []

    segments = []
    current_start = start_time

    while current_start < max_datetime:
        current_end = current_start + segment_duration

        segment_data = df.filter((pl.col("datetime") >= current_start) & (pl.col("datetime") < current_end))

        if len(segment_data) > 0:
            segments.append(segment_data)

        # next step
        current_start += segment_duration

    if verbose:
        print(f"Created {len(segments)} segments of {segment_length_hours} hours each")

    return segments


def clip_ppg_signal_to_stay_duration(ppg_signal, pat_id, cfg, verbose=False):
    print(f"Clipping PPG signal for patient {pat_id} to stay duration...")
    with HiddenPrints():
        # Clip signal to stay between surgery start and end time
        admission_time, surgery_time, icu_transfer_time, ward_transfer_time, complication_time, discharge_time = (
            prepare_interval_dates(cfg=cfg, cohort="real_life_set", culling=False)
        )
        end_dict = discharge_time
        endpoint_dict = complication_time
        skip_list = []

    # Get surgery start time for this patient
    try:
        start_date = surgery_time[int(pat_id)]
        procedure_dict = surgery_time
        # Make sure to get as much data as possible
        max_stay_length = timedelta(days=100)

        stay_start, stay_end = calculate_endpoint(
            end_dict,
            endpoint_dict,
            int(pat_id),
            skip_list,
            start_date,
            procedure_dict,
            culling_time=max_stay_length,
        )

        # Filter PPG data to only include the stay period
        ppg_signal = ppg_signal.filter(
            pl.col("datetime").dt.replace_time_zone(None).is_between(stay_start, stay_end)
        )

        print(f"Filtered PPG data for {pat_id} to stay period: {stay_start} to {stay_end}")
        print(f"Remaining data points for {pat_id} : {len(ppg_signal)}")
    except Exception as e:
        print(f"Error getting patient stay times, using all available data: {e}")
        stay_start = ppg_signal["datetime"].min()
        stay_end = ppg_signal["datetime"].max()

    return ppg_signal, stay_start, stay_end
