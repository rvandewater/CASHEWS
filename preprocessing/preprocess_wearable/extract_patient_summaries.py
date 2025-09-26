import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import sys
import argparse
import logging
import polars as pl
from omegaconf import OmegaConf
import datetime
# Add the parent directory of 'preprocessing' to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.preprocess_wearable.ppg_paths import RAW_PPG_PATH
from preprocessing.preprocess_wearable.wearable_utils import (
    clip_ppg_signal_to_stay_duration,
    load_cleaned_ppg_for_patient,
    load_raw_ppg_for_patient,
)
from preprocessing.dataloader.load_methods import return_core

cfg = OmegaConf.load(str(Path(__file__).resolve().parent.parent / "config.yaml"))
PPG_CLEANED_PATH = cfg.dataloader.folders.ppg_cleaned

#"/sc-projects/sc-proj-cc08-cassandra/Prospective_Preprocessed/ppg_cleaned/2025-01-14_gap-60s_corsanov2-infrared-light"
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = f"patient_summaries_{current_datetime}.parquet"


def create_patient_summary(patient_id, ppg_path=PPG_CLEANED_PATH, raw_data=True):
    """Creates a summary row for a single patient with data quality analysis"""
    cfg = OmegaConf.load("config.yaml")
    # Load PPG data
    ppg_file = Path(ppg_path) / f"{patient_id}.parquet"
    if not ppg_file.exists():
        logging.info(f"No PPG data found for patient {patient_id}")
        return None

    # core_file = Path(cfg.dataloader.folders.core) / f"{patient_id}.parquet"
    # if not core_file.exists():
    #     logging.info(f"No Core data found for patient {patient_id}")
    #     core_df = None
    # else:
    #     core_df = pl.read_parquet(core_file)

    summary = {}

    if raw_data:
        summary["corsano_v1"] = False
        summary["corsano_v2"] = False
        logging.info(f"Loading raw PPG data from {ppg_file}")
        ppg_df, sampling_freq = load_raw_ppg_for_patient(patient_id, ppg_path)
        if "ppg_infrared_value" in ppg_df.columns:
            summary["corsano_v2"] = True
            if "ppg" in ppg_df.columns:
                summary["corsano_v1"] = True
                ppg_df = ppg_df.with_columns(
                    pl.col("ppg")
                    .fill_null(pl.col("ppg_infrared_value"))
                    .fill_nan(pl.col("ppg_infrared_value"))
                )
            else:
                ppg_df = ppg_df.with_columns(pl.col("ppg_infrared_value").alias("ppg"))
        else:
            summary["corsano_v1"] = True
    else:
        logging.info(f"Loading clean PPG data from {ppg_file}")
        ppg_df, sampling_freq = load_cleaned_ppg_for_patient(patient_id, ppg_path)
    logging.info(f"Loaded {len(ppg_df)} rows")
    ppg_df = ppg_df.with_columns(pl.col("datetime").cast(pl.Datetime))
    # Basic PPG statistics
    start_time = ppg_df["datetime"].min()
    end_time = ppg_df["datetime"].max()
    duration_seconds = (end_time - start_time).total_seconds()

    # Calculate sampling frequency
    # time_diffs = ppg_df['datetime'].diff().dt.total_seconds()
    # time_diffs_filtered = time_diffs.filter((time_diffs > 0) & (time_diffs < 1))
    # if len(time_diffs_filtered) > 0:
    #    sampling_freq = 1 / time_diffs_filtered.median()
    # else:
    #    logging.info(f"Warning: Could not calculate sampling frequency for patient {patient_id}")
    #    sampling_freq = 25.0  # Fallback to expected frequency????

    # Basic summary
    summary.update(
        {
            "patient_id": patient_id,
            "recording_start": start_time,
            "recording_end": end_time,
            "recording_total_duration": duration_seconds / 3600,
            "recording_hours": len(ppg_df) / sampling_freq / 3600,
            "recording_missing_fraction": min(
                1.0 - (len(ppg_df) / sampling_freq / 3600 / (duration_seconds / 3600)), 0.0
            ),
            "total_samples": len(ppg_df),
            "missing_values": ppg_df["ppg"].is_null().sum(),
            "nan_values": ppg_df["ppg"].is_nan().sum(),
            "zero_values": ppg_df.filter(pl.col("ppg") == 0).height,
            "sampling_freq": float(sampling_freq),
        }
    )

    ppg_df, _, _ = clip_ppg_signal_to_stay_duration(ppg_df, patient_id, cfg, verbose=False)
    summary.update({"recording_hours_clipped": len(ppg_df) / sampling_freq / 3600})

    # Gap detection
    time_diffs = ppg_df["datetime"].diff().dt.total_seconds()
    expected_interval = 1 / sampling_freq

    gap_ranges = {
        "lt_1s": (expected_interval * 2, 1),
        "1s_to_5s": (1, 5),
        "5s_to_1m": (5, 60),
        "1m_to_5m": (60, 300),
        "5m_to_60m": (300, 3600),
        "gt_60m": (3600, 1e10),
    }

    for gap_type, (min_gap, max_gap) in gap_ranges.items():
        gaps = time_diffs.filter((time_diffs > min_gap) & (time_diffs <= max_gap))

        summary[f"{gap_type}_gap_count"] = len(gaps)
        summary[f"{gap_type}_gap_duration"] = float(gaps.sum() if len(gaps) > 0 else 0)
        summary[f"{gap_type}_gap_frac"] = float(
            (gaps.sum() / duration_seconds * 100) if len(gaps) > 0 else 0
        )

    # Calculate completeness
    total_gap_duration = sum(summary[f"{gap_type}_gap_duration"] for gap_type in gap_ranges.keys())
    expected_samples = int((duration_seconds - total_gap_duration) * sampling_freq)
    valid_samples = summary["total_samples"] - summary["missing_values"] - summary["nan_values"]

    summary["completeness_pct"] = min(
        100.0 * valid_samples / expected_samples if expected_samples > 0 else 0.0, 100.0
    )
    # Core Info
    # if core_df is not None:
    #     if len(core_df) == 0:
    #         logging.info(f"Data for {patient_id} in core empty.")
    #         return
    #     if "date_time" in core_df.columns:
    #         date_column = "date_time"
    #     elif "datetime" in core_df.columns:
    #         date_column = "datetime"
    #     core_df = core_df.with_columns(pl.col(date_column).dt.replace_time_zone(None))
    #     if len(core_df.columns) < 5:
    #         return
    #     if len(core_df.columns) == 6:
    #         core_df.columns = ["datetime", "skin_temp", "core_temp", "empty", "quality", "device_uuid"]
    #     elif len(core_df.columns) == 7:
    #         core_df.columns = [
    #             "datetime",
    #             "skin_temp",
    #             "core_temp",
    #             "empty",
    #             "quality",
    #             "empty2",
    #             "device_uuid",
    #         ]
    #     else:
    #         core_df.columns = ["datetime", "skin_temp", "core_temp", "empty", "quality"]
    core_df = return_core(data_dir=cfg.dataloader.core_folder, pid=patient_id)
    core_df = core_df.with_columns(pl.col("datetime").cast(pl.Datetime))
    start_time = core_df["datetime"].min()
    end_time = core_df["datetime"].max()
    duration_seconds = (end_time - start_time).total_seconds()
    time_diffs = core_df["datetime"].diff().dt.total_seconds()
    diff = time_diffs.median()
    sampling_freq = 1 / diff

    summary.update(
        {
            "core_recording_start": start_time,
            "core_recording_end": end_time,
            "core_recording_total_duration": duration_seconds / 3600,
            "core_recording_hours": len(core_df) / sampling_freq / 3600,
            "core_recording_missing_fraction": min(
                1.0 - (len(core_df) / sampling_freq / 3600 / (duration_seconds / 3600)), 0.0
            ),
            "core_total_samples": len(core_df),
            "core_missing_values": core_df["core_temp"].is_null().sum(),
            "core_nan_values": core_df["core_temp"].is_nan().sum(),
            "core_zero_values": core_df.filter(pl.col("core_temp") == 0).height,
        }
    )

    core_df, _, _ = clip_ppg_signal_to_stay_duration(core_df, patient_id, cfg, verbose=False)
    summary.update({"core_recording_hours_clipped": len(core_df) / sampling_freq / 3600})

    gap_ranges = {"9m_to_60m": (540, 3600), "gt_60m": (3600, 1e10)}

    for gap_type, (min_gap, max_gap) in gap_ranges.items():
        gaps = time_diffs.filter((time_diffs > min_gap) & (time_diffs <= max_gap))

        summary[f"core_{gap_type}_gap_count"] = len(gaps)
        summary[f"core_{gap_type}_gap_duration"] = float(gaps.sum() if len(gaps) > 0 else 0)
        summary[f"core_{gap_type}_gap_frac"] = float(
            (gaps.sum() / duration_seconds * 100) if len(gaps) > 0 else 0
        )

    # Calculate completeness
    total_gap_duration = sum(
        summary[f"core_{gap_type}_gap_duration"] for gap_type in gap_ranges.keys()
    )
    expected_samples = int((duration_seconds - total_gap_duration) * sampling_freq)
    valid_samples = (
            summary["core_total_samples"] - summary["core_missing_values"] - summary["core_nan_values"]
    )

    summary["core_completeness_pct"] = min(
        100.0 * valid_samples / expected_samples if expected_samples > 0 else 0.0, 100.0
    )
    logging.info(f"Successfully added summary for patient {patient_id}")
    return pl.DataFrame([summary])

# except Exception as e:
# logging.error(f"Error processing patient {patient_id}: {str(e)}")
# logging.error(traceback.format_exc())
# return None


def process_folder(ppg_path, limit_n_patients=None, out_file_name=OUTPUT_FILE, raw_data=False):
    """Process all patients in a folder"""
    logging.info(f"\nProcessing PPG data from: {ppg_path}")
    ppg_path = Path(ppg_path)

    # results = []
    # errors = 0

    patient_files = list(ppg_path.glob("*.parquet"))
    logging.info(f"Found {len(patient_files)} patient files")

    if limit_n_patients is not None:
        logging.info(f"Using first {limit_n_patients} files")
        patient_files = patient_files[:limit_n_patients]
    patient_ids = [pat_file.stem for pat_file in patient_files]

    n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 16
    with Pool(processes=n_cpus) as p:
        results = p.map(partial(create_patient_summary, ppg_path=ppg_path, raw_data=raw_data), patient_ids)

    results = [r for r in results if r is not None]
    errors = len(patient_files) - len(results)

    final_df = pl.concat(results, how="diagonal")
    logging.info(f"\nProcessed {len(patient_files)} patients with {errors} errors.")
    logging.info(f"Created summary DataFrame with shape: {final_df.shape}")

    final_df.write_parquet(out_file_name)
    logging.info(f"Results saved to {out_file_name}")

    return final_df


# logging.info("Starting analysis...")
# summary_df = process_folder(RAW_PPG_PATH, out_file_name="raw_patient_summaries.parquet", raw_data=True)

# if summary_df is not None:
#     logging.info("\nFinal Summary:")
#     logging.info(summary_df.select(pl.all()))


def main():
    parser = argparse.ArgumentParser(description="Process PPG data and generate patient summaries.")
    parser.add_argument("--ppg-path", type=str, default=PPG_CLEANED_PATH, help="Path to PPG data folder.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of patients to process.")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file name.")
    parser.add_argument("--raw-data", action="store_false", default=True, help="Use raw PPG data instead of cleaned data.")

    args = parser.parse_args()
    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    )
    # Example usage
    logging.info("Logger is set up and ready.")
    logging.info(f"arguments: {args}")
    logging.info("Starting analysis...")

    if args.raw_data:
        logging.info(f"Using raw data - overwriting specified ppg-path with raw data path ({RAW_PPG_PATH})")
        args.ppg_path = RAW_PPG_PATH

    summary_df = process_folder(
        ppg_path=args.ppg_path,
        limit_n_patients=args.limit,
        out_file_name=args.output,
        raw_data=args.raw_data
    )

    if summary_df is not None:
        logging.info("\nFinal Summary:")
        logging.info(summary_df.select(pl.all()))


if __name__ == "__main__":
    main()