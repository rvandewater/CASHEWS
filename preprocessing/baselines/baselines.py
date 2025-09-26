# Description: This file contains the code to extract the baseline dynamic dataset
# (i.e. the dataset with dynamic features with a dummy feature) from the given data.
import json
import logging
import os
import sys
from datetime import datetime, timedelta

import polars as pl
from joblib import Parallel, delayed

from preprocessing.data_utils.utils import convert_to_timedelta
from preprocessing.post_processing.filter_cohort import (
    filter_columns_by_null_percentage,
)
from preprocessing.post_processing.save import save_modalities_dict
from preprocessing.prepare_segmentation import load_modalities
from preprocessing.segment_utils import aggregate_pre_prediction_data, calculate_endpoint
from omegaconf import DictConfig


def extract_baseline_aggregated(
    base_static: pl.DataFrame,
    admission_dict: dict,
    endpoint_dict: dict,
    procedure_dict: dict,
    start_dict: dict,
    config: object,
    datetime_col: str = "datetime",
    id_col: str = "id",
    n_jobs: int = -1,
    retro_threshold: int = 70,
    cohort: str = "normal_ward"
) -> None:
    """
    Extracts the baseline aggregated dataset from the given data.

    This function combines static and aggregated dynamic features for each patient.
    It processes numerical and categorical modalities, aggregates pre-prediction data,
    and saves the resulting dataset and metadata to disk.

    Args:
        base_static (pl.DataFrame): Static features DataFrame.
        admission_dict (dict): Admission information per patient.
        endpoint_dict (dict): Endpoint times per patient.
        procedure_dict (dict): Procedure information per patient.
        start_dict (dict): Start times per patient.
        config (object): Configuration object with output root path.
        datetime_col (str, optional): Name of the datetime column. Defaults to "datetime".
        id_col (str, optional): Name of the patient ID column. Defaults to "id".
        cohort (str, optional): type of data to aggregate. If ICU will include preoperative. Normal ward will
        additionally include ICU.
    Returns:
        None
    """
    cfg = config
    root = cfg.root
    cat_modalities, copra_modalities, ishmed_modalities, num_modalities = load_modalities(cfg)
    name = ""
    if cohort == "ICU":
        name = "intra-operative"
    elif cohort == "normal_ward":
        name = "post-operative"
    else:
        raise ValueError(f"Unknown cohort type: {cohort}. Please use 'ICU' or 'normal_ward'.")
    logging.info(f"Creating static baseline for {name} period...")
    for item in num_modalities:
        num_modalities[item] = num_modalities[item].with_columns(pl.col(datetime_col).cast(pl.Datetime))

    for item in num_modalities:
        num_modalities[item] = num_modalities[item].with_columns(pl.col(datetime_col).cast(pl.Datetime))
    retro_list = []
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    def process_patient(patient_key, logger):
        if endpoint_dict.get(patient_key) and endpoint_dict.get(patient_key) < start_dict.get(patient_key):
            logger.info(f"Patient {patient_key} has endpoint before start, skipping")
            return None

        if procedure_dict.get(patient_key) is not None:
            static_retrospective = aggregate_pre_prediction_data(
                admission_dict=admission_dict,
                datetime_col=datetime_col,
                categorical_dfs=cat_modalities,
                numerical_dfs=num_modalities,
                id_col=id_col,
                patient_key=patient_key,
                procedure_dict=procedure_dict,
                start_date=start_dict.get(patient_key),
                cohort= cohort
            )
        else:
            static_retrospective = None

        return static_retrospective


    # Use joblib to parallelize the process_patient function
    retro_list = Parallel(n_jobs=n_jobs)(delayed(process_patient)(key, logger) for key in start_dict.keys())

    retro_list = [x for x in retro_list if x is not None]
    # Filter out None values from the results
    retro_aggregated = pl.concat(retro_list, how="diagonal_relaxed")

    # retro_aggregated = pl.concat(retro_list, how="align")

    version = f"dataset_baseline_cohort_{name}_static_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    save_path = os.path.join(root, "Prospective_Preprocessed", "yaib_format", version)
    logging.info(f"Saving dataset to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    retro_aggregated = filter_columns_by_null_percentage(retro_aggregated, retro_threshold)
    base_static = base_static.with_columns(pl.col(id_col).cast(pl.Int32))
    static = base_static.join(retro_aggregated, on=id_col, how="inner")
    static.write_parquet(os.path.join(save_path, "sta.parquet"))

    outcome = base_static.select(pl.col("id")).with_columns(
        pl.when(pl.col("id").is_in(endpoint_dict.keys())).then(1).otherwise(0).alias("label")
    )
    outcome.write_parquet(os.path.join(save_path, "outc.parquet"))
    # retro_aggregated = pl.concat(base_static, how="diagonal_relaxed")

    vars_dict = {
        "GROUP": "id",
        "LABEL": "label",
        "STATIC": base_static.select(pl.exclude("id")).columns,
    }

    modality_dict, total = save_modalities_dict(static_data=base_static, copra_modalities=copra_modalities,
                                                ishmed_modalities=ishmed_modalities, wearable_modalities=None)
    logging.info(f"Saved modality dict to {save_path}")

    with open(os.path.join(save_path, "vars.gin"), "w") as handle:
        # pickle.dump(vars_dict, handle)
        handle.write("vars = " + json.dumps(vars_dict))
        handle.write("\n")
        handle.write("modality_mapping = " + json.dumps(modality_dict))


def extract_baseline_flat(
    base_static: pl.DataFrame,
    complication_time: dict,
    config: DictConfig,
    admission_dict: dict = None,
) -> None:
    """
    Extracts the baseline flat dataset (i.e. the dataset with only static features) from the given data.
    Used to extract the preoperative data for benchmarking.

    Args:
        base_static (pl.DataFrame): Static features DataFrame.
        complication_time (dict): Dictionary mapping patient IDs to complication times.
        config (object): Configuration object with output root path.
        admission_dict (dict): Filter by admission dictionary. If None, all patients are included.
    Returns:
        None
    """
    cfg = config
    root = cfg.root
    if admission_dict is not None:
        # Filter base_static by admission_dict
        base_static = base_static.filter(pl.col("id").is_in(list(admission_dict.keys())))
    version = f"dataset_baseline_cohort_pre-operative_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    save_path = os.path.join(root, "Prospective_Preprocessed", "yaib_format", version)
    logging.info(f"Saving dataset to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    outcome = base_static.select(pl.col("id")).with_columns(
        pl.when(pl.col("id").is_in(complication_time.keys())).then(1).otherwise(0).alias("label")
    )
    outcome.write_parquet(os.path.join(save_path, "outc.parquet"))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # base_static.join(pl.from_pandas(patient_details[['id','birth_day','birth_month','birth_year']]),on="id", how="left")
    base_static.write_parquet(os.path.join(save_path, "sta.parquet"))

    vars_dict = {
        "GROUP": "id",
        "LABEL": "label",
        "STATIC": base_static.select(pl.exclude("id")).columns,
    }

    modality_dict = save_modalities_dict(static_data=base_static)
    logging.info(f"Saved modality dict to {save_path}")

    with open(os.path.join(save_path, "vars.gin"), "w") as handle:
        # pickle.dump(vars_dict, handle)
        handle.write("vars = " + json.dumps(vars_dict))
        handle.write("\n")
        handle.write("modality_mapping = " + json.dumps(modality_dict))


def extract_baseline_dynamic(
    base_static: pl.DataFrame,
    complication_time: dict,
    outtake_time: dict,
    surgery_time: dict,
    root: str,
    complications: dict,
    interval: str = "1h",
    detection_interval: timedelta = timedelta(hours=6),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Extracts the baseline dynamic dataset (i.e. the dataset with dynamic features with a dummy feature) from the given data.

    Args:
        base_static (pl.DataFrame): Static features DataFrame.
        complication_time (dict): Dictionary mapping patient IDs to complication times.
        outtake_time (dict): Dictionary mapping patient IDs to outtake times.
        surgery_time (dict): Dictionary mapping patient IDs to surgery times.
        root (str): Root directory for saving output files.
        complications (dict): Dictionary of complications per patient.
        interval (str, optional): Time interval for segmentation (e.g., "6h"). Defaults to "6h".
        detection_interval (timedelta, optional): Time window before complication for labeling. Defaults to 24 hours.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the segmented dynamic DataFrame and the outcome DataFrame.
    """
    # outcome = base_static.select(pl.col("id")).with_columns(pl.when(pl.col("id").is_in(complications["id"])).then(1).otherwise(0).alias("label"))
    patient_start = {}
    patient_end = {}
    for patient_key, value in outtake_time.items():
        if patient_key in surgery_time.keys():
            patient_start[patient_key], patient_end[patient_key] = calculate_endpoint(
                outtake_time, complications, patient_key, surgery_time
            )

    # Define start and end dictionaries
    start_dict = patient_start
    end_dict = patient_end

    # Generate the date ranges and create the DataFrame
    data = []
    for key in start_dict:
        start = start_dict[key]
        end = end_dict[key]
        date_range = pl.datetime_range(start, end, interval, eager=True).alias("datetime")
        df = pl.DataFrame({"id": [key] * len(date_range), "datetime": date_range})
        data.append(df)

    # Concatenate all DataFrames
    segmented_df = pl.concat(data)
    segmented_df = segmented_df.to_pandas()

    # print(segmented_df)
    # Add the "label" column
    def label_func(row, complication_time):
        # print(row)
        id = row["id"]
        datetime_val = row["datetime"]
        if id in complication_time.keys():
            time = complication_time[id]
            if datetime_val > time - detection_interval:
                return 1
        return 0

    segmented_df["label"] = segmented_df.apply(lambda row: label_func(row, complication_time), axis=1)
    segmented_df["dummy_feature"] = segmented_df.groupby("id").cumcount()
    version = f"dataset_baseline_dynamic_{interval}_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
    save_path = os.path.join(root, "Prospective_Preprocessed", "yaib_format", version)
    logging.info(f"Saving dataset to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    segmented_df = pl.from_pandas(segmented_df)
    # outcome = base_static.select(pl.col("id")).with_columns(
    #     pl.when(pl.col("id").is_in(complication_time.keys())).then(1).otherwise(0).alias("label"))
    # outcome.write_parquet(os.path.join(save_path, "outc.parquet"))
    outcome = segmented_df.select(["id", "datetime", "label"])
    outcome.write_parquet(os.path.join(save_path, "outc.parquet"))

    segmented_df = segmented_df.drop(["label"])
    segmented_dataframe_pd = convert_to_timedelta(segmented_df)
    segmented_dataframe_pd.to_parquet(os.path.join(save_path, "dyn.parquet"))

    base_static.write_parquet(os.path.join(save_path, "sta.parquet"))

    vars_dict = {
        "GROUP": "id",
        "SEQUENCE": "datetime",
        "LABEL": "label",
        "DYNAMIC": segmented_df.select(pl.exclude(["id", "datetime"])).columns,
        "STATIC": base_static.select(pl.exclude("id")).columns,
    }
    with open(os.path.join(save_path, "vars.gin"), "w") as handle:
        # pickle.dump(vars_dict, handle)
        handle.write("vars = " + json.dumps(vars_dict))
    return segmented_dataframe_pd, outcome
