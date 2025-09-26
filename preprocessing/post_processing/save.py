import json
import logging
import os
from datetime import timedelta

import pandas as pd
import polars as pl
from polars import selectors as cs

from preprocessing.data_utils.utils import convert_to_timedelta


def save_modalities_dict(
        segmented_dataframe: pl.DataFrame = None,
        static_data: pl.DataFrame = None,
        copra_modalities: dict = None,
        ishmed_modalities: dict = None,
        cat_modalities: dict = None,
        wearable_modalities=None,
) -> (dict, list):
    """
    Creates dictionary of modalities from the segmented dataframe and static data.
    :param segmented_dataframe:
    :param static_data:
    :param copra_modalities:
    :param ishmed_modalities:
    :param cat_modalities:
    :param wearable_modalities:
    :return:
        modality_dict: Dictionary with modalities as keys and their columns as values.
        accounted_for: List of all columns that are accounted for in the modality_dict.
    """
    if wearable_modalities is None:
        wearable_modalities = [
            "activity",
            "core",
            "ppgfeature",
            "ppgembedding",
        ]
    if copra_modalities is None:
        copra_modalities = {}
        logging.info("copra_modalities is None, initializing to empty dictionary.")
    if ishmed_modalities is None:
        ishmed_modalities = {}
        logging.info("ishmed_modalities is None, initializing to empty dictionary.")
    if cat_modalities is None:
        cat_modalities = {}
        logging.info("cat_modalities is None, initializing to empty dictionary.")
    modality_dict = {}

    if segmented_dataframe is not None:
        for item in wearable_modalities:
            modality_dict[f"wearable_{item}"] = segmented_dataframe.select(
                cs.contains(f"wearable_{item}")
            ).columns
        for key in copra_modalities.keys():
            modality_dict[f"copra_{key}"] = segmented_dataframe.select(
                cs.contains(f"copra_{key}")
            ).columns  # .select(f'^copra_{key}.*$').columns
        for key in ishmed_modalities.keys():
            modality_dict[f"ishmed_{key}"] = segmented_dataframe.select(cs.contains(f"ishmed_{key}")).columns
        for key in cat_modalities.keys():  # cs.contains(f"emb_{key}")
            modality_dict[f"cat_{key}"] = segmented_dataframe.select(cs.contains(f"{key}")).columns
        if len(segmented_dataframe.select(cs.contains("hour")).columns) > 0:
            modality_dict["hour"] = segmented_dataframe.select(cs.contains("hour")).columns
    if static_data is not None:
        modality_dict["static"] = static_data.select(cs.contains("static")).columns
        modality_dict["static_retro_icu"] = static_data.select(cs.contains("static_retro_icu")).columns
        modality_dict["static_retro_pre_intra"] = static_data.select(cs.contains("static_retro_pre_intra")).columns
        # Remove items from 'static' key if they are in 'static_retro_icu' or 'static_retro_pre_intra'
        modality_dict["static"] = [
            item for item in modality_dict["static"]
            if item not in modality_dict.get("static_retro_icu", []) and item not in modality_dict.get(
                "static_retro_pre_intra", [])
        ]
    else:
        logging.info("No static data provided, skipping static modality.")
    total = 0
    accounted_for = []

    for key, value in modality_dict.items():
        total += len(value)
        accounted_for.extend(value)
        logging.info(f"Modality {key}, columns: {len(value)}")
    logging.info(
        f"Length of columns: {total} (note that this includes columns that might not exist for some patients)"
    )

    return modality_dict, accounted_for


def set_positive_horizon(
    outcome_df: pd.DataFrame,
    horizon_hours: int,
    segment_length: timedelta = timedelta(hours=1)
) -> pd.DataFrame:
    """
    Sets the positive label for the last `horizon` segments for each patient in the outcome DataFrame.

    Args:
        outcome_df (pd.DataFrame): DataFrame containing at least 'id', 'datetime', and 'label' columns.
        horizon_hours (int): Number of hours at the end of each group to set as positive (label=1).
        segment_length (timedelta, optional): Length of each segment. Defaults to 1 hour.

    Returns:
        pd.DataFrame: DataFrame with updated 'label' column according to the positive horizon logic.
    """
    # Ensure the DataFrame is sorted by 'id' and 'datetime'    # Ensure the DataFrame is sorted by 'id' and 'datetime'
    outcome_df = outcome_df.sort_values(by=["id", "datetime"])
    horizon = int(horizon_hours / (segment_length.total_seconds() / 3600))  # Convert horizon to number of segments
    logging.info(f"Setting positive horizon of {horizon_hours} with segment length {segment_length} "
                 f"to {horizon} segments.")
    # Define a custom function to apply the horizon logic
    def apply_horizon(group):
        if group["label"].any():
            group.iloc[:-1, group.columns.get_loc("label")] = 0  # Set all except the last label to 0
            group.iloc[-min(len(group), horizon):, group.columns.get_loc("label")] = 1
        return group

    # Apply the custom function to each group
    outcome_df = outcome_df.groupby("id").apply(apply_horizon).reset_index(drop=True)
    return outcome_df


# Example usage
# horizon = 12  # Set the desired horizon
# outcome_pd = segmented_dataframe.select(["id", "datetime", "label"])
# outcome_pd_pandas = outcome_pd.to_pandas()
# # outcome_pd = convert_to_timedelta(outcome_pd)
# outcome_pd_pandas = set_positive_horizon(outcome_pd_pandas, horizon)
# # outcome_pd.to_parquet(os.path.join(save_path, "outc.parquet"))


def prepare_dataset(
        segmented_dataframe,
        base_static,
        root,
        copra_modalities,
        ishmed_modalities,
        cat_modalities,
        version,
        retro_aggregated=None,
        horizons=None,
        segment_length=timedelta(hours=1)
):
    """
    Prepares the dataset for the YAIB format and saves the variables to a .gin file.

    Args:
        segmented_dataframe (pl.DataFrame): The segmented dynamic data.
        base_static (pl.DataFrame): The base static data.
        root (str): The root directory for saving the dataset.
        copra_modalities (dict): Dictionary of COPRA modalities.
        ishmed_modalities (dict): Dictionary of ISHMED modalities.
        cat_modalities (dict): Dictionary of CAT modalities.
        version (str): The version of the dataset.
        retro_aggregated (pl.DataFrame, optional): The retro aggregated data. Defaults to None.
        horizons (list, optional): List of horizons for the outcome labels. Defaults to [3, 6, 9, 12, 18, 24].
        segment_length (timedelta, optional): Length of each segment. Defaults to 1 hour.
    Returns:
        None
    """
    if horizons is None:
        horizons = horizons=[1,2,3,4,5, 6, 7,8, 9,10, 12, 18, 24]
    save_path = os.path.join(root, "Prospective_Preprocessed", "yaib_format", version)
    logging.info(f"Preparing and saving dataset to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Add file handler to log to a text file
    log_file_path = os.path.join(save_path, "dataset_properties.txt")
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Get the root logger and add the file handler
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logging.info("Replacing infinite values")
    # Check and log infinite values, then replace them
    float_cols = segmented_dataframe.select(cs.float()).columns
    if float_cols:
        inf_counts = segmented_dataframe.select([pl.col(col).is_infinite().sum().alias(col) for col in float_cols])
        cols_with_inf = [col for col, count in inf_counts.to_dicts()[0].items() if count > 0]
        if cols_with_inf:
            logging.warning(
                f"Found infinite values in {len(cols_with_inf)} columns: "
                f"{dict(zip(cols_with_inf, [inf_counts[col].item() for col in cols_with_inf]))}")
        else:
            logging.info("No infinite values found in float columns.")
    segmented_dataframe = segmented_dataframe.with_columns(cs.float().replace([float("inf"), -float("inf")], None))
    # Segment amounts statistics
    logging.info(f"Dynamic segmented dataframe shape: {segmented_dataframe.shape}")
    logging.info(f"Amount of columns (features):{len(segmented_dataframe.columns)}")
    logging.info(f"Final dynamic segmented columns: {segmented_dataframe.columns}")
    logging.info(f"Static dataframe shape: {base_static.shape}")
    logging.info(f"Static columns: {base_static.columns}")
    logging.info("Average segments per patient:")
    logging.info(segmented_dataframe.group_by("id").len().describe().select(["statistic", "len"]))

    # Remove columns that are entirely null
    segmented_dataframe = segmented_dataframe[
        [s.name for s in segmented_dataframe if not (s.null_count() == segmented_dataframe.height)]
    ]
    # Create outcome dataframe
    outcome_pd = segmented_dataframe.select(["id", "datetime", "label"])
    outcome_real_time = convert_to_timedelta(outcome_pd, input_column="datetime", output_column="timestep",
                                             segment_length=segment_length)
    outcome_real_time.to_parquet(os.path.join(save_path, "outc_actual_time.parquet"))
    outcome_pd = convert_to_timedelta(outcome_pd, segment_length=segment_length, input_column="datetime", output_column="datetime")
    # outcome_pd_pandas = outcome_pd.to_pandas()
    # outcome_pd.to_parquet(os.path.join(save_path, "outc.parquet"))

    for horizon in horizons:
        outcome_horizon = set_positive_horizon(outcome_pd, horizon, segment_length=segment_length)
        outcome_horizon.to_parquet(os.path.join(save_path, f"outc_{horizon}.parquet"))

    # Create dynamic dataframe
    segmented_dataframe_pd = convert_to_timedelta(segmented_dataframe, input_column="datetime",
                                                  output_column="datetime",
                                                  segment_length=segment_length)
    logging.info(f"Timestep in dynamic dataframe: {segmented_dataframe_pd['datetime'].dtype}")
    segmented_dataframe_pd = segmented_dataframe_pd.drop(columns=["label"])
    segmented_dataframe_pd.to_parquet(os.path.join(save_path, "dyn.parquet"))

    # base_static.join(pl.from_pandas(patient_details[['id','birth_day','birth_month','birth_year']]),on="id", how="left")
    if retro_aggregated is not None:
        static = base_static.join(
            retro_aggregated.with_columns(pl.col("id").cast(pl.Int64)),
            on="id",
            how="left",
        )
    else:
        static = base_static
    static.write_parquet(os.path.join(save_path, "sta.parquet"))

    vars_dict = {
        "GROUP": "id",
        "SEQUENCE": "datetime",
        "LABEL": "label",
        "DYNAMIC": segmented_dataframe.select(pl.exclude(["id", "datetime", "label"])).columns,
        # segmented_dataframe.select(pl.exclude('^emb_note.*$')).select(pl.exclude(["id", "datetime", "label"])).columns,
        "STATIC": static.select(pl.exclude("id")).columns,
    }
    col_len = len(segmented_dataframe.select(pl.exclude(["id", "datetime", "label"])).columns) + len(
        static.select(pl.exclude("id")).columns
    )
    logging.info(f"Length of columns:{col_len}")
    modality_dict, accounted_for = save_modalities_dict(
        segmented_dataframe,
        static_data=static,
        copra_modalities=copra_modalities,
        ishmed_modalities=ishmed_modalities,
        cat_modalities=cat_modalities,
    )
    logging.info(f"Length of columns in modality dict: {len(accounted_for)}, columns not in vars_dict: "
                 f"{(set(vars_dict['DYNAMIC']).union(set(vars_dict['STATIC']))) - set(accounted_for)}")
    logging.info(f"Saved modality dict to {save_path}")
    with open(os.path.join(save_path, "vars.gin"), "w") as handle:
        # pickle.dump(vars_dict, handle)
        handle.write("vars = " + json.dumps(vars_dict, indent=4))
        handle.write("\n")
        handle.write("modality_mapping = " + json.dumps(modality_dict, indent=4))

    logging.info(f"Label values:{segmented_dataframe['label'].value_counts()}")
    logger.removeHandler(file_handler)
    file_handler.close()
