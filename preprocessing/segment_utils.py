import datetime
import logging

import polars as pl

aggregation_sets = {
    "embeddings_basic": [
        (lambda col: pl.col(col).mean(), "mean")
    ],
    "basic": [
        (lambda col: pl.col(col).mean(), "mean"),
        (lambda col: pl.col(col).min(), "min"),
        (lambda col: pl.col(col).max(), "max"),
    ],
    "standard": [
        (lambda col: pl.col(col).mean(), "mean"),
        (lambda col: pl.col(col).min(), "min"),
        (lambda col: pl.col(col).max(), "max"),
        (lambda col: pl.col(col).std(), "std"),
        (lambda col: pl.col(col).count(), "count"),
    ],
    "advanced": [
        (lambda col: pl.col(col).mean(), "mean"),
        (lambda col: pl.col(col).min(), "min"),
        (lambda col: pl.col(col).max(), "max"),
        (lambda col: pl.col(col).std(), "std"),
        (lambda col: pl.col(col).count(), "count"),
        (lambda col: pl.col(col).skew(), "skew"),
        (lambda col: pl.col(col).entropy(), "entropy"),
    ],
    "extended": [
        (lambda col: pl.col(col).mean(), "mean"),
        (lambda col: pl.col(col).min(), "min"),
        (lambda col: pl.col(col).max(), "max"),
        (lambda col: pl.col(col).std(), "std"),
        (lambda col: pl.col(col).count(), "count"),
        (lambda col: pl.col(col).skew(), "skew"),
        (lambda col: pl.col(col).entropy(), "entropy"),
        (lambda col: pl.col(col).median(), "median"),
        (lambda col: pl.col(col).sum(), "sum"),
    ]
}

aggregation_periods = {
    "reduced": [
        datetime.timedelta(hours=24),
        datetime.timedelta(hours=1)
    ],
    "basic": [
        datetime.timedelta(hours=24),
        datetime.timedelta(hours=12),
        datetime.timedelta(hours=1)
        ],
    "extended": [
        datetime.timedelta(hours=24),
        datetime.timedelta(hours=12),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=1),
        datetime.timedelta(minutes=30),
    ]
}

def map_dict(dict, func):
    return {k: func(v) for k, v in dict.items()}


def generate_aggregated_data(
        numerical_dfs: dict[str, pl.DataFrame],
        categorical_dfs: dict[str, pl.DataFrame],
        start_interval: datetime.datetime,
        end_interval: datetime.datetime,
        datetime_col: str,
        patient_key: int,
        id_col: str = "id",
        prefix: str = "static_retro",
        agg_set: str = "advanced",
):
    """
    Generated a flattened dataframe with all the retrospective data for a patient. Normally used for normal ward cohorts
    where we have a procedure date (i.e., time they spend in the ICU). Can be used for ICU for preoperative data.

    :param numerical_dfs: Filtered list of numerical dataframes for single patient id.
    :param categorical_dfs: Filtered list of numerical dataframes for single patient id.
    :param start_interval: Start of the interval to consider.
    :param end_interval: End of the interval to consider.
    :param datetime_col: Name of the datetime column.
    :return: aggregated: Wide dataframe with aggregated static data.
    """
    def filter_between(x: pl.DataFrame) -> pl.DataFrame:
        return x.filter(pl.col(datetime_col).is_between(start_interval, end_interval))
    # dfs_emb_interval = *map(
    #     lambda x: x.filter(pl.col(datetime_col).is_between(start_interval, end_interval)),
    #     dfs_emb),
    numerical_dfs_interval = map_dict(numerical_dfs, filter_between)
    # dfs_emb_interval = map_dict(dfs_emb, filter_between)
    # for key, val in categorical_dfs.items():
    #     logging.info(f"key")

    agg_val = "value"
    numerical_dfs_interval = numerical_dfs_interval.values()
    selected_agg_funcs = aggregation_sets[agg_set]

    # Apply aggregations
    agg_exprs = [func(agg_val).alias(f"{agg_val}_{suffix}")
                 for func, suffix in selected_agg_funcs]

    dfs_num_seg = (
        *map(
            lambda x: x.group_by("type").agg(*agg_exprs),
            numerical_dfs_interval,
        ),
    )

    dfs_num_seg = map(lambda x: x.with_columns(id=patient_key), dfs_num_seg)
    # dfs_emb_interval = map_dict(dfs_emb_interval, add_id)

    dfs_num_seg = (
        *map(
            lambda x: x.pivot(
                values=[item for item in x.columns if item not in ["type", datetime_col]],
                columns="type",
                aggregate_function="median",
                index="id",
            ),
            dfs_num_seg,
        ),
    )

    aggregated = pl.concat(dfs_num_seg, how="align")

    aggregated = aggregated.select(pl.all()).rename(
        lambda x: prefix + x if x not in [id_col, datetime_col] else x
    )

    return aggregated


def aggregate_pre_prediction_data(
        admission_dict: dict[int, datetime.datetime],
        datetime_col: str,
        categorical_dfs: dict[str, pl.DataFrame],
        numerical_dfs: dict[str, pl.DataFrame],
        id_col: str,
        patient_key: int,
        procedure_dict: dict[int, datetime.datetime],
        start_date: datetime.datetime,
        cohort: str = "normal_ward",
) -> pl.DataFrame:
    """
    Aggregate pre-prediction (static retrospective) data for a patient.

    :param admission_dict: Dictionary mapping patient IDs to admission datetimes.
    :param datetime_col: Name of the datetime column.
    :param categorical_dfs: Dictionary of categorical/embedding dataframes for the patient.
    :param numerical_dfs: Dictionary of numerical dataframes for the patient.
    :param id_col: Name of the ID column.
    :param patient_key: Patient ID to process.
    :param procedure_dict: Dictionary mapping patient IDs to procedure datetimes.
    :param start_date: Start date for the patient.
    :param cohort: Cohort type, e.g., "normal_ward".
    :return: Aggregated static retrospective dataframe for the patient.
    """
    if cohort == "normal_ward":
        logging.info(f"Adding ICU data for patient {patient_key} ")
        # Aggregate data for patient between procedure and defines start date
        icu_data = generate_aggregated_data(
            numerical_dfs,
            categorical_dfs,
            start_interval=procedure_dict[patient_key],
            end_interval=start_date,
            datetime_col=datetime_col,
            patient_key=patient_key,
            id_col=id_col,
            prefix="static_retro_icu_",
        )
    else:
        icu_data = pl.DataFrame()
    # pre- and intra-operative data aggregated/flattened
    if admission_dict.get(patient_key) is not None:
        logging.info(f"Adding pre-intra-operative data for patient {patient_key} ")
        pre_intra_operative_data = generate_aggregated_data(
            numerical_dfs,
            categorical_dfs,
            admission_dict[patient_key],
            procedure_dict[patient_key],
            datetime_col,
            patient_key,
            id_col,
            prefix="static_retro_pre_intra_",
        )
        if len(icu_data) > 0:
            static_retrospective = pl.concat([pre_intra_operative_data, icu_data], how="align")
        else:
            static_retrospective = pre_intra_operative_data
    else:
        static_retrospective = icu_data
    return static_retrospective


def add_missing_indicators_and_impute(df: pl.DataFrame, column_pattern: str, id_col: str, zero_impute=False) -> pl.DataFrame:
    """
    Add missing indicators for grouped columns and perform forward fill imputation.

    Args:
        df: DataFrame to process
        column_pattern: Pattern to match columns (e.g., "emb", "wearable")
        id_col: ID column for grouping during imputation
        zero_impute: Whether to zero impute after forward filling.
    Returns:
        DataFrame with missing indicators added and imputation performed
    """
    # Get columns matching the pattern (excluding any existing missing indicator columns)
    matching_cols = [col for col in df.columns
                     if column_pattern in col and not col.endswith("_missing")]

    if not matching_cols:
        return df

    def extract_base_name(col_name: str) -> str:
        """Extract base name by removing numeric suffixes and time/aggregation patterns."""
        # First, handle aggregation patterns by splitting on "_agg_"
        if "_agg_" in col_name:
            base_part = col_name.split("_agg_")[0]
        else:
            base_part = col_name

        parts = base_part.split("_")

        # Remove suffixes working backwards
        while len(parts) > 1:
            last_part = parts[-1]
            # Remove time patterns (1h, 6h, etc.)
            if last_part.endswith('h') and last_part[:-1].isdigit():
                parts = parts[:-1]
            # Remove pure numbers (dimension indices)
            elif last_part.isdigit():
                parts = parts[:-1]
            else:
                break

        return "_".join(parts)

    # Get unique base names
    base_names = {extract_base_name(col) for col in matching_cols}

    # Create all missing indicators at once using expressions
    missing_indicators = [
        pl.any_horizontal([pl.col(col).is_null() for col in matching_cols if col.startswith(base_name)])
        .alias(f"{base_name}_missing")
        for base_name in base_names
        if any(col.startswith(base_name) for col in matching_cols)
    ]

    # Create all imputation expressions at once
    if zero_impute:
        imputation_exprs = [
            pl.col(col).forward_fill().over(id_col).fill_null(0)
            for col in matching_cols
        ]
    else:
        imputation_exprs = [
            pl.col(col).forward_fill().over(id_col)
            for col in matching_cols
        ]

    # Apply all transformations in a single with_columns call
    return df.with_columns(missing_indicators + imputation_exprs)


def calculate_endpoint(
        end_dict,
        endpoint_dict,
        patient_key,
        skip_list,
        stay_start,
        surgery_dict=None,
        culling_time=datetime.timedelta(days=20),
        cohort_type: str = "normal_ward"
):
    """
    Calculates the end of the stay for a patient (either the endpoint or the end of the stay).
    :param surgery_dict:
    :param cohort_type:
    :param culling_time: Cut off time for the stay (maximum stay length)
    :param stay_start:
    :param end_dict:
    :param endpoint_dict:
    :param patient_key:
    :param skip_list:
    :return: stay_start, stay_end
    """

    if patient_key in endpoint_dict:
        if cohort_type == "ICU":
            if endpoint_dict[patient_key] < end_dict[patient_key]:
                stay_end = endpoint_dict[patient_key]
            else:
                logging.warning(
                    f"Endpoint for patient {patient_key} is after the end of the stay, using end_dict instead: "
                    f"{end_dict[patient_key]}"
                )
                stay_end = end_dict[patient_key]
        else:
            stay_end = endpoint_dict[patient_key]
    elif patient_key in end_dict:
        # Does not experience endpoint
        stay_end = end_dict[patient_key]
    else:
        logging.warning(f"Patient {patient_key} not found in end_dict, skipping")
        skip_list.append(patient_key)
        return None, None

    if stay_end - stay_start < datetime.timedelta(0):
        logging.warning(f"Stay end before stay start for patient {patient_key}, skipping")
        return None, None

    if surgery_dict is not None and patient_key in surgery_dict.keys():
        # We cut off after culling time after surgery, except for patients that experience the endpoint
        if stay_end - surgery_dict[patient_key] > culling_time and endpoint_dict.get(patient_key) is None:
            logging.info(f"Stay end: {stay_end}, surgery time: {surgery_dict[patient_key]}")
            # We calculate from the end of the surgery
            stay_end = surgery_dict[patient_key] + culling_time
    elif stay_end - stay_start > culling_time and endpoint_dict.get(patient_key) is None:
        # We calculate from the start of the stay
        stay_end = stay_start + culling_time
    return stay_start, stay_end


def drop_sparse_per_id(df: pl.DataFrame, id_col: str = "id", threshold: float = 0.95) -> pl.DataFrame:
    """
    Drop columns where the ratio of IDs with only null values exceeds the threshold.

    Args:
        df: The dataframe to analyze
        id_col: Column name containing the IDs (default: "id")
        threshold: Threshold ratio above which columns are dropped (default: 0.95)

    Returns:
        DataFrame with sparse columns removed
    """
    data_cols = [col for col in df.columns if col != id_col]
    total_unique_ids = df.select(id_col).n_unique()

    # Find columns to drop
    columns_to_drop = (
        df.group_by(id_col)
        .agg([
            pl.col(col).is_null().all().alias(f"{col}_all_null")
            for col in data_cols
        ])
        .select([
            pl.col(f"{col}_all_null").sum().alias(col)
            for col in data_cols
        ])
        .unpivot(
            variable_name="column",
            value_name="ids_with_all_nulls"
        )
        .with_columns(
            (pl.col("ids_with_all_nulls") / total_unique_ids).alias("null_ratio")
        )
        .filter(pl.col("null_ratio") > threshold)
        .get_column("column")
        .to_list()
    )

    # Drop the identified columns
    columns_to_keep = [col for col in df.columns if col not in columns_to_drop]

    logging.info(f"Dropping {len(columns_to_drop)} columns with null ratio > {threshold}")
    return df.select(columns_to_keep)


def rename_and_select_wearable(
    df_activity_segment: pl.DataFrame,
    version: int | str = 2
) -> pl.DataFrame:
    """
    Selects and optionally renames columns from a wearable activity or core dataframe segment.

    Args:
        df_activity_segment (pl.DataFrame): The input Polars DataFrame containing wearable activity or core data.
        version (int | str, optional): The version of the wearable data. Use "core" for core data,
            1 for activity v1, or 2 for activity v2. Defaults to 2.

    Returns:
        pl.DataFrame: The processed DataFrame with selected and possibly renamed columns.
    """
    if version == "core":
        columns_to_select = ["core_temp", "skin_temp", "quality"]
        columns_to_select = ["wearable_core_" + x for x in columns_to_select]
        # logging.info(df_activity_segment.columns)
        df_activity_segment = df_activity_segment.select(columns_to_select + ["datetime"])
        # df_activity_segment = df_activity_segment.unpivot(
        #     index="datetime",
        #     on=columns_to_select,
        #     variable_name="type", value_name="value")
        # df_activity_segment = df_activity_segment.cast({"value": pl.Float64})
        return df_activity_segment
    else:
        columns_to_select = [
            "activity_type",
            "energy_ex",
            "heart_rate_filtered",
            "heart_rate_filtered_q",
            "respiration_rate_filtered",
            "respiration_rate_filtered_q",
            "speed",
            "skin_temperature",
            "wearing_status",
        ]
        # Features unique to v2
        v2_cols = [
            "battery_level",
            "cadence_raw",
            "spo2",
            "vo2max_q",
            "core_body_temperature",
            "energy_ex_raw_intermittent",
            "energy_ex_raw_q",
        ]
        columns_to_select = ["wearable_activity_" + x for x in columns_to_select]
        if version == 1:
            if "wearable_activity_spo2" in df_activity_segment.columns:
                # Also contains v2 data
                # Dictionary specifying columns to add together
                columns_to_add = {
                    "activity": "activity_type",
                    "hrm_filtered": "heart_rate_filtered",
                    "hrm_q": "heart_rate_filtered_q",
                    "resp_filtered": "respiration_rate_filtered",
                    "resp_q": "respiration_rate_filtered_q",
                    "resp_raw": "respiration_rate_raw",
                    "step_count": "last_step_count",
                    "wearing": "wearing_status",
                    "bracelet_temp": "skin_temperature",
                }
                # logging.info(df_activity_segment.columns)
                # Add the columns together and drop the old columns
                for col1, col2 in columns_to_add.items():
                    df_activity_segment = df_activity_segment.with_columns(
                        pl.coalesce(
                            [pl.col(f"wearable_activity_{col1}"), pl.col(f"wearable_activity_{col2}")]
                        ).alias(f"wearable_activity_{col2}")
                    )
                    df_activity_segment = df_activity_segment.drop([f"wearable_activity_{col1}"])
                v2_cols = ["wearable_activity_" + x for x in v2_cols]
                columns_to_select.extend(v2_cols)
            else:
                rename_dict = {
                    "activity": "activity_type",
                    "hrm_filtered": "heart_rate_filtered",
                    "hrm_q": "heart_rate_filtered_q",
                    "resp_filtered": "respiration_rate_filtered",
                    "resp_q": "respiration_rate_filtered_q",
                    "resp_raw": "respiration_rate_raw",
                    "step_count": "last_step_count",
                    "wearing": "wearing_status",
                    "bracelet_temp": "skin_temperature",
                }
                rename_dict = {
                    "wearable_activity_" + k: "wearable_activity_" + v for k, v in rename_dict.items()
                }
                df_activity_segment = df_activity_segment.rename(rename_dict)
        else:

            v2_cols = ["wearable_activity_" + x for x in v2_cols]
            columns_to_select.extend(v2_cols)
        df_activity_segment = df_activity_segment.select(columns_to_select + ["datetime"])
        # df_activity_segment = df_activity_segment.unpivot(
        #     index="datetime",
        #     on=columns_to_select,
        #     variable_name="type", value_name="value")
        # df_activity_segment = df_activity_segment.cast({"value": pl.Float64})
        return df_activity_segment



