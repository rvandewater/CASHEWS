# import seaborn as sn
import datetime
import logging
from datetime import timedelta

import polars as pl
from joblib import Parallel, delayed, parallel_config
from polars import DataFrame

from .dataloader.dataloader import Dataloader
from .post_processing.filter_cohort import filter_columns_by_null_percentage
from .preprocess_wearable.feature_extraction.ppg_feature_selection import (
    select_wearable_features,
)
from .segment_utils import map_dict, aggregate_pre_prediction_data, add_missing_indicators_and_impute, \
    calculate_endpoint, aggregation_sets, aggregation_periods, drop_sparse_per_id, rename_and_select_wearable
import math

def segment_modalities(
        num_df_dict: dict[str, pl.DataFrame],
        cat_df_dict: dict[str, pl.DataFrame],
        dataloader: Dataloader,
        start_dict: dict[int, datetime.datetime],
        end_dict: dict[int, datetime.datetime],
        endpoint_dict: dict[int, datetime.datetime],
        procedure_dict: dict[int, datetime.datetime],
        admission_dict: dict[int, datetime.datetime],
        segment_length: datetime.timedelta = datetime.timedelta(hours=1),
        prediction_horizon: datetime.timedelta = datetime.timedelta(hours=6),
        datetime_col: str = "datetime",
        id_col: str = "id",
        max_stay_length: datetime.timedelta = datetime.timedelta(days=12), #datetime.timedelta(days=25),
        min_stay_length: datetime.timedelta = datetime.timedelta(hours=0),
        debug: bool | list | int = False,
        jobs: int = -1,
        notebook: bool = False,
        wearable_cohort: bool = True,
        filter_wearable: bool = True,
        use_activity: bool = True,
        use_ppg_features: bool = True,
        reduce_ppg_feature_set: bool = True,
        use_ppg_embeddings: bool = True,
        normalise_ppg_features: bool = True,
        exclude_ids: list[int] = None,
        cohort_type: str = "normal_ward",
        add_time_of_day: bool = True,
        impute_embeddings: bool = True,
        drop_sparse_per_id_ratio: float = 0.97,
        retro_threshold = 95
) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """
    Segments the dataframes in df_list based on the start and end times in start_dict and end_dict.
    Segments are aggregated, the data is upsampled to have a uniform time interval, and labels are added based
    on the endpoint_dict.

    :param drop_sparse_per_id_ratio: float, ratio of sparse columns per patient to drop
    :param impute_embeddings: Whether to impute embeddings for the patient.
    :param add_time_of_day: if we should add the hour of the day as a feature to the aggregated data (can result in unintended correlations).
    :param cohort_type: Name of the cohort type: "normal_ward", "ICU", "ICU_and_normal_ward".
    :param use_ppg_embeddings: Use ppg embeddings in dataset.
    :param exclude_ids: Exclude these ids by default.
    :param num_df_dict: numerical modalities as list
    :param cat_df_dict: categorical modalities as list
    :param dataloader: dataloader for loading individual wearable data
    :param start_dict: start times for each patient
    :param end_dict: end times for each patient
    :param endpoint_dict: label times for each patient
    :param segment_length: length of each segment
    :param prediction_horizon: how long before the endpoint happens do we want to label positive (prediction horizon)
    :param datetime_col: datetime column in the dataframes
    :param id_col: patient id column in the dataframes
    :param max_stay_length: maximum stay length for a patient
    :param min_stay_length: minimum stay length for a patient
    :param debug: only process these patients
    :param jobs: number of parallel jobs
    :param notebook: if running in a notebook
    :param wearable_cohort: if we should add wearable data to the cohort.
    :param filter_wearable: if we should filter the wearable cohort to only include patients with at least one wearable modality
    :param use_activity: if we should use activity data
    :param use_ppg_features: if we should use ppg features
    :param normalise_ppg_features: if we should normalise ppg features
    :param reduce_ppg_feature_set: if we should reduce the ppg feature set
    :param procedure_dict: procedure times for each patient
    :param admission_dict: admission times for each patient
    :param retro_threshold: threshold for the ratio of null values in the retrospective data to keep a column
    :return:
    tuple of aggregated dataframes for the (dynamic) segments and static retrospective data if succeeded
    """

    patient_list = []
    skip_list = []
    logging.getLogger().setLevel(logging.INFO)

    if debug:
        if type(debug) is list:
            start_dict = {x: start_dict[x] for x in debug if x in start_dict}
        elif type(debug) is int:
            start_dict = {x: start_dict[x] for x in [debug] if x in start_dict}
        else:
            start_dict = {x: start_dict[x] for x in list(start_dict.keys())[:5]}
        logging.info(f"Debug mode, only processing {len(start_dict)} patients: {start_dict.keys()}")
        logging.getLogger().setLevel(logging.DEBUG)

    if exclude_ids is not None:
        logging.info(f"Excluding ids: {exclude_ids}")
        start_dict = {k: v for k, v in start_dict.items() if k not in exclude_ids}

    if wearable_cohort:
        # We create a cohort that has at least one modality of wearable data available.
        core_ids = dataloader.get_ids("core")
        activity_ids = dataloader.get_ids("activity")
        ppg_ids = []
        if use_ppg_features:
            ppg_ids.extend(dataloader.get_ids("ppg_features"))
        if use_ppg_embeddings:
            ppg_ids.extend(dataloader.get_ids("ppg_embeddings"))
        patient_dict = {}
        if filter_wearable:
            for patient_key, date in start_dict.items():
                if (
                        patient_key in core_ids or patient_key in activity_ids or patient_key in ppg_ids
                ):
                    patient_dict[patient_key] = date
        start_dict = patient_dict
    else:
        logging.info("Not using wearable data, disabling any ")
        use_activity = False
        use_ppg_features = False
    # Drop patients who spent less than the minimum stay time
    for key, value in start_dict.items():
        if key in end_dict:
            if end_dict[key] - value < min_stay_length:
                logging.info(
                    f"Patient {key} spent {end_dict[key] - value} in the supplied time interval"
                    f", shorter than {min_stay_length}, skipping (negative values could be due "
                    f"to complications in the ICU)"
                )
                skip_list.append(key)
    start_dict = {k: v for k, v in start_dict.items() if k not in skip_list}

    if len(start_dict) == 0:
        logging.info("No patients to process, exiting")
        return None
    logging.info(f"Skipped patients: {skip_list}, amount: {len(skip_list)}")
    logging.info(f"Number of patients to process: {len(start_dict)}")

    # We iterate over the start_dict
    for item in num_df_dict:
        num_df_dict[item] = num_df_dict[item].with_columns(pl.col(datetime_col).cast(pl.Datetime))

    for item in cat_df_dict:
        cat_df_dict[item] = cat_df_dict[item].with_columns(pl.col(datetime_col).cast(pl.Datetime))

    def process_patient(
            subject_key: int, start_date: datetime.datetime
    ) -> tuple[int, DataFrame | None, DataFrame] | None:
        """
        Internal function that processes a single patient in parallel by segmenting their data based on the provided
        start date.

        :param subject_key: The patient ID to process.
        :param start_date: Start date for the patient.
        :return: Tuple of segmented dataframe and static retrospective dataframe, or None if processing fails.
        """
        output_tuple = segment_single_patient(
            cat_df_dict=cat_df_dict,
            dataloader=dataloader,
            datetime_col=datetime_col,
            detection_interval=prediction_horizon,
            end_dict=end_dict,
            endpoint_dict=endpoint_dict,
            procedure_dict=procedure_dict,
            num_df_dict=num_df_dict,
            use_ppg_embeddings=use_ppg_embeddings if wearable_cohort else False,
            use_activity=use_activity if wearable_cohort else False,
            use_ppg_features=use_ppg_features if wearable_cohort else False,
            segment_length=segment_length,
            skip_list=skip_list,
            patient_key=subject_key,
            start_date=start_date,
            id_col=id_col,
            culling_time=max_stay_length,
            admission_dict=admission_dict,
            cohort=cohort_type,
            impute_embeddings=impute_embeddings,
        )
        if output_tuple is None:
            return None
        segment_list, retrospective_df = output_tuple
        if segment_list is None and retrospective_df is None:
            return None
        return subject_key, segment_list, retrospective_df

    # We iterate over the start_dict
    backend = "threading" if notebook else "loky"
    with parallel_config(backend=backend, n_jobs=jobs):
        results = Parallel()(
            delayed(process_patient)(patient_key, start_date)
            for (patient_key, start_date) in start_dict.items()
        )

    # Check if we can concatenate the results
    if results is None or len(results) == 0:
        logging.warning("No patients processed, exiting")
        return None
    results_len = len(results)
    results = [x for x in results if x is not None]
    logging.info(f"Number of patients processed: {len(results)} out of {results_len}")
    static_patient_list = []

    for key, segment_df, retrospective in results:
        if segment_df is not None:
            patient_list.append(segment_df)
            if segment_df.shape[0] == 0:
                skip_list.append(key)
                logging.warning(f"Skipping {key} as it has no segments")
            else:
                logging.info(f"Patient {key} processed with {len(segment_df)} segments")
        else:
            logging.warning(f"Patient {key} has no segment data, skipping")
            skip_list.append(key)
        if retrospective is not None:
            static_patient_list.append(retrospective)
        else:
            skip_list.append(key)
    static_patient_list = [x for x in static_patient_list if x is not None]

    if len(patient_list) == 0:
        logging.warning("No patients processed, exiting")
        return None

    retro_aggregated = pl.concat(static_patient_list, how="diagonal_relaxed")

    logging.info(f"Filtering columns in retrospectively aggregated data with more than {retro_threshold} null values")
    retro_aggregated = filter_columns_by_null_percentage(retro_aggregated, retro_threshold)
    # Concatenate the extracted intervals for each patient
    aggregated = pl.concat(patient_list, how="diagonal_relaxed")

    # reduce wearable features by correlation and mutual information
    if use_ppg_features and any(col.startswith("wearable_ppgfeature") for col in aggregated.columns):
        if reduce_ppg_feature_set:
            logging.info(f"Reducing wearable features. Wearable features length before: {len(aggregated.columns)}")
            aggregated = select_wearable_features(
                aggregated,
                feature_base_folder=dataloader.ppg_features_folder,
                corr_thresh=0.8,
                mi_thresh=1e-6,
                verbose=True,
            )
            logging.info(f"Length after: {len(aggregated.columns)}")
        # normalise wearable features
        if normalise_ppg_features:
            logging.info("Normalising ppg features")
            aggregated = aggregated.with_columns(
                (pl.col("^wearable_ppgfeature.*$") - pl.col("^wearable_ppgfeature.*$").mean())
                / pl.col("^wearable_ppgfeature.*$").std()
            )
    # logging.info(aggregated)
    logging.info(f"Number of patients skipped: {len(skip_list)}: {skip_list}")
    # Sort columns
    aggregated = aggregated.select(
        [pl.col(id_col), pl.col(datetime_col), pl.all().exclude([id_col, datetime_col])]
    )
    aggregated = aggregated.select(
        [col for col in aggregated.columns if aggregated[col].null_count() < len(aggregated)])

    # Add time of day
    if add_time_of_day:
        logging.info("Adding hour of day to aggregated data. Note that this might introduce bias for clinical events.")
        aggregated = aggregated.with_columns(pl.col(datetime_col).dt.hour().alias("hour"))
        aggregated = aggregated.with_columns(
            (pl.col("hour") * 2 * math.pi / 24).sin().alias("hour_sin")
        )
    # aggregated = aggregated.with_columns(cs.contains("emb").fill_null(0))
    # aggregated = aggregated.with_columns(cs.contains("wearable").fill_null(0))
    if drop_sparse_per_id_ratio < 1.0:
        aggregated = drop_sparse_per_id(aggregated, id_col=id_col, threshold=drop_sparse_per_id_ratio)
    return aggregated, retro_aggregated


def segment_single_patient(
        cat_df_dict: dict[str, pl.DataFrame],
        num_df_dict: dict[str, pl.DataFrame],
        dataloader: Dataloader,
        datetime_col: str,
        detection_interval: datetime.timedelta,
        end_dict: dict[int, datetime.datetime],
        endpoint_dict: dict[int, datetime.datetime],
        procedure_dict: dict[int, datetime.datetime],
        admission_dict: dict[int, datetime.datetime] | None,
        use_activity: bool,
        use_ppg_features: bool,
        use_ppg_embeddings: bool,
        segment_length: datetime.timedelta,
        skip_list: list[int],
        patient_key: int,
        start_date: datetime.datetime,
        id_col: str,
        culling_time: datetime.timedelta,
        cohort: str = "normal_ward",
        impute_embeddings: bool = True
) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """
    Segment the data for a single patient.

    :param impute_embeddings: Whether to impute embeddings for the patient.
    :param use_ppg_embeddings: precomputed PPG embeddings are used in the dataset.
    :param cohort: Cohort type, e.g., "normal_ward", "ICU", "ICU_and_normal_ward".
    :param use_activity: Whether activity data is available.
    :param cat_df_dict: Dictionary of categorical modalities, includes embeddings.
    :param dataloader: Dataloader for wearable data.
    :param datetime_col: Name of the datetime column.
    :param detection_interval: Interval before endpoint to consider as positive (horizon).
    :param end_dict: Dictionary of end times for the patients.
    :param endpoint_dict: Dictionary of endpoint times for the patients.
    :param procedure_dict: Dictionary of procedure times for the patients.
    :param admission_dict: Dictionary of admission times for the patients.
    :param num_df_dict: Dictionary of numerical modalities.
    :param use_ppg_features: Whether to use features extracted from PPG data.
    :param use_ppg_features: Whether to use embeddings extracted from PPG data.
    :param segment_length: Length of the prediction segments.
    :param skip_list: List of patients to skip.
    :param patient_key: The patient ID to process.
    :param start_date: Start date for the patient.
    :param id_col: Name of the ID column.
    :param culling_time: Cut off time for the stay (maximum stay length).
    :return: Tuple of segmented dataframe and static retrospective dataframe, or None if processing fails.
    """
    logging.info(f"Processing patient {patient_key}")
    # Check if we have a corresponding end point
    stay_start, stay_end = calculate_endpoint(
        end_dict, endpoint_dict, patient_key, skip_list, start_date, procedure_dict, culling_time=culling_time,
        cohort_type=cohort
    )
    if stay_start is None:
        return None, None
    segment_start = stay_start
    # Filter out individual patient from each dataframe. Note that the notation converts the map into a list
    def filter_id(x): return x.filter(pl.col(id_col) == patient_key)
    numerical_dfs = map_dict(num_df_dict, filter_id)
    categorical_dfs = map_dict(cat_df_dict, filter_id)

    # Add static retrospective data (pre-operative data)
    if procedure_dict.get(patient_key) is not None:
        static_retrospective = aggregate_pre_prediction_data(
            admission_dict=admission_dict,
            categorical_dfs=categorical_dfs,
            numerical_dfs=numerical_dfs,
            datetime_col=datetime_col,
            id_col=id_col,
            patient_key=patient_key,
            procedure_dict=procedure_dict,
            start_date=start_date,
            cohort=cohort
        )
    else:
        static_retrospective = None

    # Filter out data before start and after end
    def filter_between(x): return x.filter(pl.col(datetime_col).is_between(segment_start, stay_end))
    numerical_dfs = map_dict(numerical_dfs, filter_between)
    categorical_dfs = map_dict(categorical_dfs, filter_between)

    logging.debug(f"Filtered data for patient id {patient_key}")
    for df_name, df in (numerical_dfs | categorical_dfs).items():
        logging.debug(f"Length of df {df_name}: {len(df)}")

    numerical_dfs = numerical_dfs.values()
    categorical_dfs = categorical_dfs.values()

    # Filter out duplicate entries
    numerical_dfs = (*map(lambda x: x.unique(subset=[datetime_col]), numerical_dfs),)
    categorical_dfs = (*map(lambda x: x.unique(subset=[datetime_col]), categorical_dfs),)

    numerical_dfs = [i for i in numerical_dfs if len(i) > 0]
    categorical_dfs = [i for i in categorical_dfs if len(i) > 0]
    segment_counter = 0
    if all(len(x) == 0 for x in numerical_dfs) and all(len(x) == 0 for x in categorical_dfs):
        logging.info(f"Length of data for id {patient_key}, stay start {segment_start}, stay end {stay_end} is 0, skipping")
        return
    num_dfs_list = []
    cat_dfs_list = []

    # Create segments until we reach the end of the stay
    while segment_start < stay_end:
        logging.debug(
            f"Segment: {segment_counter}, Segment start: {segment_start}, Segment end: {segment_start + segment_length}"
        )
        # Filter out data before start and after end
        dfs_num_segments = (
            *map(
                lambda x: x.filter(
                    pl.col(datetime_col).is_between(segment_start, segment_start + segment_length)
                ),
                numerical_dfs,
            ),
        )
        dfs_cat_segments = (
            *map(
                lambda x: x.filter(
                    pl.col(datetime_col).is_between(segment_start, segment_start + segment_length)
                ),
                categorical_dfs,
            ),
        )
        # Aggregate data within the segment
        agg_val = "value"
        selected_agg_funcs = aggregation_sets["advanced"]

        # Apply aggregations
        agg_exprs = [func(agg_val).alias(f"{agg_val}_{suffix}")
                     for func, suffix in selected_agg_funcs]

        dfs_num_segments = (
            *map(
                lambda x: x.group_by("type").agg(*agg_exprs),
                dfs_num_segments,
            ),
        )
        dfs_cat_segments = (*map(lambda x: x.unique(), dfs_cat_segments),)
        # Combine dataframes within the segment
        if len(dfs_num_segments) > 0:
            dfs_num_conc = pl.concat(dfs_num_segments, how="vertical")
            dfs_num_conc = dfs_num_conc.with_columns(datetime=segment_start)
            num_dfs_list.append(dfs_num_conc)
        if len(dfs_cat_segments) > 0:
            # logging.info(dfs_cat_segments)
            # Average the categorical dimensions because there might be multiple embeddings per segment
            cat_segments_agg = (*map(lambda x: x.select(pl.exclude(["datetime", "id"])).mean(), dfs_cat_segments),)
            cat_segments_agg = [*map(lambda x: x.with_columns(datetime=segment_start), cat_segments_agg), ]
            cat_dfs_list.append(list(cat_segments_agg))

        # upsample ppg
        segment_start = segment_start + segment_length
        segment_counter += 1
    logging.debug(f"Num dfs: {len(num_dfs_list)}")
    logging.debug(num_dfs_list)
    # Pivot the dataframes such that the values become the columns
    if len(num_dfs_list) > 0:
        num_dfs_list = (
            *map(
                lambda x: x.pivot(
                    values=[item for item in x.columns if item not in ["type", datetime_col]],
                    columns="type",
                    index=datetime_col,
                    # aggregate_function="median",
                ),
                num_dfs_list,
            ),
        )
        segmented_df = pl.concat(num_dfs_list, how="diagonal")
        logging.debug("Segment list")
        logging.debug(segmented_df)
    else:
        segmented_df = pl.DataFrame(pl.datetime_range(start=stay_start, end=stay_end, interval=segment_length,
                                                      eager=True).alias(datetime_col))
    if len(cat_dfs_list) > 0:
        # Add embeddings if there are more than 1 per segment
        concat_cat_list = [pl.concat(i, how="align") if len(i) > 1 else i[0] for i in cat_dfs_list]
        # Filter out empty dfs
        concat_cat_list = [item for item in concat_cat_list if len(item) > 0]
        # logging.info(f"concat_cat_list {patient_key}: {concat_cat_list}")
        if len(concat_cat_list) > 0:
            # Diagonally concatenate the categoricals to align datetimes
            appended_cat_df = pl.concat(concat_cat_list, how="diagonal")
            # Replace 0 with None for imputing later
            appended_cat_df = appended_cat_df.select(pl.all().replace(0, None))
            logging.debug("Appended embedding df")
            logging.debug(appended_cat_df)
            appended_cat_df = appended_cat_df.select(pl.all()).rename(
                lambda x: "emb_" + x if x not in [id_col, datetime_col] else x
            )
            # Add the embeddings to the segmented dataframe, aligning by datetime
            segmented_df = pl.concat([segmented_df, appended_cat_df], how="align")
            segmented_df = segmented_df.sort(by=datetime_col)
            logging.debug(segmented_df)
        else:
            logging.info(f"No categorical values found for patient {patient_key} in period, skipping categorical values")
    # Concatenate the segments
    # Upsample the data to have a uniform time interval (we might have segments with completely missing data)
    old_length = len(segmented_df)
    segmented_df = segmented_df.upsample(time_column=datetime_col, every=segment_length, maintain_order=True)
    logging.info(f"Segmented dataframe for patient {patient_key} has {old_length} rows, upsampled to {len(segmented_df)}")

    # Extract wearable features
    if use_activity or use_ppg_features or use_ppg_embeddings:
        segmented_df, activity_version = wearable_feature_extraction(
            segmented_df,
            use_activity_core=use_activity,
            datetime_col=datetime_col,
            use_ppg_features=use_ppg_features,
            use_ppg_embeddings=use_ppg_embeddings,
            segment_length=segment_length,
            start_date=start_date,
            dataloader=dataloader,
            patient_key=patient_key,
            stay_end=stay_end,
        )
    # Fill in missing segments
    segmented_df = fill_in_segments(datetime_col, segment_length, segmented_df, stay_end, stay_start)
    # Add the patient id to the data
    segmented_df = segmented_df.with_columns(id=patient_key)
    segmented_df = segmented_df.sort(by=[id_col, datetime_col])
    # Forward fill wearable and embedding data
    if impute_embeddings:
        logging.info("Imputing missing values in embeddings and wearable data")
        segmented_df = add_missing_indicators_and_impute(df=segmented_df, column_pattern="wearable", id_col=id_col)
        segmented_df = add_missing_indicators_and_impute(df=segmented_df, column_pattern="emb", id_col=id_col)
    # Add label to the data
    if endpoint_dict.get(patient_key) is not None:
        segmented_df = segmented_df.with_columns(
            pl.when(pl.col("datetime") + detection_interval < endpoint_dict[patient_key])
            .then(0)
            .otherwise(1)
            .alias("label")
        )
        logging.info(f"Label added to id {patient_key}")
    else:
        segmented_df = segmented_df.with_columns(label=0)
    if use_activity:
        # Add metadata feature on which wearable used
        segmented_df.with_columns(wearable_activity_version=activity_version)

    return segmented_df, static_retrospective


def fill_in_segments(datetime_col, segment_length, segmented_df, stay_end, stay_start):
    segmented_dummy = pl.datetime_range(start=stay_start, end=stay_end, interval=segment_length, eager=True).alias(
        datetime_col
    )
    segmented_dummy = pl.DataFrame().with_columns(datetime=segmented_dummy)
    segmented_df = segmented_df.join(segmented_dummy, on=[datetime_col], how="outer", coalesce=True)
    return segmented_df


def wearable_feature_extraction(
        segmented_df: pl.DataFrame,
        use_activity_core: bool,
        use_ppg_features: bool,
        use_ppg_embeddings: bool,
        datetime_col: str,
        segment_length: datetime.timedelta,
        start_date: datetime.datetime,
        dataloader: Dataloader,
        patient_key: int,
        stay_end: datetime.datetime,
        create_wearable_groups: bool = True
) -> tuple[pl.DataFrame, str]:
    """
    Extracts wearable features and adds them to the segmented dataframe.

    :param use_ppg_embeddings:
    :param create_wearable_groups:
    :param segmented_df: DataFrame containing the segmented data.
    :param use_activity_core: Whether to use activity data.
    :param use_ppg_features: Whether to use features extracted from PPG data.
    :param datetime_col: Name of the datetime column.
    :param segment_length: Length of each segment.
    :param start_date: Start date for the patient.
    :param dataloader: Dataloader for loading individual wearable data.
    :param patient_key: The patient ID to process.
    :param stay_end: End date for the patient stay.
    :return: Tuple of the updated segmented dataframe and activity version.
    """
    def filter_between(x): return x.filter(pl.col(datetime_col).is_between(start_date, stay_end))

    df_ppg_features = None
    # segmented_df = segmented_df.sort(by=datetime_col)
    # segmented_df = segmented_df.upsample(
    #     datetime_col, every=segment_length, maintain_order=True
    # ).interpolate()
    if use_ppg_features:
        dataloader.load_ppg_features(patient_key)
        data = dataloader.get_data_dict()
        df_ppg_features = data["ppg_features"].get(patient_key)
        if df_ppg_features is not None:
            df_ppg_features = filter_and_check(df=df_ppg_features, datetime_col=datetime_col,
                                               filter_between=filter_between, patient_key=patient_key,
                                               df_name="PPG features")
        else:
            logging.info(f"PPG feature data not found for patient {patient_key}")

    df_ppg_embeddings = None
    if use_ppg_embeddings:
        dataloader.load_ppg_embeddings(patient_key)
        data = dataloader.get_data_dict()
        df_ppg_embeddings = data["ppg_embeddings"].get(patient_key)
        if df_ppg_embeddings is not None:
            df_ppg_embeddings = filter_and_check(df=df_ppg_embeddings, datetime_col=datetime_col,
                                               filter_between=filter_between, patient_key=patient_key,
                                               df_name="PPG embeddings")
        else:
            logging.info(f"PPG embedding data not found for patient {patient_key}")
    df_activity = None
    df_core = None

    if use_activity_core:
        dataloader.load_activity(patient_key)
        dataloader.load_core(patient_key)
        data = dataloader.get_data_dict()
        df_activity = data["activity"].get(patient_key)
        activity_version = data["activity_versions"].get(patient_key)
        df_core = data["core"].get(patient_key)
        if df_core is not None:
            df_core = filter_and_check(df=df_core, datetime_col=datetime_col,
                                               filter_between=filter_between, patient_key=patient_key,
                                               df_name="Core data")

        else:
            logging.info(f"Core data not found for patient {patient_key}")
        if df_activity is not None:
            df_activity = filter_and_check(df=df_activity, datetime_col=datetime_col,
                                               filter_between=filter_between, patient_key=patient_key,
                                               df_name="Activity data")
        else:
            logging.info(f"Activity data not found for patient {patient_key}")

    # Align with main segmented dataframe
    first_time = pl.DataFrame({datetime_col: start_date})
    last_time = pl.DataFrame({datetime_col: stay_end})
    use_activity_core = True
    if use_activity_core:
        if df_activity is not None and len(df_activity) > 0:
            df_activity = df_activity.with_columns(pl.col(datetime_col).dt.cast_time_unit("us"))
            df_activity = pl.concat([first_time, df_activity, last_time], how="diagonal")
            df_activity = rename_and_select_wearable(df_activity, activity_version)
            if create_wearable_groups:
                df_activity_extracted = create_historical_groups(df_activity, value_cols=None, periods="extended",
                                                                 segment_length=segment_length)
            else:
                df_activity_extracted = df_activity
            segmented_df = segmented_df.join(
                df_activity_extracted, left_on=datetime_col, right_on=datetime_col, how="outer", coalesce=True
            )
        if df_core is not None and len(df_core) > 0:
            df_core = df_core.with_columns(pl.col(datetime_col).dt.cast_time_unit("us"))
            df_core = pl.concat([first_time, df_core, last_time], how="diagonal")
            df_core = rename_and_select_wearable(df_core, "core")
            if create_wearable_groups:
                df_core_extracted = create_historical_groups(df_core, value_cols=None, periods="extended",
                                                             segment_length=segment_length)
            else:
                df_core_extracted = df_core
            segmented_df = segmented_df.join(
                df_core_extracted, left_on=datetime_col, right_on=datetime_col, how="outer", coalesce=True
            )
    if use_ppg_features and df_ppg_features is not None and len(df_ppg_features) > 0:
        df_ppg_features = df_ppg_features.with_columns(pl.col(datetime_col).dt.cast_time_unit("us"))
        df_ppg_features = pl.concat([first_time, df_ppg_features, last_time], how="diagonal")
        if create_wearable_groups:
            df_ppg_features_extracted = create_historical_groups(
                df_ppg_features, value_cols=None, periods="reduced", segment_length=segment_length
            )
        else:
            df_ppg_features_extracted = df_ppg_features
        segmented_df = segmented_df.join(
            df_ppg_features_extracted, left_on=datetime_col, right_on=datetime_col, how="outer", coalesce=True
        )
    if use_ppg_embeddings and df_ppg_embeddings is not None and len(df_ppg_embeddings) > 0:
        df_ppg_embeddings = df_ppg_embeddings.with_columns(pl.col(datetime_col).dt.cast_time_unit("us"))
        df_ppg_embeddings = pl.concat([first_time, df_ppg_embeddings, last_time], how="diagonal")
        if create_wearable_groups:
            df_ppg_embeddings_extracted = create_historical_groups(
                df_ppg_embeddings, periods="reduced", segment_length=segment_length, agg_set="embeddings_basic"
            )
        else:
            df_ppg_embeddings_extracted = df_ppg_embeddings
        segmented_df = segmented_df.join(
            df_ppg_embeddings_extracted,
            left_on=datetime_col,
            right_on=datetime_col,
            how="outer",
            coalesce=True,
        )
    # segmented_df = segmented_df.sort(by=datetime_col)
    # segmented_df = segmented_df.upsample(
    #     datetime_col, every=segment_length, maintain_order=True
    # ).interpolate()
    return segmented_df, activity_version


def filter_and_check(df, datetime_col, filter_between, patient_key, df_name="PPG embedding data"):
    df = df.sort(by=pl.col(datetime_col))
    df = filter_between(df)
    df = df.unique(subset=[datetime_col])
    if len(df) == 0:
        logging.warning(f"{df_name} exists but not in period of interest for: {patient_key}")
    else:
        logging.info(
            f"Length of df {df_name} after filtering: {len(df)}, "
            f"timespan: {df[datetime_col].max() - df[datetime_col].min()}, "
            f"min: {df[datetime_col].min()}, max: {df[datetime_col].max()}"
        )
    return df


def upsample_df(df, freq, datetime_col="datetime", impute=True):
    """
    Upsamples polars dataframe according to frequency
    :param df:
    :param freq:
    :param datetime_col:
    :return:
    """
    # Sorting required
    df = df.sort(by=pl.col(datetime_col))
    logging.info(f"Upsampling with frequency: {freq}")
    # Upsampling with the frequency
    df = (
        df.upsample("datetime", every=datetime.timedelta(milliseconds=1 / freq * 1000), maintain_order=True)
        .interpolate()
        .fill_null(strategy="forward")
    )
    return df



def create_historical_groups(
        df: pl.DataFrame,
        periods: str = "reduced",
        value_cols: list[str] | None = None,
        segment_length: timedelta = timedelta(hours=1),
        datetime_col: str = "datetime",
        agg_set: str = "advanced",
        include_boundaries: bool = False,
) -> pl.DataFrame:
    """
    Aggregates historical statistics over specified periods for each time point in the dataframe.

    :param df: Input polars DataFrame containing time series data.
    :param periods: List of time intervals (as timedelta) over which to compute historical features.
    :param value_cols: List of column names to aggregate. If None, all columns except datetime_col are used.
    :param segment_length: String representing the frequency for dynamic grouping (e.g., "1h").
    :param datetime_col: Name of the datetime column in the dataframe.
    :param agg_set: Which set of aggregation functions to use: 'basic', 'standard', or 'advanced'.
    :param include_boundaries: Whether to generate boundaries in the grouping. Good for debugging periods.
    :return: DataFrame with aggregated historical features for each period.
    """
    if value_cols is None:
        value_cols = list(df.select(pl.all().exclude([datetime_col])).columns)
    logging.debug(f"Creating historical groups with {len(value_cols)} columns, using aggregation set: {agg_set}")
    selected_agg_funcs = aggregation_sets.get(agg_set, aggregation_sets["advanced"])
    selected_agg_periods = aggregation_periods.get(periods, aggregation_periods["reduced"])
    # df = df.with_columns(pl.col(value_cols).replace(0, None))
    # df = df.with_columns(pl.col(value_cols).fill_null(strategy="forward"))
    historical = None

    for period in selected_agg_periods:
        # df_offset = df.with_columns(pl.col(datetime_col) + period)
        df_offset = df
        min = df_offset[datetime_col].min()
        smaller_hour_offset = segment_length - datetime.timedelta(
            minutes=min.minute, seconds=min.second, microseconds=min.microsecond
        )
        # print(f"offset:{-period-smaller_hour_offset}")
        # aggregated = df_offset.group_by_dynamic(
        #     index_column=datetime_col, period=period, every=segment_length, offset=-period - smaller_hour_offset
        # ).agg(
        #     [pl.col(col).mean().alias(f"{col}_mean_{period}") for col in value_cols]
        #     + [pl.col(col).min().alias(f"{col}_min_{period}") for col in value_cols]
        #     + [pl.col(col).max().alias(f"{col}_max_{period}") for col in value_cols]
        #     + [pl.col(col).std().alias(f"{col}_std_{period}") for col in value_cols]
        #     + [pl.col(col).count().alias(f"{col}_count_{period}") for col in value_cols]
        #     + [pl.col(col).skew().alias(f"{col}_skew_{period}") for col in value_cols]
        #     + [pl.col(col).entropy().alias(f"{col}_entropy_{period}") for col in value_cols]
        # )
        # aggregated = aggregated.with_columns(
        #     (pl.col(datetime_col) + (period - segment_length)).alias(datetime_col)
        # )
        agg_expressions = []
        for col in value_cols:
            for func, func_name in selected_agg_funcs:
                if period >= datetime.timedelta(hours=24):
                    # Convert timedelta to total hours if >= 24 hours to prevent large timedelta strings
                    total_hours = int(period.total_seconds() / 3600)
                    period_str = f"{total_hours}:00"
                else:
                    period_str = str(period)

                expr = func(col).alias(f"{col}_agg_with_{func_name}_over_{period_str}h")
                agg_expressions.append(expr)

        # We need to have a strange offset that allows for segment_lengths smaller than one hour
        aggregated = df.group_by_dynamic(
            datetime_col, every=segment_length, period=period, include_boundaries=include_boundaries, closed="right",
            offset=-(period + smaller_hour_offset + timedelta(hours=1) + segment_length)
        ).agg(agg_expressions)
        # aggregated = aggregated.with_columns(pl.col("_lower_boundary").alias(f"{period}_lower_bound"),
        #                                      pl.col("_upper_boundary").alias(f"{period}_upper_bound"))
        if include_boundaries:
            aggregated = aggregated.rename({"_lower_boundary": f"{period}_{value_cols[0]}_lower_bound",
                                            "_upper_boundary": f"{period}_{value_cols[0]}_upper_bound"})
        if segment_length < datetime.timedelta(hours=1):
            aggregated = aggregated.with_columns(pl.col(datetime_col)+ period + 2*segment_length - timedelta(hours=1))
        else:
            aggregated = aggregated.with_columns(pl.col(datetime_col) + period + segment_length - timedelta(hours=1))
        # aggregated = df_offset.group_by_dynamic(
        #     index_column=datetime_col,
        #     period=period,
        #     every=segment_length,
        #     offset=-period #- smaller_hour_offset
        # ).agg(agg_expressions)
        # segment = df.group_by_dynamic(index_column="datetime", period="1h", every="1h").agg(pl.col("core_temp").mean().alias("core_temp_mean_segment"))
        aggregated = aggregated.filter(
            pl.col(datetime_col).is_between(df[datetime_col].min(), df[datetime_col].max())
        )
        if historical is None:
            historical = aggregated
        else:
            historical = historical.join(aggregated, on=datetime_col, how="left")
    logging.debug(f"Created historical groups with {len(historical.columns)} columns from {len(value_cols)} columns")
    return historical

# def create_historical_embedding_groups(
#         df: pl.DataFrame,
#         periods: list[datetime.timedelta] = [timedelta(hours=24), timedelta(hours=1)],
#         value_cols: list[str] | None = None,
#         segment_length: str | timedelta = "1h",
#         datetime_col: str = "datetime",
#     ) -> pl.DataFrame:
#     """
#     Aggregates historical statistics over specified periods for each time point in the dataframe.
#
#     :param df: Input polars DataFrame containing time series data.
#     :param periods: List of time intervals (as timedelta) over which to compute historical features.
#     :param value_cols: List of column names to aggregate. If None, all columns except datetime_col are used.
#     :param segment_length: String representing the frequency for dynamic grouping (e.g., "1h").
#     :param datetime_col: Name of the datetime column in the dataframe.
#     :return: DataFrame with aggregated historical features for each period.
#     """
#     if value_cols is None:
#         value_cols = list(df.select(pl.all().exclude([datetime_col])).columns)
#     # logging.info(value_cols)
#     df = df.with_columns(pl.col(value_cols).replace(0, None))
#     df = df.with_columns(pl.col(value_cols).fill_null(strategy="forward"))
#     historical = None
#     for period in periods:
#         # df_offset = df.with_columns(pl.col(datetime_col) + period)
#         df_offset = df
#         min = df_offset[datetime_col].min()
#         smaller_hour_offset = datetime.timedelta(hours=1) - datetime.timedelta(
#             minutes=min.minute, seconds=min.second
#         )
#         # print(f"offset:{-period-smaller_hour_offset}")
#         aggregated = df_offset.group_by_dynamic(
#             index_column=datetime_col, period=period, every=segment_length, offset=-period - smaller_hour_offset
#         ).agg([pl.col(col).mean().alias(f"{col}_mean_{period}") for col in value_cols])
#         aggregated = aggregated.with_columns(
#             (pl.col(datetime_col) + (period - segment_length)).alias(datetime_col)
#         )
#         # segment = df.group_by_dynamic(index_column="datetime", period="1h", every="1h").agg(pl.col("core_temp").mean().alias("core_temp_mean_segment"))
#         aggregated = aggregated.filter(
#             pl.col(datetime_col).is_between(df[datetime_col].min(), df[datetime_col].max())
#         )
#         if historical is None:
#             historical = aggregated
#         else:
#             historical = historical.join(aggregated, on=datetime_col, how="left")
#     return historical
