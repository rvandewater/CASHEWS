import argparse
import datetime
import logging

from omegaconf import DictConfig, OmegaConf

from preprocessing.dataloader.dataloader import Dataloader
from preprocessing.post_processing.save import prepare_dataset
from preprocessing.prepare_segmentation import (
    prepare_interval_dates,
    load_modalities, prepare_base,
)
from preprocessing.segment import segment_modalities


def main(cfg: DictConfig, cohort_name: str = None,
         complication_cohort: str = "real_life_set",
         complication_types = None):
    """
        Main function to run data segmentation.
    :param cfg: Configuration object containing all necessary parameters.
    :param cohort_name: Custom name for the cohort, if None it will be derived from the configuration.
    :param complication_cohort: Cohort for complications, should be 'real_life_set' or 'lab_set'.
    :param complication_types: Type of endpoints to consider, can only be one of 'SSI-3', 'POPF', 'Galleleck/BDA'.
    :return:
        None
    """
    segment_length = datetime.timedelta(hours=cfg.segment.segment_length_hours)
    horizon = datetime.timedelta(hours=cfg.segment.horizon_hours)
    debug = cfg.segment.debug
    cohort = cfg.segment.cohort
    min_stay_length = datetime.timedelta(hours=cfg.segment.min_stay_hours)

    logging.info(cfg)
    logging.info(
        f"Running data segmentation for cohort {cohort} with segment length {segment_length}, "
        f"horizon {horizon}, minimum stay time {min_stay_length}, complication type {complication_types}"
    )

    # Load Datasets
    root = cfg.root

    cat_modalities, copra_modalities, ishmed_modalities, num_modalities = load_modalities(cfg)

    base, base_static = prepare_base(cfg)

    dataloader = Dataloader(cfg, to_load=["v1_activity", "v2_activity", "core", "ppg_features"])

    admission_time, surgery_time, icu_transfer_time, ward_transfer_time, complication_time, discharge_time = (
        prepare_interval_dates(cfg, cohort=complication_cohort, culling=True, complication_types=complication_types
    ))


    exclude_ids = []
    exclude_ids.extend(cfg.segment.exclude_ids)
    logging.info(f"Excluding amount of patients: {len(exclude_ids)}")
    # Main segmentation

    start_time, end_time = calculate_start_end_segment_for_cohort(cohort, discharge_time, complication_time,
                                                                  icu_transfer_time, surgery_time, ward_transfer_time)

    segmented_dataframe, retro_aggregated = segment_modalities(
        num_df_dict=num_modalities,
        cat_df_dict=cat_modalities,
        dataloader=dataloader,
        start_dict=start_time,
        end_dict=end_time,
        endpoint_dict=complication_time,
        segment_length=segment_length,
        prediction_horizon=horizon,
        jobs=-1,
        debug=debug,
        procedure_dict=surgery_time,
        admission_dict=admission_time,
        min_stay_length=min_stay_length,
        wearable_cohort=False if cohort == "ICU" else True,
        exclude_ids=exclude_ids,
        cohort_type=cohort,
    )

    if cohort_name is None:
        cohort_name = ""

    prepare_dataset(
        segmented_dataframe=segmented_dataframe,
        retro_aggregated=retro_aggregated,
        base_static=base_static,
        root=root,
        copra_modalities=copra_modalities,
        ishmed_modalities=ishmed_modalities,
        cat_modalities=cat_modalities,
        version=f"{cohort_name}_{cohort}_{'_'.join(complication_types).replace('/', '_')}_segment_{segment_length.total_seconds() / 3600}"
        f"_{debug if debug else 'full'}"
        f"_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        segment_length=segment_length
    )


def calculate_start_end_segment_for_cohort(
    cohort: str,
    discharge_time: dict[str, datetime.datetime],
    complication_time: dict[str, datetime.datetime],
    icu_transfer_time: dict[str, datetime.datetime],
    surgery_time: dict[str, datetime.datetime],
    ward_transfer_time: dict[str, datetime.datetime],
) -> tuple[dict[str, datetime.datetime], dict[str, datetime.datetime]]:
    """
    Calculate the start and end times for segmentation based on the cohort type and event times.

    Args:
        cohort (str): The cohort type, e.g., "ICU", "normal_ward", or "ICU_and_normal_ward".
        discharge_time (dict[str, datetime.datetime]): Mapping of patient IDs to discharge times.
        complication_time (dict[str, datetime.datetime]): Mapping of patient IDs to complication event times.
        icu_transfer_time (dict[str, datetime.datetime]): Mapping of patient IDs to ICU transfer times.
        surgery_time (dict[str, datetime.datetime]): Mapping of patient IDs to surgery times.
        ward_transfer_time (dict[str, datetime.datetime]): Mapping of patient IDs to ward transfer times.

    Returns:
        tuple[dict[str, datetime.datetime], dict[str, datetime.datetime]]:
            Two dictionaries mapping patient IDs to their respective start and end times for segmentation.
    """

    start_time = {}
    end_time = {}
    if cohort == "normal_ward":
        start_time = ward_transfer_time.copy()
        end_time = discharge_time.copy()
    if cohort == "ICU":
        start_time = icu_transfer_time.copy()
        # We stop when the patient enters the ward
        end_time = ward_transfer_time.copy()
        # If cohort is ICU, include the culled discharge time for other complications
        for key, ward_time in end_time.items():
            if ward_time > discharge_time[key]:
                end_time[key] = discharge_time[key]

    # If cohort is ICU and normal ward, include patients that go directly from OP to normal ward
    if cohort == "ICU_and_normal_ward":
        start_time = icu_transfer_time.copy()
        for key, value in surgery_time.items():
            if key not in start_time.keys():
                if surgery_time[key] < ward_transfer_time[key]:
                    start_time[key] = ward_transfer_time[key]
                else:
                    # Make sure the surgery time is not after the ward transfer time
                    start_time[key] = surgery_time[key]
        end_time = discharge_time.copy()

    # If the patient has a complication, we use the complication time as the end time
    for key, value in end_time.items():
        if key in complication_time.keys():
            if cohort == "ICU":
                if complication_time[key] < end_time[key]:
                    end_time[key] = complication_time[key]
            else:
                end_time[key] = complication_time[key]
        if key in ward_transfer_time.keys() and key in complication_time.keys():
            logging.debug(f"Patient {key} has {end_time[key] - ward_transfer_time[key]}")
    return start_time, end_time


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    log_format = "%(asctime)s - %(levelname)s - %(name)s : %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(format=log_format, datefmt=date_format)

    parser = argparse.ArgumentParser(description="Data Segmentation Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument(
        "--root", type=str, default="/sc-projects/sc-proj-cc08-cassandra/", help="Root folder in filesystem"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Cohort type",
        choices=["ICU", "normal_ward", "ICU_and_normal_ward"],
    )
    # parser.add_argument(
    #     "--complication_type",
    #     type=list,
    #     default=None,
    #     help="Type of complication",
    #     choices=["SSI-3"] #"POPF", "Galleleck_BDA"],
    # )
    parser.add_argument("--cohort_name", type=str, default=None, help="Cohort name")

    parser.add_argument("--segment_length_hours", type=float, default=None, help="Segment length in hours")
    parser.add_argument("--horizon_hours", type=int, default=None, help="Horizon in hours")
    parser.add_argument("--debug", type=int, default=0, help="Debug level")
    parser.add_argument(
        "--complication_cohort",
        type=str,
        default="lab_set",
        help="Cohort for complications, e.g., 'real_life_set' or 'lab_set'",
    )
    parser.add_argument("--complication_types", type=str, default=None,
                        help="String of complication types seperated by comma. "
                        "Can be only be passed as one of these three combinations: 'SSI-3,POPF,Galleleck/BDA'.")
    args = parser.parse_args()

    # Load the config file
    cfg = OmegaConf.load(args.config)

    # Override config values with passed arguments if provided
    if args.root:
        cfg.segment.root = args.root
    if args.segment_length_hours:
        cfg.segment.segment_length_hours = args.segment_length_hours
    if args.horizon_hours:
        cfg.segment.horizon_hours = args.horizon_hours
    if args.debug is not None:
        cfg.segment.debug = args.debug
    if args.cohort:
        cfg.segment.cohort = args.cohort

    if args.complication_types:
        complication_types = args.complication_types.split(",")
        complication_types = [item.strip() for item in complication_types]  # Remove any extra spaces
        print(complication_types)
    else:
        complication_types = cfg.segment.default_complication_types
    logging.info(f"cohort: {cfg.segment.cohort}")
    main(cfg, cohort_name=args.cohort_name, complication_cohort=args.complication_cohort, complication_types=complication_types)
