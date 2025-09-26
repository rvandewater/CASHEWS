import pandas as pd
from pdms.utils import case_id_to_numeric
from datetime import datetime
import pandas as pl
from omegaconf import OmegaConf
import os

def check_wearable_normal_ward(cfg: str) -> pd.DataFrame:
    """
    Loads configuration and computes the time difference between the first transfer to a normal ward
    and the earliest wearable data timestamp for each patient. Returns a DataFrame with the timedelta
    for each patient.

    Args:
        cfg: Path to the OmegaConf YAML configuration file.

    Returns:
        pd.DataFrame: DataFrame indexed by patient id with a 'timedelta' column.
    """
    cfg = OmegaConf.load(cfg)

    transfer_time = pl.read_parquet(cfg.transfer_path)
    transfer_time.rename(columns={"cassandra1_id": "id"}, inplace=True)
    transfer_time = case_id_to_numeric(transfer_time, "id")
    transfer_time["first_transfer_ward_UTC"] = pd.to_datetime(transfer_time["first_transfer_ward_UTC"])
    transfer_time_dict = pd.Series(
        transfer_time.first_transfer_ward_UTC.values, index=transfer_time.id.values
    ).to_dict()

    core_dir = (cfg.dataloader.folders.core)  # "device_hub_export_via_api/core_no_cuttoff_UTC/"
    core_time = {}
    for item in transfer_time_dict.items():
        if str(item[0]) + ".parquet" in os.listdir(core_dir):
            dat = pl.read_parquet(os.path.join(core_dir, str(item[0]) + ".parquet"))
            dat = dat.drop_nulls()
            if "datetime" in dat.columns:
                core_time[item[0]] = dat["datetime"].min()
            else:
                core_time[item[0]] = dat["date_time"].min()
            if core_time[item[0]] is None:
                print(f"Patient has no data {item[0]}")
    for key, value in core_time.items():
        if value is not None:
            core_time[key] = value.replace(tzinfo=None)
    stay_times = {}
    for key, value in transfer_time_dict.items():
        if key in core_time.keys():
            try:
                stay_times[key] = core_time[key] - value
            except Exception as e:
                print(f"Patient {key} no data {core_time[key]}, {value}, {e}")
        else:
            print(f"Patient {key} has no wearable time")
    stay_df = pd.DataFrame.from_dict(stay_times, orient="index", columns=["timedelta"])
    stay_df.index.rename("id")
    print(stay_df.describe(percentiles=[0.10, 0.25, 0.75, 0.90, 0.95, 0.99]))
    return stay_df


def check_wearable(id: str | int, date_col: str, cfg: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads activity and core wearable data for a given patient ID, sorts and groups the data by day,
    and returns the aggregated counts per day for both activity and core datasets.

    Args:
        id (str or int): Patient identifier.
        date_col (str): Name of the datetime column to use for sorting and grouping.
        cfg (str): Path to the OmegaConf YAML configuration file.

    Returns:
        tuple: Two DataFrames (activity, core) with daily aggregated counts.
    """
    cfg = OmegaConf.load(cfg)
    act = pl.read_parquet(
        f"{cfg.dataloader.folders.activity}{id}.parquet"
    )
    core = pl.read_parquet(
        f"{cfg.dataloader.folders.core}{id}.parquet"
    )
    act = act.sort(by=date_col)
    act = act.group_by_dynamic(date_col, every=datetime.timedelta(days=1)).agg(
        pl.col(date_col).count().alias("count")
    )
    date_col = "datetime"
    core = core.sort(by=date_col)
    core = core.group_by_dynamic(date_col, every=datetime.timedelta(days=1)).agg(
        pl.col(date_col).count().alias("count")
    )
    return act, core

