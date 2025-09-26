import logging
import os
import pathlib
from typing import Any

import polars as pl
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def return_ppg_features(
    data_dir: pathlib.Path,
    pid: int | None,
    pretrain: bool,
    exclusion_columns: list[str] | None = None
) -> dict[int, pl.DataFrame]:
    """
    Loads PPG feature data for a given patient ID or all available files.

    Args:
        data_dir (pathlib.Path): Path to the folder containing PPG feature data.
        pid (int | None): Patient ID to load. If None, loads all available files.
        pretrain (bool): If True, loads only a third of the data for pretraining.
        exclusion_columns (list[str] | None): Columns to exclude from prefixing.

    Returns:
        dict[int, pl.DataFrame]: Dictionary mapping patient IDs to their PPG feature dataframes.
    """
    if pid is None:
        ppg_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    else:
        ppg_files = [os.path.join(data_dir, f"{pid}.parquet")]
    ppg_dfs = {}
    if pretrain:
        ppg_files = ppg_files[: len(ppg_files) // 3]
    for file in tqdm(ppg_files, desc="Loading ppg feature files"):
        try:
            pid = int(file.split("/")[-1].split(".")[0])
            if not os.path.exists(file):
                logging.info(f"No data for {pid} in ppg features.")
                continue
            df = pl.read_parquet(file)
            df = df.with_columns(pl.col("datetime").dt.replace_time_zone(None))
            ppg_dfs[pid] = df.rename(
                lambda x: f"wearable_ppgfeature_{x}" if x not in exclusion_columns else x
            )
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
    return ppg_dfs


def return_activity(
    data_dir: pathlib.Path,
    pid: int | None,
    exclusion_columns: list[str] | None = None
) -> tuple[dict[Any, Any], dict[Any, Any]] | None:
    """
    Loads activity data for a given patient ID or all available files.

    Args:
        data_dir (pathlib.Path): Path to the folder containing activity data.
        pid (int | None): Patient ID to load. If None, loads all available files.
        exclusion_columns (list[str] | None): Columns to exclude from prefixing.

    Returns:
        data: dict[int, pl.DataFrame] with patient IDs and their respective activity dataframes.
        versions: dict[int, int] with patient IDs and their respective activity version numbers.
    """
    # data_dir = pathlib.Path(os.path.join(self.root, folder, f"v{version}_activity"))
    if pid is None:
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    else:
        files = [os.path.join(data_dir, f"{pid}.parquet")]
    dataframes = {}
    logging.info(files)

    versions = {}
    with logging_redirect_tqdm():
        for file in tqdm(
                files,
                desc=f"Loading activity files for {pid}",
        ):
            dataframes[pid] = None
            versions[pid] = None
            if os.path.exists(file):
                logging.info(f"Loading {file}")
                try:
                    df = pl.read_parquet(file)
                except Exception as e:
                    logging.error(f"Could not read activity file for {pid}, exception: {e}")
                    continue
                if len(df) == 0:
                    logging.info(f"Data for {pid} in activity empty.")
                    continue
                if "date_time" in df.columns:
                    df = df.rename({"date_time": "datetime"})
                df = df.with_columns(pl.col("datetime").dt.replace_time_zone(None))
                dataframes[pid] = df.rename(
                    lambda x: f"wearable_activity_{x}" if x not in exclusion_columns else x
                )
                if "hrm_filtered" in df.columns:
                    versions[pid] = 1
                else:
                    versions[pid] = 2

            else:
                logging.info(f"No data for {pid} in activity.")
    return dataframes, versions


def return_core(
    data_dir: str,
    pid: int | None,
    exclusion_columns: list[str]
) -> dict[int, pl.DataFrame]:
    """
    Loads core temperature data for a given patient ID or all available files.

    Args:
        data_dir (str): Path to the folder containing core data.
        pid (int | None): Patient ID to load. If None, loads all available files.
        exclusion_columns (list[str]): Columns to exclude from prefixing.

    Returns:
        dict[int, pl.DataFrame]: Dictionary mapping patient IDs to their core dataframes.
    """
    if pid is None:
        core_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    else:
        core_files = [os.path.join(data_dir, f"{pid}.parquet")]
    core_dfs = {}
    with logging_redirect_tqdm():
        for file in tqdm(core_files, desc=f"Loading core files for ID {pid}"):
            if os.path.exists(file):
                try:
                    df = pl.read_parquet(file)
                except Exception as e:
                    logging.error(f"Could not read core file for {pid}, exception: {e}")
                    continue
                if len(df) == 0:
                    logging.info(f"Data for {pid} in core empty.")
                    continue
                if "date_time" in df.columns:
                    date_column = "date_time"
                elif "datetime" in df.columns:
                    date_column = "datetime"
                df = df.with_columns(pl.col(date_column).dt.replace_time_zone(None))
                if len(df.columns) < 5:
                    return
                if len(df.columns) == 6:
                    df.columns = ["datetime", "skin_temp", "core_temp", "empty", "quality", "device_uuid"]
                elif len(df.columns) == 7:
                    df.columns = [
                        "datetime",
                        "skin_temp",
                        "core_temp",
                        "empty",
                        "quality",
                        "empty2",
                        "device_uuid",
                    ]
                else:
                    df.columns = ["datetime", "skin_temp", "core_temp", "empty", "quality"]
                core_dfs[pid] = df.rename(
                    lambda x: f"wearable_core_{x}" if x not in exclusion_columns else x
                )
            else:
                logging.info(f"No data for {pid} in core")
    return core_dfs


def return_cleaned_ppg(data_dir, pid, pretrain):
    if pid is None:
        ppg_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    else:
        ppg_files = [os.path.join(data_dir, f"{pid}.parquet")]
    ppg_dfs = {}
    if pretrain:
        ppg_files = ppg_files[: len(ppg_files) // 3]
    for file in tqdm(ppg_files, desc="Loading cleaned ppg files"):
        if os.path.exists(file):
            try:
                pid = int(file.split("/")[-1].split(".")[0])
                df = pl.read_parquet(file)
                ppg_dfs[pid] = df
            except Exception as e:
                logging.error(f"Error reading {file}: {e}")
        else:
            logging.info(f"No data for {pid} in cleaned ppg.")
    return ppg_dfs


def return_v1_ppg(
    folder: str,
    pid: int | None,
) -> dict[int, pl.DataFrame]:
    """
    Loads version PPG data from the specified folder.

    Args:
        folder (str): Path to the folder containing data.
        pid (int | None): Patient ID to load. If None, loads all available files.

    Returns:
        dict[int, pl.DataFrame]: Dictionary mapping patient IDs to their PPG dataframes.
    """
    data_dir = pathlib.Path(os.path.join(folder, "v1_ppg"))
    if pid is None:
        ppg_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    else:
        ppg_files = [os.path.join(data_dir, f"{pid}.parquet")]
    ppg_dfs = {}
    for file in tqdm(ppg_files, desc="Loading v1_ppg files"):
        pid = int(file.split("/")[-1].split(".")[0])
        if not os.path.exists(file):
            logging.info(f"No data for {pid} in v1_ppg.")
            continue
        try:
            df = pl.read_parquet(file)
            ppg_dfs[pid] = df
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
    return ppg_dfs


def return_v2_ppg(
    folder: str,
    pid: int | None,
    pretrain: bool
) -> dict[int, pl.DataFrame]:
    """
    Loads version 2 PPG data from the specified folder.

    Args:
        folder (str): Path to the folder containing v2_ppg data.
        pid (int | None): Patient ID to load. If None, loads all available files.
        pretrain (bool): If True, loads only a third of the data for pretraining.

    Returns:
        dict[int, pl.DataFrame]: Dictionary mapping patient IDs to their PPG dataframes.
    """
    data_dir = pathlib.Path(os.path.join(folder, "v2_ppg"))
    if pid is None:
        ppg_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    else:
        ppg_files = [os.path.join(data_dir, f"{pid}.parquet")]
    ppg_dfs = {}
    if pretrain:
        ppg_files = ppg_files[: len(ppg_files) // 3]
    for file in tqdm(ppg_files, desc=f"Loading v2_ppg files for id {pid}"):
        pid = int(file.split("/")[-1].split(".")[0])
        if not os.path.exists(file):
            logging.info(f"No data for {pid} in v2_ppg.")
            continue
        try:
            df = pl.read_parquet(file)
            ppg_dfs[pid] = df
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
    return ppg_dfs


def return_ppg_embeddings(
    pid: int | None,
    pretrain: bool,
    data_dir: pathlib.Path,
    exclusion_columns: list[str] | None = None
) -> dict[int, pl.DataFrame]:
    """
    Loads PPG embedding data for a given patient ID or all available files.

    Args:
        pid (int | None): Patient ID to load. If None, loads all available files.
        pretrain (bool): If True, loads only a third of the data for pretraining.
        data_dir (pathlib.Path): Path to the folder containing PPG embedding data.
        exclusion_columns (list[str] | None): Columns to exclude from prefixing.

    Returns:
        dict[int, pl.DataFrame]: Dictionary mapping patient IDs to their PPG embedding dataframes.
    """
    if pid is None:
        ppg_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    else:
        ppg_files = [os.path.join(data_dir, f"{pid}.parquet")]
    ppg_dfs = {}
    if pretrain:
        ppg_files = ppg_files[: len(ppg_files) // 3]
    for file in tqdm(ppg_files, desc="Loading ppg embedding files"):
        pid = int(file.split("/")[-1].split(".")[0])
        if not os.path.exists(file):
            logging.info(f"No data for {pid} in ppg embeddings.")
            continue
        try:
            df = pl.read_parquet(file)
            df = df.with_columns(pl.col("datetime").dt.replace_time_zone(None))
            ppg_dfs[pid] = df.rename(
                lambda x: f"wearable_ppgembedding_{x}" if x not in exclusion_columns else x
            )
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return ppg_dfs
