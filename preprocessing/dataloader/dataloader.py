import os
import pathlib

import pandas as pd

from preprocessing.dataloader.load_methods import return_ppg_features, return_activity, return_core, return_v1_ppg, \
    return_v2_ppg, return_ppg_embeddings, return_cleaned_ppg


class Dataloader:
    """
    Dataloader class for loading and managing data from multiple information systems and modalities.

    This class supports loading data from sources such as ishmed, copra6, and redcap (complications),
    as well as wearable device data (core, PPG, activity, features, embeddings). It provides methods
    to load, prefix, and organize data for downstream processing.

    Attributes:
        data_dict (dict): Stores loaded dataframes, keyed by modality or source.
        exclusion_columns (list): Columns to exclude from prefixing for modality separation.

    Methods:
        check_modality(pid, data): Loads data for a specific modality and patient ID.
        get_data(pid): Loads all specified modalities for a patient ID.
        load_ishmed(): Loads ishmed data.
        load_copra6(): Loads copra6 data.
        load_redcap(): Loads redcap data.
        load_core(pid): Loads core wearable data.
        load_v1_ppg(pid): Loads version 1 PPG data.
        load_v2_ppg(pid, pretrain): Loads version 2 PPG data, with optional pretraining subset.
        load_cleaned_ppg(pid, pretrain): Loads cleaned PPG data, with optional pretraining subset.
        load_ppg_features(pid, pretrain): Loads PPG feature data, with optional pretraining subset.
        load_ppg_embeddings(pid, pretrain): Loads PPG embedding data, with optional pretraining subset.
        load_activity(pid): Loads activity data.
        get_ids(modality): Returns available IDs for a given modality.
        get_data_dict(): Returns the internal data dictionary.
    """
    data_dict = {}
    # columns to exclude when prefixing to be able to separate modalities at runtime
    exclusion_columns = ["id", "datetime", "date_time"]

    def __init__(self, cfg, to_load: list, pretrain: bool = False):
        """
        Initialize the Dataloader.

        Args:
            cfg: Configuration object containing folder paths and settings.
            to_load (list): List of modalities to load.
            pretrain (bool, optional): Whether to use pretraining subset. Defaults to False.
        """
        self.cfg = cfg
        self.root = cfg.root
        self.to_load = to_load
        self.pretrain = pretrain
        self.core_folder = cfg.dataloader.folders.core
        self.activity_folder = cfg.dataloader.folders.activity
        self.ppg_features_folder = cfg.dataloader.folders.ppg_features
        self.ppg_embeddings_folder = cfg.dataloader.folders.ppg_embeddings
        self.ppg_folder = cfg.dataloader.folders.ppg
        # self.load_devices(offset=offset)

    def check_modality(self, pid: int, data: str) -> object:
        """
        Load data for a specific modality and patient ID.

        Args:
            pid (int): Patient ID.
            data (str): Modality name.

        Returns:
            object: Loaded data for the specified modality.
        """
        if data == "core":
            self.load_core(pid=pid)
        if data == "ishmed":
            self.load_ishmed()
        if data == "copra6":
            self.load_copra6()
        if data == "redcap":
            self.load_redcap()
        if data == "v1_ppg":
            self.load_v1_ppg(pid=pid)
        if data == "v2_ppg":
            self.load_v2_ppg(pid=pid, pretrain=self.pretrain)
        if data == "cleaned_ppg":
            self.load_cleaned_ppg(pid=pid, pretrain=self.pretrain)
        if data == "ppg_features":
            self.load_ppg_features(pid=pid, pretrain=self.pretrain)
        if data == "ppg_embeddings":
            self.load_ppg_embeddings(pid=pid, pretrain=self.pretrain)
        if data == "v1_activity":
            self.load_activity(pid=pid, version=1)
        if data == "v2_activity":
            self.load_activity(pid=pid, version=2)
        return self.data_dict[data]

    def get_data(self, pid: int) -> dict:
        """
        Load all specified modalities for a patient ID.

        Args:
            pid (int): Patient ID.

        Returns:
            dict: Dictionary of loaded dataframes keyed by modality.
        """
        for data in self.to_load:
            if data == "core":
                self.load_core(pid=pid)
            if data == "ishmed":
                self.load_ishmed()
            if data == "copra6":
                self.load_copra6()
            if data == "redcap":
                self.load_redcap()
            if data == "v1_ppg":
                self.load_v1_ppg(pid=pid)
            if data == "v2_ppg":
                self.load_v2_ppg(pid=pid, pretrain=self.pretrain)
            if data == "cleaned_ppg":
                self.load_cleaned_ppg(pid=pid, pretrain=self.pretrain)
            if data == "ppg_features":
                self.load_ppg_features(pid=pid, pretrain=self.pretrain)
            if data == "ppg_embeddings":
                self.load_ppg_embeddings(pid=pid, pretrain=self.pretrain)
            if data == "v1_activity":
                self.load_activity(pid=pid, version=1)
            if data == "v2_activity":
                self.load_activity(pid=pid, version=2)
        return self.data_dict

    def load_ishmed(self) -> None:
        """
        Load ishmed data and store in data_dict.
        """
        folder = os.path.join(self.root, self.cfg.folders.ishmed)
        to_load = self.cfg.ishmed.to_load
        for item in to_load:
            db = f"{item}.parquet"
            self.data_dict[item] = pd.read_parquet(os.path.join(folder, db))

        # procedure = procedure.rename(columns={"end": "datetime"})

    def load_copra6(self) -> None:
        """
        Load copra6 data and store in data_dict.
        """
        folder = os.path.join(self.root, self.cfg.folders.copra6)
        to_load = self.cfg.copra6.to_load
        for item in to_load:
            db = f"{item}.parquet"
            self.data_dict[item] = pd.read_parquet(os.path.join(folder, db))

    # TODO: Prefix all data with the source
    def load_redcap(self) -> None:
        """
        Load redcap data and store in data_dict.
        """
        folder = self.cfg.redcap_folder
        to_load = self.cfg.dataloader.redcap.to_load
        for item in to_load:
            db = f"{item}.parquet"
            self.data_dict[item] = pd.read_parquet(os.path.join(folder, db))

    def load_core(self, pid: int = None) -> object:
        """
        Load core wearable data for a patient.

        Args:
            pid (int, optional): Patient ID.
            offset (int, optional): Offset for loading. Defaults to 0.

        Returns:
            object: Loaded core data.
        """
        folder = self.core_folder
        core_dfs = return_core(data_dir=folder, pid=pid, exclusion_columns=self.exclusion_columns)
        self.data_dict["core"] = core_dfs

    def load_v1_ppg(self, pid: int = None) -> None:
        """
        Load version 1 PPG data for a patient.

        Args:
            pid (int, optional): Patient ID.
        """
        # Corsano
        folder = self.ppg_folder
        ppg_dfs = return_v1_ppg(folder=folder, pid=pid)
        self.data_dict["v1_ppg"] = ppg_dfs

    def load_v2_ppg(self, pid: int = None, pretrain: bool = False) -> None:
        """
        Load version 2 PPG data for a patient.

        Args:
            pid (int, optional): Patient ID.
            pretrain (bool, optional): Use pretraining subset. Defaults to False.
        """
        # Corsano
        # only load a third of the data if pretrain is true
        folder = self.ppg_folder
        ppg_dfs = return_v2_ppg(folder, pid, pretrain)
        self.data_dict["v2_ppg"] = ppg_dfs

    def load_cleaned_ppg(self, pid: int = None, pretrain: bool = False) -> None:
        """
        Load cleaned PPG data for a patient.

        Args:
            pid (int, optional): Patient ID.
            pretrain (bool, optional): Use pretraining subset. Defaults to False.
        """
        # Corsano (Cleaned)
        # only load a third of the data if pretrain is true
        folder = self.cfg.folders.ppg_cleaned
        data_dir = pathlib.Path(os.path.join(folder))
        ppg_dfs = return_cleaned_ppg(data_dir=data_dir, pid=pid, pretrain=pretrain)
        self.data_dict["cleaned_ppg"] = ppg_dfs

    def load_ppg_features(self, pid: int = None, pretrain: bool = False) -> None:
        """
        Load PPG feature data for a patient. Only load a third of the data if pretrain is true

        Args:
            pid (int, optional): Patient ID.
            pretrain (bool, optional): Use pretraining subset. Defaults to False.
        """
        # Corsano (Cleaned)
        #
        data_dir = pathlib.Path(self.ppg_features_folder)
        self.data_dict["ppg_features"] = return_ppg_features(data_dir, pid, pretrain, self.exclusion_columns)

    def load_ppg_embeddings(self, pid: int = None, pretrain: bool = False) -> None:
        """
        Load cleaned precomputed PPG embedding data for a patient. Only load a third of the data if pretrain is true

        Args:
            pid (int, optional): Patient ID.
            pretrain (bool, optional): Use pretraining subset. Defaults to False.
        """
        #
        data_dir = pathlib.Path(self.ppg_embeddings_folder)
        ppg_dfs = return_ppg_embeddings(pid=pid, pretrain=pretrain, data_dir=data_dir,
                                        exclusion_columns=self.exclusion_columns)
        self.data_dict["ppg_embeddings"] = ppg_dfs

    def load_activity(self, pid: int = None) -> None:
        """
        Load Corsano activity data for a patient.

        Args:
            pid (int, optional): Patient ID.
        """
        data_dir = pathlib.Path(self.activity_folder)
        data, versions = return_activity(data_dir=data_dir, pid=pid, exclusion_columns=self.exclusion_columns)
        # if data is not None and data[pid] is not None:
        self.data_dict["activity"] = data
        self.data_dict["activity_versions"] = versions


    def get_ids(self, modality: str) -> list:
        """
        Return available IDs for a given modality.

        Args:
            modality (str): Modality name.

        Returns:
            list: List of available IDs.
        """
        if modality == "core":
            folder = self.core_folder
        elif modality == "activity":
            folder = self.activity_folder
        elif modality == "ppg_features":
            folder = self.ppg_features_folder
        elif modality == "ppg_embeddings":
            folder = self.ppg_embeddings_folder
        else:
            return ValueError("Modality not supported")
        data_dir = pathlib.Path(folder)
        core_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
        core_ids = [int(f.split("/")[-1].split(".")[0]) for f in core_files]
        return core_ids

    def get_data_dict(self) -> dict:
        """
        Return the internal data dictionary.

        Returns:
            dict: Internal data dictionary.
        """
        return self.data_dict
