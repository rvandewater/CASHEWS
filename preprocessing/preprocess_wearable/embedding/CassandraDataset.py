import datetime
import os

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from preprocessing.preprocess_wearable.cleaning.ppg_cleaning import (
    split_df_by_time_gaps,
)
from preprocessing.preprocess_wearable.ppg_paths import CLEANED_PPG_BASE_PATH
from preprocessing.preprocess_wearable.wearable_utils import (
    add_time_diff_col,
    compute_sampling_freq,
    resample_df,
)

CLEANED_DATA_DIR = f"{CLEANED_PPG_BASE_PATH}/2025-01-09_gap-60s_corsanov2-green-light"
DEFAULT_WINDOW_SIZE = 32 * 60  # 60 seconds
DEFAULT_STRIDE = DEFAULT_WINDOW_SIZE // 2


def get_windows(ppg_data, window_size, stride, skip_zero_stddev=True):
    windows = []
    num_windows = 0
    skipped_windows = 0
    for start in range(0, len(ppg_data) - window_size + 1, stride):
        end = start + window_size
        window = ppg_data[start:end]
        # acc_window = acc_data[start:end]

        num_windows += 1
        # Test if window std.dev. is not zero
        if skip_zero_stddev and np.std(window) == 0:
            skipped_windows += 1
            # print(f'Skipping window with zero std.dev. at {pat_file} start={start} end={end}')
            continue

        windows.append(window.reshape((1, -1)))
    return windows, num_windows, skipped_windows


def get_cassandra_dataloaders(
    ppg_path=CLEANED_DATA_DIR,
    window_size=DEFAULT_WINDOW_SIZE,
    stride=DEFAULT_STRIDE,
    batch_size=32,
    test_batch_size=32,
    patient_subset=1.0,
    train_fraction=0.8,
    undersample_n=None,
):
    if not os.path.exists(ppg_path):
        raise FileNotFoundError(f"Path {ppg_path} does not exist.")

    all_patients = [f for f in os.listdir(ppg_path) if ".parquet" in f]
    print(f"Found {len(all_patients)} patients.")
    if patient_subset is None:
        patient_subset = 1.0
    if isinstance(patient_subset, float):
        all_patients = np.random.choice(all_patients, int(len(all_patients) * patient_subset), replace=False)
    elif isinstance(patient_subset, list):
        all_patients = [
            f
            for f in all_patients
            if any([f"{single_patient}.parquet" == f for single_patient in patient_subset])
        ]
    else:
        raise ValueError("patient_subset must be a float or a list of strings.")
    print(f"Using {len(all_patients)} patients.")
    train_patients = np.random.choice(all_patients, int(len(all_patients) * train_fraction), replace=False)
    train_ds = CassandraDataset(
        ppg_path, window_size=window_size, stride=stride, patient_subset=train_patients
    )
    test_ds = CassandraDataset(
        ppg_path,
        window_size=window_size,
        stride=stride,
        patient_subset=[f for f in all_patients if f not in train_patients],
    )
    print(f"Loaded {len(train_ds)} training samples and {len(test_ds)} test samples.")

    train_sampler = test_sampler = None
    if undersample_n is not None:
        print(
            f"Undersampling to {undersample_n} samples. (train: {int(undersample_n * train_fraction)}, test: {int(undersample_n * (1 - train_fraction))})"
        )
        # Undersample the data
        train_sampler = torch.utils.data.RandomSampler(
            train_ds, replacement=False, num_samples=int(undersample_n * train_fraction)
        )
        test_sampler = torch.utils.data.RandomSampler(
            test_ds, replacement=False, num_samples=(undersample_n - int(undersample_n * train_fraction))
        )

    # Prepare data_loader
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None)
    )
    if train_fraction == 1.0:
        test_dl = []
    else:
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=test_batch_size, sampler=test_sampler, shuffle=(train_sampler is None)
        )
    print(f"Loaded data loaders with batch size {batch_size} and test batch size {test_batch_size}.")
    print(f"Number of batches: train: {len(train_dl)}, test: {len(test_dl)}")

    return train_dl, test_dl


class CassandraDataset(Dataset):
    def __init__(
        self,
        data_dir=CLEANED_DATA_DIR,
        ppg_segments=None,
        window_size=DEFAULT_WINDOW_SIZE,
        stride=DEFAULT_STRIDE,
        transform=None,
        patient_subset=None,
        include_acc=False,
    ):
        """
        Args:
            data_dir (str): Path to the directory containing participant data.
            window_size (int): Size of each time-series window.
            stride (int): Step size for moving the window.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        # self.windows = {}
        self.windows = []

        if ppg_segments is None:
            self._load_and_preprocess_data(patient_subset, include_acc)
        else:
            self.segment_n_windows = []
            self._preprocess_data_segments(ppg_segments, include_acc)

    def _preprocess_data_segments(self, ppg_segments, include_acc):
        total_data = pl.concat(ppg_segments)
        ppg_mean = total_data["ppg"].mean()
        ppg_std = total_data["ppg"].std()
        if include_acc:
            total_data["acc"].mean()
            total_data["acc"].std()

        for segment in ppg_segments:
            np_segment = segment.select((pl.col("ppg") - ppg_mean) / ppg_std).to_numpy()
            # Todo include accuracy data in windows
            windows, split_num_windows, _ = get_windows(
                np_segment, self.window_size, self.stride, skip_zero_stddev=False
            )
            self.windows.extend(windows)
            self.segment_n_windows.append(split_num_windows)
            print(f"Loaded {split_num_windows} windows.")

    def _load_and_preprocess_data(self, patient_subset, include_acc):
        print("Starting data load")
        if patient_subset is None:
            patient_subset = os.listdir(self.data_dir)

        skipped_windows = 0
        num_windows = 0
        for pat_file in tqdm(patient_subset):
            # if not ('.parquet' in pat_file and (patient_subset is None or pat_file in patient_subset)):
            if ".parquet" not in pat_file:
                continue

            # Load participant data and labels
            df = pl.read_parquet(os.path.join(self.data_dir, pat_file))
            # Todo split at longer gaps again
            df = add_time_diff_col(df)
            ppg_mean = df["ppg"].mean()
            ppg_std = df["ppg"].std()
            if include_acc:
                acc_mean = df["acc"].mean()
                acc_std = df["acc"].std()

            split_dfs = split_df_by_time_gaps(
                df, self.window_size // compute_sampling_freq(df), verbose=False
            )

            for split_df in split_dfs:
                # upsample Corsano V1
                if split_df.row(1, named=True)["datetime"] - split_df.row(0, named=True)[
                    "datetime"
                ] > datetime.timedelta(microseconds=31250):
                    split_df = resample_df(split_df)

                ppg_data = split_df.select((pl.col("ppg") - ppg_mean) / ppg_std).to_numpy()
                if include_acc:
                    split_df.select((pl.col("acc") - acc_mean) / acc_std).to_numpy()
                    # Todo include accuracy data in windows

                # Create windows
                split_windows, split_num_windows, split_num_skipped_windows = get_windows(
                    ppg_data, self.window_size, self.stride
                )
                num_windows += split_num_windows
                skipped_windows += split_num_skipped_windows
                self.windows.extend(split_windows)

        if num_windows > 0:
            print(
                f"Skipped {skipped_windows}/{num_windows} ({skipped_windows/num_windows*100:.2f}%) window with zero std.dev."
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        if self.transform:
            window = self.transform(window)
        return torch.tensor(window, dtype=torch.float32)
