import logging
import os

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_selection import mutual_info_classif


def _compute_feature_correlations(feature_base_folder: str, null_frac_thresh: float = 0.8, verbose=False):
    corrs = {}
    n_features = None
    feature_names = None
    for filename in os.listdir(feature_base_folder):
        if "parquet" not in filename:
            continue
        df = pl.read_parquet(f"{feature_base_folder}/{filename}")
        if df.shape[0] < 2:
            continue
        df = df.select(pl.exclude("datetime", "signal_completeness"))
        df = df.select(sorted(df.columns))
        if not n_features:
            n_features = df.shape[1]
            feature_names = df.columns

        # Replace features with null count >= 0.8 with nan
        null_features = np.array(df.columns)[
            df.select(pl.all().null_count() / pl.all().len() >= null_frac_thresh).to_numpy().squeeze()
        ]
        df = df.with_columns([pl.lit(float("nan")).alias(col) for col in null_features])

        df = df.drop_nulls()
        # logging.info(df.schema)
        # if df.shape[0] <= 1:
        #     corrs[filename.split('.')[0]] = pl.DataFrame(data=np.zeros((len(df.columns), len(df.columns))), schema=df.schema)
        #     logging.info(f'Only one value for patient {filename.split(".")[0]}')
        #     continue
        if df.shape[0] > 1:
            # Compute correlation matrix
            corrs[filename.split(".")[0]] = df.corr()

    if verbose:
        excluded_patients = [key for key, val in corrs.items() if val.shape != (n_features, n_features)]
        logging.info(f"All NaN feature(s) for {len(excluded_patients)} patients: {excluded_patients}")

    # Aggregate all correlation matrices into a single 3D array
    all_corrs_arr = np.stack(
        [val.to_numpy() for val in corrs.values() if val.shape == (n_features, n_features)], axis=-1
    )
    mean_corrs = np.nanmean(all_corrs_arr, axis=-1)
    if verbose:
        logging.info(
            f"Mean correlations:\n{pd.DataFrame(data=mean_corrs, columns=feature_names, index=feature_names)}"
        )

    return mean_corrs, feature_names


def _compute_mutual_information(feature_base_folder: str, target_data: pl.DataFrame, verbose=False):
    # overall
    joint_dfs = []
    feature_names = None
    if isinstance(target_data.unique("id").select(pl.col("id")).to_numpy().squeeze().tolist(), int):
        iterable = [target_data.unique("id").select(pl.col("id")).to_numpy().squeeze()]
    else:
        iterable = target_data.unique("id").select(pl.col("id")).to_numpy().squeeze().tolist()
    for pat_id in iterable:
        if os.path.exists(f"{feature_base_folder}/{pat_id}.parquet"):
            df = pl.read_parquet(f"{feature_base_folder}/{pat_id}.parquet")
            df = df.select(sorted(df.columns))
            if not feature_names:
                feature_names = df.select(pl.exclude("datetime", "signal_completeness")).columns

            patient_labels = target_data.filter(pl.col("id") == pat_id)
            patient_labels = patient_labels.with_columns(
                (pl.col("datetime") - pl.col("datetime").min()).alias("datetime_hour")
            )

            df = df.with_columns(
                (
                    pl.col("datetime").cast(pl.Datetime)
                    - patient_labels.select(pl.col("datetime").min()).item()
                ).alias("datetime_hour")
            )
            # truncate to hours
            df = df.with_columns(
                ((pl.col("datetime_hour").cast(pl.Int64) // 3_600_000_000) * 3_600_000_000).cast(
                    pl.Duration("us")
                )
            )

            # join features and labels
            joint_data = df.join(patient_labels, on="datetime_hour", how="inner")
            joint_data = joint_data.select(
                pl.exclude(["datetime", "datetime_right", "id", "datetime_hour", "signal_completeness"])
            )
            joint_data = joint_data.drop_nulls()

            if joint_data.height > 0:
                joint_dfs.append(joint_data)
        else:
            logging.info(f"No PPG feature data for patient {pat_id}")
    if len(joint_dfs) > 0:
        overall_df = pl.concat(joint_dfs)
        overall_mis = mutual_info_classif(
            overall_df.select(pl.exclude("label")), overall_df.select(pl.col("label")).to_series()
        )
        overall_mis_df = pd.DataFrame(columns=["MI"], data=overall_mis, index=feature_names)
        overall_mis_df.sort_values(by="MI", ascending=False)

        if verbose:
            logging.info(f"Mutual Information:\n{overall_mis_df}")

        return overall_mis_df
    else:
        logging.info("No joint data found for any patients.")
        return pd.DataFrame(columns=["MI"], index=feature_names)


def select_wearable_features(
    aggregated_dataframe: pl.DataFrame,
    feature_base_folder: str,
    corr_thresh: float = 0.8,
    mi_thresh: float = 1e-6,
    verbose=False,
):
    logging.info(f"feature_base_folder: {feature_base_folder}")
    feature_correlations, feature_names = _compute_feature_correlations(feature_base_folder, verbose=verbose)
    indices = np.argwhere(np.abs(feature_correlations) > corr_thresh)
    highly_correlated_features = [(feature_names[i], feature_names[j]) for i, j in indices if j > i]
    if verbose:
        logging.info(f"Highly correlated features: {highly_correlated_features}")

    mis = _compute_mutual_information(
        feature_base_folder,
        aggregated_dataframe.select(pl.col("id"), pl.col("datetime"), pl.col("label")),
        verbose,
    )
    dropped_features = list(mis[mis["MI"] <= mi_thresh].index)

    # additionally remove one of the correlated features
    for feature_a, feature_b in highly_correlated_features:
        if feature_a in dropped_features or feature_b in dropped_features:
            continue
        if mis.loc[feature_a, "MI"] > mis.loc[feature_b, "MI"]:
            dropped_features.append(feature_b)
        elif mis.loc[feature_a, "MI"] < mis.loc[feature_b, "MI"]:
            dropped_features.append(feature_a)
        else:
            feature_to_drop = np.random.choice([feature_a, feature_b])
            logging.info(
                f"same MI for {feature_a} and {feature_b}, dropping {feature_to_drop} chosen randomly"
            )
            dropped_features.append(feature_to_drop)

    if verbose:
        logging.info(f"Dropping features: {dropped_features}")

    return aggregated_dataframe.select(
        pl.exclude([f"^wearable_ppgfeature_{feature}.*$" for feature in dropped_features])
    )
