import logging
import os

import polars as pl


def get_per_id_nan_ratio(dyn: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the ratio of null values per ID for columns starting with 'wearable_'.

    Parameters:
    dyn (pl.DataFrame): The input DataFrame containing the data.

    Returns:
    pl.DataFrame: A DataFrame with the ratio of null/nan values per ID.
    """
    # Filter columns that start with 'wearable_'
    wearable_columns = [col for col in dyn.columns if col.startswith("wearable_")]

    # Calculate the number of null columns per id
    null_counts_per_id = (
        dyn.select(
            pl.col("id"), *[pl.col(col).is_null().cast(pl.Int32).alias(col) for col in wearable_columns]
        )
        .group_by("id")
        .agg(
            [pl.sum(col).alias(f"{col}_null_count") for col in wearable_columns]
            + [pl.count().alias("row_count")]
        )
    )

    # Calculate the total number of null columns per id
    total_null_counts_per_id = null_counts_per_id.with_columns(
        (sum([pl.col(f"{col}_null_count") for col in wearable_columns]) / pl.col("row_count")).alias(
            "average_null_columns_per_id"
        )
    ).select(pl.col("id"), pl.col("average_null_columns_per_id"))
    total_null_counts_per_id = total_null_counts_per_id.with_columns(
        null_ratio=pl.col("average_null_columns_per_id") / len(wearable_columns)
    )
    total_null_counts_per_id = total_null_counts_per_id.sort("average_null_columns_per_id", descending=True)
    return total_null_counts_per_id


def create_filtered_df(
    dyn: pl.DataFrame, sta: pl.DataFrame, outc: pl.DataFrame, cut_off: float, save_path: str
):
    """
    Filter out rows with a high ratio of null values and save the filtered DataFrames.

    Parameters:
    dyn (pl.DataFrame): The dynamic data DataFrame.
    sta (pl.DataFrame): The static data DataFrame.
    outc (pl.DataFrame): The outcome data DataFrame.
    cut_off (float): The threshold for the null ratio to filter out rows.
    nan_counts (pl.DataFrame): The DataFrame containing the null ratios per ID.
    save_path (str): The directory path to save the filtered DataFrames.
    """
    # Calculate nan ratio for each id
    nan_counts = get_per_id_nan_ratio(dyn)
    ids_with_nan_ratio = nan_counts.filter(pl.col("null_ratio") > cut_off).select("id")
    logging.info(f"Filtering:{ids_with_nan_ratio}")
    filtered_dyn = dyn.filter(~pl.col("id").is_in(ids_with_nan_ratio["id"]))
    filtered_sta = sta.filter(~pl.col("id").is_in(ids_with_nan_ratio["id"]))
    filtered_outc = outc.filter(~pl.col("id").is_in(ids_with_nan_ratio["id"]))
    os.mkdir(save_path)
    filtered_outc.write_parquet(f"{save_path}/outc.parquet")
    filtered_sta.write_parquet(f"{save_path}/sta.parquet")
    filtered_dyn.write_parquet(f"{save_path}/dyn.parquet")


def inspect_nan_modality(data: pl.DataFrame, modality: str = "wearable") -> pl.DataFrame:
    """
    Inspect the null values for columns starting with a specific modality.
    :param data:
    :param modality:
    :return:
    """
    # Filter columns that start with 'wearable_ppgfeature'
    modality_columns = [col for col in data.columns if col.startswith(modality)]

    # Initialize a list to store the results
    results = []

    for column_name in modality_columns:
        # Group by 'id' and calculate the number of NaNs/nulls and total entries for each patient
        nan_counts = data.group_by("id").agg(
            [pl.col(column_name).is_null().sum().alias("nan_count"), pl.count().alias("total_count")]
        )

        # Calculate the ratio of NaNs/nulls to total entries for each patient
        nan_counts = nan_counts.with_columns((pl.col("nan_count") / pl.col("total_count")).alias("nan_ratio"))

        # Calculate the range of NaN ratios per patient
        nan_range = nan_counts.select(
            [
                pl.col("nan_ratio").min().alias("min_nan_ratio"),
                pl.col("nan_ratio").max().alias("max_nan_ratio"),
                pl.col("nan_ratio").mean().alias("mean_nan_ratio"),
                pl.col("nan_ratio").std().alias("std_nan_ratio"),
                pl.col("nan_ratio").quantile(0.25).alias("q1_nan_ratio"),
                pl.col("nan_ratio").median().alias("median_nan_ratio"),
                pl.col("nan_ratio").quantile(0.75).alias("q3_nan_ratio"),
            ]
        )
        nan_range = nan_range.with_columns(pl.lit(column_name).alias("column_name"))

        results.append(nan_range)

    # Concatenate the results for all columns
    final_result = pl.concat(results, how="vertical")
    return final_result
