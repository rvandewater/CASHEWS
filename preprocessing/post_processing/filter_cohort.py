import logging
import os
import shutil

import polars as pl
import glob

def filter_columns_by_null_percentage(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """
    Filters columns in a Polars DataFrame based on the percentage of null values.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    threshold (float): The percentage threshold for filtering columns.

    Returns:
    pl.DataFrame: The filtered DataFrame with columns having null values below the threshold.
    """
    # Calculate the percentage of null values for each column
    null_percentage = df.select(
        [(pl.col(col).is_null().sum() / pl.count() * 100).alias(col) for col in df.columns]
    )

    # Filter columns based on the threshold
    columns_to_keep = [
        col for col in null_percentage.columns if null_percentage.select(pl.col(col)).item() < threshold
    ]

    # Select the filtered columns
    filtered_df = df.select(columns_to_keep)
    logging.info(f"Filtered out columns: {set(df.columns) - set(columns_to_keep)}")
    return filtered_df


def get_per_id_nan_ratio(dyn: pl.DataFrame):
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


def create_filtered_by_nan_dataset(input_path, cut_off, output_path):
    dyn = pl.read_parquet(f"{input_path}/dyn.parquet")
    sta = pl.read_parquet(f"{input_path}/sta.parquet")

    # Process all outc_* files
    outc_files = glob.glob(f"{input_path}/outc_*")
    filtered_outc_files = {}

    nan_counts = get_per_id_nan_ratio(dyn)
    ids_with_nan_ratio = nan_counts.filter(pl.col("null_ratio") > cut_off).select("id")
    print(f"Filtering {len(ids_with_nan_ratio)} ids: {ids_with_nan_ratio}")

    filtered_dyn = dyn.filter(~pl.col("id").is_in(ids_with_nan_ratio["id"]))
    filtered_sta = sta.filter(~pl.col("id").is_in(ids_with_nan_ratio["id"]))

    # Filter each outc_* file
    for outc_file in outc_files:
        outc = pl.read_parquet(outc_file)
        filtered_outc = outc.filter(~pl.col("id").is_in(ids_with_nan_ratio["id"]))
        filtered_outc_files[outc_file] = filtered_outc

    # Create output directory
    os.mkdir(output_path)

    # Save filtered files
    for outc_file, filtered_outc in filtered_outc_files.items():
        outc_filename = os.path.basename(outc_file)
        filtered_outc.write_parquet(f"{output_path}/{outc_filename}")

    filtered_sta.write_parquet(f"{output_path}/sta.parquet")
    filtered_dyn.write_parquet(f"{output_path}/dyn.parquet")
    shutil.copy(f"{input_path}/vars.gin", f"{output_path}/vars.gin")


def filter_dataset(input_path, ids_to_filter, output_path):
    """
    Filters the dataset based on a cutoff value for missing data.

    Args:
        input_path (str): Path to the directory containing the dataset.
        ids_to_filter (list):
        output_path (str): Path to the directory where filtered files will be saved.

    Returns:
        None
    """
    dyn = pl.read_parquet(f"{input_path}/dyn.parquet")
    sta = pl.read_parquet(f"{input_path}/sta.parquet")
    outc = pl.read_parquet(f"{input_path}/outc.parquet")
    filtered_dyn = dyn.filter(~pl.col("id").is_in(ids_to_filter))
    filtered_sta = sta.filter(~pl.col("id").is_in(ids_to_filter))
    filtered_outc = outc.filter(~pl.col("id").is_in(ids_to_filter))
    os.mkdir(output_path)
    filtered_outc.write_parquet(f"{output_path}/outc.parquet")
    filtered_sta.write_parquet(f"{output_path}/sta.parquet")
    filtered_dyn.write_parquet(f"{output_path}/dyn.parquet")
    shutil.copy(f"{input_path}/vars.gin", f"{output_path}/vars.gin")


def create_subgroup_cohorts(input_path, subgroups, output_path, min_patients=50):
    """
    Creates filtered cohorts based on provided subgroups.

    Args:
        input_path (str): Path to the directory containing the dataset files.
        subgroups (dict): Dictionary mapping subgroup names to lists of IDs.
        output_path (str): Path to the directory where filtered cohorts will be saved.
        min_patients (int, optional): Minimum number of patients required for a subgroup. Defaults to 50.
    """
    # Print subgroup information
    for name, ids in subgroups.items():
        print(f"{name}: {len(ids)} patients")

    # Read the main data files
    dyn = pl.read_parquet(f"{input_path}/dyn.parquet")
    sta = pl.read_parquet(f"{input_path}/sta.parquet")
    outc_files = glob.glob(f"{input_path}/outc_*")

    # Create the output directory
    os.makedirs(output_path, exist_ok=True)

    # Create a cohort for each subgroup
    for subgroup_name, ids in subgroups.items():
        if len(ids) < min_patients:
            print(f"Skipping {subgroup_name} with only {len(ids)} patients")
            continue

        # Create a directory for the subgroup
        subgroup_output_path = os.path.join(output_path, subgroup_name.replace(" ", "_"))
        os.makedirs(subgroup_output_path, exist_ok=True)

        # Filter and save the data
        filtered_dyn = dyn.filter(pl.col("id").is_in(ids))
        filtered_sta = sta.filter(pl.col("id").is_in(ids))

        filtered_dyn.write_parquet(f"{subgroup_output_path}/dyn.parquet")
        filtered_sta.write_parquet(f"{subgroup_output_path}/sta.parquet")

        # Process outcome files
        for outc_file in outc_files:
            outc = pl.read_parquet(outc_file)
            filtered_outc = outc.filter(pl.col("id").is_in(ids))
            outc_filename = os.path.basename(outc_file)
            filtered_outc.write_parquet(f"{subgroup_output_path}/{outc_filename}")

        # Copy configuration file
        shutil.copy(f"{input_path}/vars.gin", f"{subgroup_output_path}/vars.gin")

def get_cat_dict(base_data, column_name, id_column="cassandra1_id"):
    """
    Creates a dictionary mapping unique values in a specified column to lists of IDs.
    :param base_data:
    :param column_name:
    :return:
    """
    dict = (
        base_data.group_by(column_name)
        .agg(pl.col(id_column).alias("ids"))
        .to_dict(as_series=False)
    )
    dict = {
        key: value for key, value in zip(dict[column_name], dict["ids"])
    }
    return dict

def create_organ_subcohorts(input_path, base_data, output_path, min_patients=50):
    """
    Creates organ-specific subcohorts using the generic subgroup cohort function.

    Args:
        input_path (str): Path to the directory containing the dataset files.
        base_data (pl.DataFrame): DataFrame containing organ system information.
        output_path (str): Path to the directory where filtered cohorts will be saved.
        min_patients (int, optional): Minimum number of patients required. Defaults to 50.
    """
    target_organsystem_dict = get_cat_dict(base_data, "target_organsystem")

    organ_systems = {
        1: "Liver",
        2: "Pancreas",
        3: "Colorectal",
        4: "Upper GI",
        5: "Cytoreductive surgery",
        6: "Other",
    }

    # Map organ system names to their corresponding IDs
    organ_cohorts = {
        organ_systems[key]: target_organsystem_dict[key]
        for key in target_organsystem_dict
    }

    print(organ_cohorts)

    # Use the generic function to create the cohorts
    create_subgroup_cohorts(input_path, organ_cohorts, output_path, min_patients)

# def create_organ_subcohorts(input_path, base_data, output_path):
#     # Correct grouping and aggregation
#     target_organsystem_dict = (
#         base_data.group_by("target_organsystem")
#         .agg(pl.col("cassandra1_id").alias("ids"))
#         .to_dict(as_series=False)
#     )
#     # Convert to the desired dictionary format
#     target_organsystem_dict = {
#         key: value for key, value in zip(target_organsystem_dict["target_organsystem"], target_organsystem_dict["ids"])
#     }
#     organ_systems = {
#         1: "Liver",
#         2: "Pancreas",
#         3: "Colorectal",
#         4: "Upper GI",
#         5: "Cytoreductive surgery",
#         6: "Other",
#     }
#     organ_cohorts = {}
#     # Map organ system names to their corresponding IDs
#     organ_cohorts = {
#         organ_systems[key]: target_organsystem_dict[key]
#         for key in target_organsystem_dict
#     }
#     print(organ_cohorts)
#     for item in organ_cohorts.items():
#         print(f"{item[0]}: {len(item[1])} patients")
#     dyn = pl.read_parquet(f"{input_path}/dyn.parquet")
#     sta = pl.read_parquet(f"{input_path}/sta.parquet")
#     outc_files = glob.glob(f"{input_path}/outc_*")
#     # Create subcohorts for each organ system
#     os.mkdir(output_path)
#     for organ_system, ids in organ_cohorts.items():
#         if len(ids) < 50:
#             print(f"Skipping {organ_system} with only {len(ids)} patients")
#             continue
#         organ_output_path = os.path.join(output_path, organ_system.replace(" ", "_"))
#         os.mkdir(organ_output_path)
#
#         # Filter dyn and sta files
#         filtered_dyn = dyn.filter(pl.col("id").is_in(ids))
#         filtered_sta = sta.filter(pl.col("id").is_in(ids))
#         # Save filtered dyn and sta files
#         filtered_dyn.write_parquet(f"{organ_output_path}/dyn.parquet")
#         filtered_sta.write_parquet(f"{organ_output_path}/sta.parquet")
#         # Filter and save outc_* files
#         for outc_file in outc_files:
#             outc = pl.read_parquet(outc_file)
#             filtered_outc = outc.filter(pl.col("id").is_in(ids))
#             outc_filename = os.path.basename(outc_file)
#             filtered_outc.write_parquet(f"{organ_output_path}/{outc_filename}")
#
#         # Copy vars.gin file
#         shutil.copy(f"{input_path}/vars.gin", f"{organ_output_path}/vars.gin")
#     create_filtered_dataset(input_path, cut_off, output_path)
# def get_per_id_nan_ratio(dyn: pl.DataFrame, filter_string: str = 'wearable_') -> pl.DataFrame:
#     # Filter columns that start with 'wearable_'
#     wearable_columns = [col for col in dyn.columns if col.startswith(filter_string)]
#
#     # Calculate the number of null columns per id
#     null_counts_per_id = dyn.select(
#         pl.col("id"),
#         *[pl.col(col).is_null().cast(pl.Int32).alias(col) for col in wearable_columns]
#     ).group_by("id").agg(
#         [pl.sum(col).alias(f"{col}_null_count") for col in wearable_columns] + [pl.count().alias("row_count")]
#     )
#
#     # Calculate the total number of null columns per id
#     total_null_counts_per_id = null_counts_per_id.with_columns(
#         (sum([pl.col(f"{col}_null_count") for col in wearable_columns]) / pl.col("row_count")).alias("average_null_columns_per_id")
#     ).select(
#         pl.col("id"),
#         pl.col("average_null_columns_per_id")
#     )
#     total_null_counts_per_id = total_null_counts_per_id.with_columns(null_ratio=pl.col("average_null_columns_per_id") / len(wearable_columns))
#     total_null_counts_per_id = total_null_counts_per_id.sort("average_null_columns_per_id", descending=True)
#     return total_null_counts_per_id
