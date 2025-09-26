import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from utils import case_id_to_numeric, drop_hdl_columns


def create_parser(pdms: str):
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some input from the command line.")
    if pdms not in ["corsano", "copra6", "redcap"]:
        Exception("PDMS must be either corsano, copra6 or redcap")
    logging.getLogger().setLevel(logging.INFO)
    log_format = "%(asctime)s - %(levelname)s - %(name)s : %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(format=log_format, datefmt=date_format)
    root = Path("/sc-projects/sc-proj-cc08-cassandra/")
    parser.add_argument(
        "--input",
        type=str,
        help="Input value from the command line",
        default=root / f"hdl_data/{pdms}/",
        required=False,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output value from the command line",
        default=root / f"Prospective_Preprocessed/{pdms}/",
        required=False,
    )
    parser.add_argument(
        "--db-format", type=str, help="Database format, can be ht or it", default="it", required=False
    )
    # Parse the arguments
    args = parser.parse_args()
    input = args.input
    output = args.output
    db_format = args.db_format
    if db_format not in ["ht", "it"]:
        Exception("Database format; must be either ht (hudi) or it (iceberg)")

    # Print the input value
    logging.info(f"Input dir: {input}")

    prefix = db_format

    return input, output, prefix


def load_data(input, db_names):
    logging.info(f"Loading data from {input}")
    dict = {}
    for item in db_names:
        db = item
        # use pyarrow as fastparquet leads to unserialized bytestrings
        logging.debug(f"Loading {item}")
        # Loads data from directory of parquet files
        dict[item] = pd.read_parquet(input / db, engine="pyarrow")
    return dict


def general_preprocessing(dict):
    for key, dataset in dict.items():
        # Drop hdl related metadata
        dataset = drop_hdl_columns(dataset)
        # Rename id to case
        dataset.rename(columns={"c_patnr": "id", "c_id": "id"}, inplace=True)
        # Make the case id numeric and uniform
        dataset = case_id_to_numeric(dataset, "id")
        dataset.columns = dataset.columns.str.removeprefix("c_")
        dict[key] = dataset
        logging.info(f"Dataset {key} has shape {dataset.shape} and columns {dataset.columns}")
    return dict


def save_preprocessed(data, name, output):
    logging.info(f"Saving {name} to {output}")
    os.makedirs(output, exist_ok=True)
    data.to_parquet(output / (name + ".parquet"))


def normalize_json(row):
    normalized_data = pd.json_normalize(row["value_decimal"])
    return normalized_data
