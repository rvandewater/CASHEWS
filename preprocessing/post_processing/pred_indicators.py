from pathlib import Path

import polars as pl


def load_and_append_prediction_files(directory: str, filename: str) -> pl.DataFrame:
    """
    Load all CSV files with the specified filename in the given directory and its subdirectories,
    and append them vertically into a single Polars DataFrame.

    Parameters:
    directory (str): The root directory to search for CSV files.
    filename (str): The specific filename to look for.

    Returns:
    pl.DataFrame: A single DataFrame containing all the appended CSV files.

    Example usage:
    directory = 'home'
    filename = 'pred_indicators.csv'
    combined_df = load_and_append_csv_files(directory, filename)
    """
    # Initialize an empty list to store DataFrames
    dataframes = []
    # Iterate through all files in the directory and subdirectories
    counter = 0
    for iteration in sorted(Path(directory).iterdir()):
        for file_path in sorted(Path(iteration).rglob(filename)):
            print(f"Loading file: {file_path}")
            # Load the CSV file as a Polars DataFrame
            df = pl.read_csv(file_path)
            # Append the DataFrame to the list
            # df = df.with_columns(pl.col(id_column) + counter * 1300)
            dataframes.append(df)
        counter += 1


    # Concatenate all DataFrames vertically
    combined_df = pl.concat(dataframes, how="vertical")

    return combined_df


def append_predictions_foldwise(directory: str, filename: str, max_id: int=1300) -> pl.DataFrame:
    """
    Load all CSV files with the specified filename in the given directory and its subdirectories,
    and append them vertically into a single Polars DataFrame.
    Files are processed fold by fold across all iterations.

    Parameters:
    directory (str): The root directory to search for CSV files.
    filename (str): The specific filename to look for.
    max_id (int): The maximum ID value to offset the IDs in each fold to prevent clashes.

    Returns:
    pl.DataFrame: A single DataFrame containing all the appended CSV files.

    Example usage:
    directory = 'home'
    filename = 'pred_indicators.csv'
    combined_df = load_and_append_csv_files(directory, filename)
    """
    dataframes = []
    id_column = "# id"
    counter = 0

    # Get all iteration directories sorted
    iterations = sorted([d for d in Path(directory).iterdir() if d.is_dir()])

    # Get all unique fold names across all iterations
    all_folds = set()
    for iteration in iterations:
        for fold_dir in iteration.iterdir():
            if fold_dir.is_dir():
                all_folds.add(fold_dir.name)

    # Process fold by fold across all iterations
    experiment_id = 0
    for fold_name in sorted(all_folds):
        for iteration in iterations:
            fold_path = iteration / fold_name
            if fold_path.exists():
                for file_path in sorted(fold_path.rglob(filename)):
                    print(f"Loading file: {file_path}")

                    # Load the CSV file
                    df = pl.read_csv(file_path)
                    df = df.with_columns(pl.col(id_column).alias("original_id"))
                    df = df.with_columns(pl.lit(experiment_id).alias("experiment_id"))
                    # Check original ID range
                    original_ids = df.select(pl.col(id_column)).to_numpy().flatten()
                    print(f"  Original ID range: {original_ids.min()} - {original_ids.max()}")

                    # Apply offset to prevent clashes
                    df = df.with_columns(pl.col(id_column) + counter * max_id)

                    # Check modified ID range
                    modified_ids = df.select(pl.col(id_column)).to_numpy().flatten()
                    print(f"  Modified ID range: {modified_ids.min()} - {modified_ids.max()}")
                    print(f"  Counter: {counter}")
                    print()
                    dataframes.append(df)
                    experiment_id += 1
        counter += 1

    # Concatenate all DataFrames vertically
    combined_df = pl.concat(dataframes, how="vertical")

    return combined_df
