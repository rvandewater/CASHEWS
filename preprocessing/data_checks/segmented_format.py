import polars as pl
import datetime

def find_non_hourly_intervals(df: pl.DataFrame, datetime_col: str) -> pl.DataFrame:
    """
    Find rows in the DataFrame where the interval between consecutive datetime values is not exactly one hour.

    Parameters:
    df (pl.DataFrame): The Polars DataFrame.
    datetime_col (str): The name of the datetime column.

    Returns:
    pl.DataFrame: DataFrame containing rows where the interval is not exactly one hour.

    Example usage:
    df = pl.DataFrame({
        "datetime": [
            datetime.datetime(2023, 10, 1, 0, 0),
            datetime.datetime(2023, 10, 1, 1, 0),
            datetime.datetime(2023, 10, 1, 2, 0),
            datetime.datetime(2023, 10, 1, 4, 0),  # Missing hour
            datetime.datetime(2023, 10, 1, 5, 0)
        ]
    })
    non_hourly_df = find_non_hourly_intervals(df, "datetime")
    """
    # Sort the DataFrame by the datetime column
    df = df.sort(datetime_col)

    # Calculate the difference between consecutive datetime values
    df = df.with_columns((pl.col(datetime_col).shift(-1) - pl.col(datetime_col)).alias("time_diff"))

    # Define one hour interval
    one_hour = datetime.timedelta(hours=1)

    # Filter rows where the interval is not exactly one hour
    non_hourly_intervals = df.filter(pl.col("time_diff") != one_hour)

    return non_hourly_intervals

