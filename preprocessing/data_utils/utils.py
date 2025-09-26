import logging
import time

import pandas as pd
import polars as pl
from polars import selectors as cs
from datetime import timedelta

def convert_to_timedelta(df, input_column="datetime", output_column="datetime", segment_length=timedelta(hours=1)):
    """
    Converts the 'datetime' column of a dataframe to show the timedelta from the first entry of each 'id'.
    :param df: the dataframe to convert
    :return: the converted dataframe
    """
    # Convert the dataframe to pandas
    df_pandas = df.to_pandas()
    # Ensure the 'datetime' column is of datetime datatype
    df_pandas[output_column] = pd.to_datetime(df_pandas[input_column])
    # Sort the dataframe by 'id' and 'datetime'
    df_pandas.sort_values(["id", output_column], inplace=True)
    # Transform the 'datetime' column to show timedelta from the first entry of each 'id'
    df_pandas[output_column] = df_pandas.groupby("id")[output_column].transform(lambda x: x - x.min())
    df_pandas[output_column] = df_pandas[output_column].dt.total_seconds() / segment_length.total_seconds()
    df_pandas[output_column] = df_pandas[output_column].astype(int)
    return df_pandas


def extract_date(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col + "_minute"] = df[date_col].dt.minute
    df[date_col + "_hour"] = df[date_col].dt.hour
    df[date_col + "_day"] = df[date_col].dt.day
    df[date_col + "_month"] = df[date_col].dt.month
    df[date_col + "_year"] = df[date_col].dt.year
    return df


def prefix_columns(df, prefix):
    return df.rename(columns={c: prefix + str(c) for c in df.columns if c not in ["id", "datetime"]})


def extract_features_functime(
    data,
    time_col="datetime",
    value_cols=["bracelet_temp", "activity"],
    aggregation_time="6h",
    sampling_freq=12,
    include_nested=False,
):
    """
    Extracts features from the data using functime library (polars backend)
    :param data: Data to extract features from
    :param time_col:
    :param value_col:
    :param aggregation_time:
    :param sampling_freq:
    :param include_nested:
    :return:
    """
    # Note: needs to be at least the amount of values in one segment
    n_lags = min(sampling_freq, len(data))
    index_mass_quantile_q = 0.5
    lempel_ziv_complexity_threshold = 3
    mean_n_absolute_max_n_maxima = 10
    range_count_lower = 0.1
    range_count_upper = 0.9
    # Group data by the specified time interval

    grouped_data = data.group_by_dynamic(time_col, every=aggregation_time, start_by="datapoint")
    # grouped_data = data
    feature_timings = {}
    results = {}
    include_nested = False

    def time_feature_extraction(feature_name, feature_func):
        start_time = time.time()
        result = feature_func()
        end_time = time.time()
        feature_timings[feature_name] = end_time - start_time
        results[feature_name] = result
        return result

    result_dfs = []
    for col in data.columns:
        if col not in value_cols and col != time_col:
            # expanding_df = grouped_data.with_columns([
            #     pl.col("value_mean").cumsum().alias("expanding_mean"),
            #     pl.col("value_std").cumsum().alias("expanding_std"),
            #     pl.col("value_min").cummin().alias("expanding_min"),
            #     pl.col("value_max").cummax().alias("expanding_max"),
            #     pl.col("value_median").cumsum().alias("expanding_median"),
            #     pl.col("value_count").cumsum().alias("expanding_count"),
            #     pl.col("value_sum").cumsum().alias("expanding_sum"),
            # ])
            aggregated_df = grouped_data.agg(
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().alias(f"{col}_std"),
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).max().alias(f"{col}_max"),
                pl.col(col).median().alias(f"{col}_median"),
                pl.col(col).count().alias(f"{col}_count"),
                pl.col(col).sum().alias(f"{col}_sum"),
            )

            segmented_time = aggregated_df.select(time_col)
            aggregated_df = aggregated_df.drop(time_col)
            result_dfs.append(aggregated_df)
    time_data = None
    for value_col in value_cols:
        logging.debug(f"Extracting features for {value_col}")
        streak_length_stats_above = data[value_col].quantile(0.75)
        streak_length_stats_threshold = 0.5
        data = data.with_columns(value_col=pl.col(value_col).cast(pl.Float64))
        # Extract individual features
        time_feature_extraction(
            "absolute_energy",
            lambda: grouped_data.agg(absolute_energy=pl.col(value_col).ts.absolute_energy()),
        )
        time_feature_extraction(
            "absolute_maximum",
            lambda: grouped_data.agg(absolute_maximum=pl.col(value_col).ts.absolute_maximum()),
        )
        time_feature_extraction(
            "absolute_sum_of_changes",
            lambda: grouped_data.agg(absolute_sum_of_changes=pl.col(value_col).ts.absolute_sum_of_changes()),
        )
        # Due to shape mismatch error
        # try:
        #     time_feature_extraction("autocorrelation",
        #                             lambda: grouped_data.agg(autocorrelation=pl.col(value_col).ts.autocorrelation(n_lags=n_lags)))
        # except:
        #     logging.info(f"Error in autocorrelation for {value_col}")
        # logging.info(data)
        time_feature_extraction(
            "benford_correlation",
            lambda: grouped_data.agg(benford_correlation=pl.col(value_col).ts.benford_correlation()),
        )
        time_feature_extraction(
            "binned_entropy",
            lambda: grouped_data.agg(binned_entropy=pl.col(value_col).ts.binned_entropy(bin_count=10)),
        )
        time_feature_extraction("c3", lambda: grouped_data.agg(c3=pl.col(value_col).ts.c3(n_lags)))
        time_feature_extraction("cid_ce", lambda: grouped_data.agg(cid_ce=pl.col(value_col).ts.cid_ce()))
        time_feature_extraction(
            "count_above", lambda: grouped_data.agg(count_above=pl.col(value_col).ts.count_above())
        )
        time_feature_extraction(
            "count_above_mean",
            lambda: grouped_data.agg(count_above_mean=pl.col(value_col).ts.count_above_mean()),
        )
        time_feature_extraction(
            "count_below", lambda: grouped_data.agg(count_below=pl.col(value_col).ts.count_below())
        )
        time_feature_extraction(
            "count_below_mean",
            lambda: grouped_data.agg(count_below_mean=pl.col(value_col).ts.count_below_mean()),
        )
        # cusum = grouped_data.agg(cusum=pl.col(value_col).ts.cusum(threshold=cumsum_threshold, warmup_period=cumsum_warmup_period))
        # print("cusum:", cusum)
        # Error: expected a value of type `f64` for column `bracelet_temp` but found a value of type `f32`

        # detrend = grouped_data.agg(detrend=pl.col(value_col).ts.detrend())
        # print("detrend:", detrend)
        # ComputeError: non-integer `dtype` passed to `int_range`: Float64
        if include_nested:
            time_feature_extraction(
                "energy_ratios", lambda: grouped_data.agg(energy_ratios=pl.col(value_col).ts.energy_ratios())
            )
        time_feature_extraction(
            "first_location_of_maximum",
            lambda: grouped_data.agg(
                first_location_of_maximum=pl.col(value_col).ts.first_location_of_maximum()
            ),
        )
        time_feature_extraction(
            "first_location_of_minimum",
            lambda: grouped_data.agg(
                first_location_of_minimum=pl.col(value_col).ts.first_location_of_minimum()
            ),
        )
        # time_feature_extraction("harmonic_mean", lambda: grouped_data.agg(harmonic_mean=pl.col(value_col).ts.harmonic_mean()))
        time_feature_extraction(
            "has_duplicate", lambda: grouped_data.agg(has_duplicate=pl.col(value_col).ts.has_duplicate())
        )
        time_feature_extraction(
            "has_duplicate_max",
            lambda: grouped_data.agg(has_duplicate_max=pl.col(value_col).ts.has_duplicate_max()),
        )
        time_feature_extraction(
            "has_duplicate_min",
            lambda: grouped_data.agg(has_duplicate_min=pl.col(value_col).ts.has_duplicate_min()),
        )
        time_feature_extraction(
            "index_mass_quantile",
            lambda: grouped_data.agg(
                index_mass_quantile=pl.col(value_col).ts.index_mass_quantile(q=index_mass_quantile_q)
            ),
        )
        time_feature_extraction(
            "large_standard_deviation",
            lambda: grouped_data.agg(
                large_standard_deviation=pl.col(value_col).ts.large_standard_deviation()
            ),
        )
        time_feature_extraction(
            "last_location_of_maximum",
            lambda: grouped_data.agg(
                last_location_of_maximum=pl.col(value_col).ts.last_location_of_maximum()
            ),
        )
        time_feature_extraction(
            "last_location_of_minimum",
            lambda: grouped_data.agg(
                last_location_of_minimum=pl.col(value_col).ts.last_location_of_minimum()
            ),
        )
        time_feature_extraction(
            "lempel_ziv_complexity",
            lambda: grouped_data.agg(
                lempel_ziv_complexity=pl.col(value_col).ts.lempel_ziv_complexity(
                    threshold=lempel_ziv_complexity_threshold
                )
            ),
        )
        time_feature_extraction(
            "linear_trend",
            lambda: grouped_data.agg(linear_trend=pl.col(value_col).ts.linear_trend()).unnest("linear_trend"),
        )
        # TODO: ERROR
        #  Cell In[11], line 103, in extract_features_debug(data, time_col, value_col, aggregation_time, sampling_freq)
        #     100 linear_trend = grouped_data.agg(linear_trend=pl.col(value_col).ts.linear_trend())
        #     101 print("linear_trend:", linear_trend)
        # --> 103 longest_losing_streak = grouped_data.agg(longest_losing_streak=pl.col(value_col).ts.longest_losing_streak())
        #     104 print("longest_losing_streak:", longest_losing_streak)
        #     106 # longest_streak_above_mean = grouped_data.agg(longest_streak_above_mean=pl.col(value_col).ts.longest_streak_above_mean())
        #     107 # print("longest_streak_above_mean:", longest_streak_above_mean)
        #     108
        #    (...)
        #     112 # longest_winning_streak = grouped_data.agg(longest_winning_streak=pl.col(value_col).ts.longest_winning_streak())
        #     113 # print("longest_winning_streak:", longest_winning_streak)
        #
        # File ~/.conda/envs/cass_pipeline/lib/python3.10/site-packages/polars/dataframe/group_by.py:982, in DynamicGroupBy.agg(self, *aggs, **named_aggs)
        #     963 def agg(
        #     964     self,
        #     965     *aggs: IntoExpr | Iterable[IntoExpr],
        #     966     **named_aggs: IntoExpr,
        #     967 ) -> DataFrame:
        #     968     """
        #     969     Compute aggregations for each group of a group by operation.
        #     970
        #    (...)
        #     979         The resulting columns will be renamed to the keyword used.
        #     980     """
        #     981     return (
        # --> 982         self.df.lazy()
        #     983         .group_by_dynamic(
        #     984             index_column=self.time_column,
        #     985             every=self.every,
        #     986             period=self.period,
        #     987             offset=self.offset,
        #     988             label=self.label,
        #     989             include_boundaries=self.include_boundaries,
        #     990             closed=self.closed,
        #     991             group_by=self.group_by,
        #     992             start_by=self.start_by,
        #     993         )
        #     994         .agg(*aggs, **named_aggs)
        #     995         .collect(no_optimization=True)
        #     996     )
        #
        # File ~/.conda/envs/cass_pipeline/lib/python3.10/site-packages/polars/lazyframe/frame.py:2027, in LazyFrame.collect(self, type_coercion, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, no_optimization, streaming, engine, background, _eager, **_kwargs)
        #    2025 # Only for testing purposes
        #    2026 callback = _kwargs.get("post_opt_callback", callback)
        # -> 2027 return wrap_df(ldf.collect(callback))
        #
        # StructFieldNotFoundError: lengths

        # time_feature_extraction("longest_losing_streak",
        #                         lambda: grouped_data.agg(longest_losing_streak=pl.col(value_col).ts.longest_losing_streak()))
        # time_feature_extraction("longest_streak_above_mean", lambda: grouped_data.agg(
        #     longest_streak_above_mean=pl.col(value_col).ts.longest_streak_above_mean()))
        # time_feature_extraction("longest_streak_below_mean", lambda: grouped_data.agg(
        #     longest_streak_below_mean=pl.col(value_col).ts.longest_streak_below_mean()))
        # time_feature_extraction("longest_winning_streak",
        #                         lambda: grouped_data.agg(longest_winning_streak=pl.col(value_col).ts.longest_winning_streak()))
        time_feature_extraction(
            "max_abs_change", lambda: grouped_data.agg(max_abs_change=pl.col(value_col).ts.max_abs_change())
        )
        time_feature_extraction(
            "mean_abs_change",
            lambda: grouped_data.agg(mean_abs_change=pl.col(value_col).ts.mean_abs_change()),
        )
        time_feature_extraction(
            "mean_change", lambda: grouped_data.agg(mean_change=pl.col(value_col).ts.mean_change())
        )
        time_feature_extraction(
            "mean_n_absolute_max",
            lambda: grouped_data.agg(
                mean_n_absolute_max=pl.col(value_col).ts.mean_n_absolute_max(
                    n_maxima=mean_n_absolute_max_n_maxima
                )
            ),
        )
        time_feature_extraction(
            "mean_second_derivative_central",
            lambda: grouped_data.agg(
                mean_second_derivative_central=pl.col(value_col).ts.mean_second_derivative_central()
            ),
        )
        time_feature_extraction(
            "number_crossings",
            lambda: grouped_data.agg(number_crossings=pl.col(value_col).ts.number_crossings()),
        )
        time_feature_extraction(
            "number_peaks",
            lambda: grouped_data.agg(number_peaks=pl.col(value_col).ts.number_peaks(sampling_freq)),
        )
        time_feature_extraction(
            "percent_reoccurring_points",
            lambda: grouped_data.agg(
                percent_reoccurring_points=pl.col(value_col).ts.percent_reoccurring_points()
            ),
        )
        time_feature_extraction(
            "percent_reoccurring_values",
            lambda: grouped_data.agg(
                percent_reoccurring_values=pl.col(value_col).ts.percent_reoccurring_values()
            ),
        )
        time_feature_extraction(
            "permutation_entropy",
            lambda: grouped_data.agg(permutation_entropy=pl.col(value_col).ts.permutation_entropy()),
        )
        time_feature_extraction(
            "range_change", lambda: grouped_data.agg(range_change=pl.col(value_col).ts.range_change())
        )
        time_feature_extraction(
            "range_count",
            lambda: grouped_data.agg(
                range_count=pl.col(value_col).ts.range_count(lower=range_count_lower, upper=range_count_upper)
            ),
        )
        time_feature_extraction(
            "range_over_mean",
            lambda: grouped_data.agg(range_over_mean=pl.col(value_col).ts.range_over_mean()),
        )
        time_feature_extraction(
            "ratio_beyond_r_sigma",
            lambda: grouped_data.agg(ratio_beyond_r_sigma=pl.col(value_col).ts.ratio_beyond_r_sigma()),
        )
        time_feature_extraction(
            "ratio_n_unique_to_length",
            lambda: grouped_data.agg(
                ratio_n_unique_to_length=pl.col(value_col).ts.ratio_n_unique_to_length()
            ),
        )
        time_feature_extraction(
            "root_mean_square",
            lambda: grouped_data.agg(root_mean_square=pl.col(value_col).ts.root_mean_square()),
        )
        if include_nested:
            time_feature_extraction(
                "streak_length_stats",
                lambda: grouped_data.agg(
                    streak_length_stats=pl.col(value_col).ts.streak_length_stats(
                        above=streak_length_stats_above, threshold=streak_length_stats_threshold
                    )
                ),
            )
        time_feature_extraction(
            "sum_reoccurring_points",
            lambda: grouped_data.agg(sum_reoccurring_points=pl.col(value_col).ts.sum_reoccurring_points()),
        )
        time_feature_extraction(
            "sum_reoccurring_values",
            lambda: grouped_data.agg(sum_reoccurring_values=pl.col(value_col).ts.sum_reoccurring_values()),
        )
        time_feature_extraction(
            "symmetry_looking",
            lambda: grouped_data.agg(symmetry_looking=pl.col(value_col).ts.symmetry_looking()),
        )
        time_feature_extraction(
            "time_reversal_asymmetry_statistic",
            lambda: grouped_data.agg(
                time_reversal_asymmetry_statistic=pl.col(value_col).ts.time_reversal_asymmetry_statistic(
                    n_lags=n_lags
                )
            ),
        )
        time_feature_extraction(
            "var_gt_std", lambda: grouped_data.agg(var_gt_std=pl.col(value_col).ts.var_gt_std())
        )
        time_feature_extraction(
            "variation_coefficient",
            lambda: grouped_data.agg(variation_coefficient=pl.col(value_col).ts.variation_coefficient()),
        )

        for feature, timing in feature_timings.items():
            logging.debug(f"{feature}: {timing:.4f} seconds")
        time_data = list(results.values())[0][time_col]
        res = pl.concat(results.values(), how="align").select(
            pl.all().exclude(time_col).name.prefix(value_col + "_")
        )
        # time_data = res.select(time_col)
        # result_dfs.append(res.drop(time_col))
        result_dfs.append(res)
    result_df = pl.concat(result_dfs, how="horizontal")
    if time_data is not None:
        result_df = result_df.with_columns(time_data)
    else:
        # We have to add the time column manually
        result_df = pl.concat([result_df, segmented_time], how="horizontal")
    result_df = result_df.with_columns(cs.boolean().cast(pl.Int32))
    result_df = result_df.select(pl.col(time_col), pl.all().exclude(time_col))
    result_df = result_df.with_columns(pl.col(time_col).dt.cast_time_unit("us"))
    return result_df
