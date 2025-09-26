import pandas as pd
from tsfresh.feature_extraction import MinimalFCParameters, extract_features


def case_id_to_numeric(data, column_name):
    """Normalizes case id to numeric
    :param data:
    :param column_name:
    :return:
    """
    data[column_name] = data[column_name].astype(str)
    data[column_name] = data[column_name].str.replace("CASS1-", "")
    data[column_name] = data[column_name].str.replace("Cass1-", "")
    data[column_name] = pd.to_numeric(data[column_name])
    return data


def drop_hdl_columns(df):
    """
    Drops HDL generated columns
    :param df:
    :return:
    """
    dropcolumns = [
        "_hdl_dokument_nummer",
        "_hdl_loadstamp_n2labor",
        "_hdl_loadstamp_n2labor001",
        "_hoodie_commit_time_n2labor001",
        "_hoodie_commit_time_n2labor",
        "_hoodie_commit_time",
        "_hoodie_commit_seqno",
        "_hoodie_record_key",
        "_hoodie_partition_path",
        "_hoodie_file_name",
        "_hdl_trace",
        "_hdl_message_hash",
        "_hdl_loadstamp",
    ]
    df = df.drop(columns=[col for col in df if col in dropcolumns])
    return df


def extract_features_df(
    df,
    groupby="c_katalog_leistungtext",
    id_val="c_id",
    sort_var="c_wert_timestamp",
    value="c_wert",
    params=MinimalFCParameters(),
):
    """
    Extracts features from temporal dataframes
    :param df:
    :param groupby:
    :param id_val:
    :param sort_var:
    :param value:
    :param params:
    :return:
    """
    all_extracted_features = pd.DataFrame()
    params.__dict__["data"].pop("sum_values")
    params.__dict__["data"].pop("length")
    for the_type, sub_df in df.groupby(groupby):
        extracted_features = extract_features(
            sub_df, column_id=id_val, column_sort=sort_var, column_value=value, default_fc_parameters=params
        )
        extracted_features.columns = extracted_features.columns.str.replace(value, f"{the_type}_")
        all_extracted_features = pd.concat([all_extracted_features, extracted_features], axis=1)
    all_extracted_features = all_extracted_features.reset_index(drop=False, names="id")
    print(f"Extracted amount of columns: {all_extracted_features.columns}")
    return all_extracted_features


def filter_before(
    group, name, before_dict={}, placeholder=pd.to_datetime("2100-01-01T00:00:00", format="ISO8601")
):
    """
    Filters values of measurements that happened after a certain (complication) date
    :param group:
    :param name:
    :param before_dict:
    :param placeholder:
    :return:
    """
    return group[group["timestamp"] < before_dict.get(name, placeholder)]


def drop_if_in(df, dropcolumns):
    """
    Drops columns if they are present
    :param df:
    :param dropcolumns:
    :return:
    """
    df = df.drop(columns=[col for col in df if col in dropcolumns])
    return df
