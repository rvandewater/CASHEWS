import pandas as pd


def compute_stays(ids, timeseries, time_difference=24 * 60 * 60, save_segments=False):
    """
    Computes the stays of the patients for a timeseries modality
    :param ids: The ids of the patients to process
    :param timeseries: Time series dataframe with at least the following columns: id, datetime
    :param time_difference: The time difference in seconds to consider a new segment, default is 24 hours
    :param save_segments: Whether to save the segments in a dict to return as second argument
    :return:
        segment_interval_dict: A dictionary with the patient id as key and a list of tuples with the start and end of the segment
        segment_dict: A dictionary with the patient id as key and a list of dataframes with the segments
    """
    # Initialize empty dicts to store segment intervals and segments per patient
    segment_interval_dict = {}
    segment_dict = {}
    for patient in ids:
        df = timeseries[timeseries.id == patient].copy()
        df.sort_values(by="datetime", inplace=True)

        # Initialize variables for the current id
        current_segment = []
        segment_list = []
        previous_timestamp = None
        segments = []
        print(f"Processing patient {patient}")
        if len(timeseries) == 0:
            continue
        # Iterate through the dataframe
        for index, row in df.iterrows():
            timestamp = row["datetime"]
            if previous_timestamp is None:
                current_segment.append(row)
            elif (timestamp - previous_timestamp).total_seconds() <= time_difference:
                current_segment.append(row)
            else:
                segments.append(pd.DataFrame(current_segment))
                segment_list.append((current_segment[0].datetime, current_segment[-1].datetime))
                current_segment = [row]
            previous_timestamp = timestamp

        # Add the last segment
        segments.append(pd.DataFrame(current_segment))
        segment_list.append((current_segment[0].datetime, current_segment[-1].datetime))
        if save_segments:
            segment_dict[patient] = segments
        segment_interval_dict[patient] = segment_list
    return segment_interval_dict, segment_dict
