import datetime
import pandas as pd

def calculate_and_describe_durations(
    surgery_time: dict,
    complication_time: dict,
    transfer_time: dict
) -> None:
    """
    Calculates and prints descriptive statistics for durations between surgery, complication, and transfer times.

    Args:
        surgery_time (dict): Dictionary mapping IDs to surgery datetime values.
        complication_time (dict): Dictionary mapping IDs to complication datetime values.
        transfer_time (dict): Dictionary mapping IDs to transfer datetime values.

    Returns:
        None

    Example:
        from preprocessing.data_checks.prepare_interval_dates import prepare_interval_dates
        surgery_time, complication_time, intake_time, outtake_time, transfer_time = prepare_interval_dates(
            root, base, register_export
        )
        calculate_and_describe_durations(surgery_time, complication_time, transfer_time)
    """

    def create_duration_dict(time_dict1, time_dict2):
        return {key: time_dict1[key] - time_dict2[key] for key in time_dict1 if key in time_dict2}

    duration_dicts = {
        "surgery_to_complication": create_duration_dict(complication_time, surgery_time),
        "surgery_to_transfer": create_duration_dict(transfer_time, surgery_time),
        "transfer_to_complication": create_duration_dict(complication_time, transfer_time),
    }

    for name, duration_dict in duration_dicts.items():
        duration_frame = pd.DataFrame.from_dict(duration_dict, orient="index", columns=["duration"])
        duration_frame = duration_frame[duration_frame["duration"] > datetime.timedelta(days=0)]
        print(f"Description for {name}:")
        print(duration_frame.describe(percentiles=[0.10, 0.25, 0.75, 0.80, 0.90, 0.95, 0.99]))





def print_patient_timeline(admission_time, surgery_time, icu_transfer_time, ward_transfer_time, complication_time, discharge_time):
    ward_transfer_after_discharge = []
    complication_after_discharge = []
    complication_in_ward = []
    complication_in_icu = []
    complication_before_surgery = []

    def print_patient_timeline(patient_id):
        print(f"Patient ID: {patient_id}")
        print(f"Admission Time: {admission_time[patient_id]}")
        print(f"Surgery Time: {surgery_time[patient_id]}")
        if patient_id in icu_transfer_time:
            print(f"ICU Transfer Time: {icu_transfer_time[patient_id]}")
        else:
            print("No ICU transfer recorded for this patient.")
        if patient_id in ward_transfer_time:
            print(f"Ward Transfer Time: {ward_transfer_time[patient_id]}")
        else:
            print("No ward transfer recorded for this patient.")
        if patient_id in complication_time:
            print(f"Complication Time: {complication_time[patient_id]}")
            if complication_time[patient_id] > discharge_time[patient_id]:
                print("Patient had a complication after discharge.")
                complication_after_discharge.append(patient_id)

            if ward_transfer_time[patient_id] < complication_time[patient_id] < discharge_time[patient_id]:
                print("Patient had a complication in the ward.")
                complication_in_ward.append(patient_id)
            if patient_id in icu_transfer_time:
                if icu_transfer_time[patient_id] < complication_time[patient_id] < ward_transfer_time[patient_id]:
                    print("Patient had a complication in the ICU.")
                    complication_in_icu.append(patient_id)
            if ward_transfer_time[patient_id] < surgery_time[patient_id]:
                print("Patient had a complication after surgery in the ward.")
                complication_before_surgery.append(patient_id)
        else:
            print("No complications recorded for this patient.")
        print(f"Discharge Time: {discharge_time[patient_id]}")
        if ward_transfer_time[patient_id] > discharge_time[patient_id]:
            print("Patient was not transferred to the ward before discharge.")
            ward_transfer_after_discharge.append(patient_id)

    for item in admission_time.keys():
        print_patient_timeline(item)
