#!/usr/bin/env python
import logging
import time

import pandas as pd
from general import create_parser, general_preprocessing, load_data, save_preprocessed
from pytz import timezone

from utils import convert_to_utc

# RedCAP time is in Europe/Berlin (adjusts for summer time)
current_timezone = timezone("Europe/Berlin")


def main():
    input, output, db_format = create_parser("redcap")
    prefix = db_format
    redcap_names = [
        f"{prefix}_register_export",
        f"{prefix}_diagnostik_export",
        f"{prefix}_komplikations_export",
        f"{prefix}_mmt_export",
        f"{prefix}_core_export",
    ]

    start = time.time()
    redcap_dict = load_data(input, redcap_names)
    redcap_dict = general_preprocessing(redcap_dict)

    RedCAPPreprocessing.register_export(redcap_dict[f"{prefix}_register_export"], output)
    RedCAPPreprocessing.diagnostics(redcap_dict[f"{prefix}_diagnostik_export"], output)
    RedCAPPreprocessing.complications(redcap_dict[f"{prefix}_komplikations_export"], output)

    end = time.time()
    logging.info(f"Preprocessing took {format(end - start, '.2f')} seconds")


class RedCAPPreprocessing:
    @staticmethod
    def register_export(input, output):
        register = input
        register.rename(
            columns={
                "organsystem": "organ_system",
                "aufnahmedatum": "intake_date",
                "entlassdatum": "outtake_date",
            },
            inplace=True,
        )
        register["intake_date"] = pd.to_datetime(register["intake_date"])
        register["outtake_date"] = pd.to_datetime(register["outtake_date"])
        register["intake_date"] = convert_to_utc(register, "intake_date", current_timezone)
        register["outtake_date"] = convert_to_utc(register, "outtake_date", current_timezone)

        register["duration"] = pd.to_datetime(register["outtake_date"]) - pd.to_datetime(
            register["intake_date"]
        )
        # print("Durations:")
        # print(register["duration"].dt.days.median())
        # print(register["duration"].dt.days.quantile(0.75))
        # print(register["duration"].dt.days.quantile(0.25))
        save_preprocessed(register, "register_export", output)

    # # Diagnostics
    @staticmethod
    def diagnostics(input, output):
        diagnostics = input
        save_preprocessed(diagnostics, "diagnostics", output)

    @staticmethod
    def complications(input, output):
        # # Complication preprocessing
        # Code for extracting endpoint data for several endpoints. Currently only the main endpoint of the CASSANDRA project is
        # implemented (SSI = Surgical State Space Infection). The "Zeitpunkt" can be translated as : 1: 9:00, 2, 15:00, 3, 23:59
        complications = input

        # ## Complication time
        # Some complications do not have a recorded time, we fill these using the column "zeitpunkt komplikation"
        def generate_timestamp(complication_time, complication_date, timepoint):
            if complication_date is not None and complication_date != "":
                if complication_time is not None and complication_time != "":
                    logging.debug(f"Exact time: {complication_date}T{complication_time}")
                    return pd.to_datetime(complication_date + "T" + complication_time)
                elif timepoint is None or "":
                    return pd.to_datetime(complication_date)
                else:
                    hours = "0:00"
                    timepoint = int(timepoint) if timepoint.isnumeric() else timepoint
                    if timepoint == 1:
                        hours = "9:00"
                    elif timepoint == 2:
                        hours = "15:00"
                    elif timepoint == 3:
                        hours = "23:59"
                    else:
                        logging.debug(f"Unexpected timepoint {timepoint}")
                    logging.debug(f"Approximate time: {complication_date}T{hours}")
                    return pd.to_datetime(complication_date + "T" + hours)

            else:
                return pd.to_datetime("2022-01-01T00:00:00", format="ISO8601")

        complications["complication_datetime"] = complications.apply(
            lambda x: generate_timestamp(x.complication_time, x.datumkomplikation, x.zeitpunkt_komplikation),
            axis=1,
        )
        complications = convert_to_utc(complications, "complication_datetime", current_timezone)
        # # SSI-III
        # SSI-III (Organ Surgical State Space Infection)
        # https://www.hopkinsmedicine.org/health/conditions-and-diseases/surgical-site-infections

        # Select only the important columns for ssi
        complications_select = complications.filter(["id", "complication_datetime", "wundheilungsst_rung"])
        complications_select.rename(columns={"wundheilungsst_rung": "ssi_severity"}, inplace=True)
        # Make numeric and impute with 0
        complications_select["ssi_severity"] = (
            pd.to_numeric(complications_select["ssi_severity"], errors="coerce")
            .infer_objects(copy=False)
            .fillna(0)
        )
        # complications_select['c_wundheilungsst_rung'].astype(int,errors='ignore')
        # Select the highest value of SSI for each patient
        complications_endpoint = complications_select[complications_select.ssi_severity > 2]
        # complications_endpoint = complications_select.loc[complications_select.groupby('id')['ssi_severity'].apply(lambda x: x>2)]
        complications_endpoint.id.value_counts()
        complications_endpoint
        # Binary label >2 interabdominal wound infection
        complications_endpoint.loc[:, "label_ssi"] = complications_endpoint["ssi_severity"].apply(
            lambda x: 1 if x > 2 else 0
        )
        complications_endpoint[complications_endpoint["label_ssi"] == 1]
        save_preprocessed(complications_endpoint, "complications", output)


if __name__ == "__main__":
    main()
