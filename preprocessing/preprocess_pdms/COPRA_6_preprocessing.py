#!/usr/bin/env python
import numpy as np
import pandas as pd
from general import create_parser, general_preprocessing, load_data, save_preprocessed

from utils import convert_to_utc, current_timezone, drop_if_in


# Needs to be adjusted to the timezone where the data is queried from
def main():
    input, output, db_format = create_parser("copra6")
    prefix = db_format
    copra6_names = [
        f"{prefix}_copra6_scores",
        f"{prefix}_copra6_observation",
        f"{prefix}_copra6_medication",
        f"{prefix}_copra6_fluid_balance",
        f"{prefix}_copra6_therapy",
    ]

    copra6_dict = load_data(input, copra6_names)
    copra6_dict = general_preprocessing(copra6_dict)

    COPRA6Preprocessing.output = output
    COPRA6Preprocessing.scores(copra6_dict[f"{prefix}_copra6_scores"], output)
    COPRA6Preprocessing.observations(copra6_dict[f"{prefix}_copra6_observation"], output)
    COPRA6Preprocessing.medication(copra6_dict[f"{prefix}_copra6_medication"], output)
    COPRA6Preprocessing.fluid_balance(copra6_dict[f"{prefix}_copra6_fluid_balance"], output)


class COPRA6Preprocessing:
    @staticmethod
    def therapy(therapy):
        therapy = drop_if_in(therapy, ["falnr"])
        therapy.rename(
            columns={
                "start": "datetime_start",
                "end": "datetime_end",
                "apparat_name": "device_name",
                "apparat_mode": "device_mode",
                "apparat_type": "device_type",
                "therapy_type": "therapy_type",
            },
            inplace=True,
        )
        print(f"Most recent timestamp: {max(therapy.datetime_start)}")
        save_preprocessed(therapy, "therapy")

    @staticmethod
    def scores(scores, output):
        # This dataset contains various scores, like SOFA or Glascow Coma Scale. @Axel: how to preprocess this? Which Score to take?
        scores.rename(
            columns={"type": "score_group", "date_time_to": "datetime", "overall_score": "overall_score"},
            inplace=True,
        )
        scores_preprocessed = drop_if_in(scores, ["falnr", "meta", "unit", "value_string"])
        # Create rename dict and replace cat values to numerical values so the column contains one type of values
        rename_values = {
            "Risiko +": 1,
            "kein Risiko": 0,
            "Hohes Risiko ++": 2,
            "Sehr hohes Risiko +++": 3,
            "n.e.": 0,
        }
        scores_preprocessed = scores_preprocessed.replace({"overall_score": rename_values})
        # Remove nu-desc
        scores_preprocessed.overall_score = scores_preprocessed.overall_score.str.replace("(Nu-Desc)", "")
        # Convert column to numeric
        scores_preprocessed.overall_score = pd.to_numeric(scores_preprocessed.overall_score)
        # Create base for the extended scores
        scores_extended = scores_preprocessed
        # Drop this column to allow for saving
        scores_preprocessed = drop_if_in(scores_preprocessed, "value_decimal")
        scores_preprocessed = scores_preprocessed[scores_preprocessed["overall_score"].notna()]
        scores_preprocessed = scores_preprocessed[scores_preprocessed["datetime"].notna()]
        scores_preprocessed.rename(columns={"score_group": "type"}, inplace=True)
        scores_preprocessed.groupby("id").get_group(1).sort_values(by="datetime")
        scores_preprocessed = convert_to_utc(scores_preprocessed, "datetime", current_timezone)
        save_preprocessed(scores_preprocessed, "scores", output)

        # ### Scores Extended
        # We explode the parts that make up the score to gain more granular data
        # Explode scores (create a separate row for the measurement, value tuples)
        scores_exploded = scores_extended.explode("value_decimal")
        # scores_exploded["c_value_decimal"].value_counts()
        scores_exploded = scores_exploded[scores_exploded["value_decimal"].notna()]
        # Split the type, value tuples
        scores_exploded[["measurement", "value"]] = scores_exploded["value_decimal"].tolist()
        scores_exploded = scores_exploded.drop(columns=["value_decimal"])  # Drop original column
        scores_exploded = convert_to_utc(scores_exploded, "datetime", current_timezone)

        save_preprocessed(scores_exploded, "scores_extended", output)

    @staticmethod
    def medication(medication, output):
        medication.rename(
            columns={
                "order_start": "datetime_order_start",
                "order_end": "datetime_order_end",
                "application_start": "datetime_application_start",
                "application_end": "datetime_application_end",
            },
            inplace=True,
        )

        for item in [
            "datetime_order_start",
            "datetime_order_end",
            "datetime_application_start",
            "datetime_application_end",
        ]:
            medication[item] = medication[item].astype("datetime64[s]")
        medication["datetime"] = pd.to_datetime(medication["datetime_application_start"])
        medication["type"] = medication["generic_name"]
        medication["value"] = medication["amount_total"]
        medication = convert_to_utc(medication, "datetime", current_timezone)
        save_preprocessed(medication, "medication", output)

    @staticmethod
    def observations(observations, output):
        observations = observations.drop(columns="falnr")
        # Because there can be multiple values per c_value_decimal column
        observations_exploded = observations.explode("value_decimal")
        # observations_exploded[['type', 'value']] = observations_exploded['c_value_decimal'].apply(pd.Series)
        observations_exploded[["type", "value"]] = observations_exploded[
            "value_decimal"
        ].tolist()  # Split the type, value tuples
        dropcolumns = ["value_decimal", "value_string", "p_year", "p_month"]
        observations_exploded = observations_exploded.drop(
            columns=[col for col in observations_exploded if col in dropcolumns]
        )
        # groups = data.groupby('type')
        # Based on: https://github.com/christophriepe/cassandra-retro/blob/main/process_intra.ipynb
        rename_values = {
            "Blutdruck Diastolisch": "bp_dia",
            "Blutdruck Systolisch": "bp_sys",
            "PAT_ANAE_SEDLINE": "sedline",
            "Puls": "hr",
            "beat_mess_AMV": "rmv",
            "beat_mess_FiO2": "fio2",
            "beat_mess_Frequenz_AF": "rr",
            "beat_mess_IntrPEEP": "vent_peep",
            "beat_mess_Kapnometrie_etCO2": "capno_et_co2",
            "beat_mess_Spitzendruck_Ppeak": "vent_p_peak",
            #  'beat_mess_exp_Des':'exp_des'
            "beat_mess_exp_Lachgas": "exp_no",
            "beat_mess_exp_Sevo": "exp_sevo",
            #  'beat_mess_pulmon_compl':'pulmon_compl'
            "vital_AF": "rr",
            "vital_HF": "hr",
            "vital_SaO2": "sao2",
            "vital_T_K": "temp",
            "vital_T_K2": "temp",
            "vital_ZVD": "cvd",
            # Added values
            "vital_SaO2_2": "sao2",
            "Score_SOFA_Wert": "sofa",
            "Score_SOFA_Creatinine": "crea_sofa",
            "Score_SOFA_PaO2": "pao2_sofa",
            "Score_SOFA_Bilirubin": "bi_sofa",
            "Score_SOFA_Thrombo": "thromb_sofa",
            "Score_SOFA_GCS": "gcs_sofa",
            "Score_SOFA_Hypotension": "hypo_sofa",
            "Patient_Gewicht": "weight",
            "Patient_Groesse": "height",
        }
        observations_exploded.replace({"type": rename_values}, inplace=True)
        # Rename blood pressure measurements for invasive/non-invasive, diastolic/systolic/mean
        mask = observations_exploded["type"].isin(
            ["bloodpressure_mean", "bloodpressure_systolic", "bloodpressure_diastolic"]
        )
        sel = observations_exploded[mask]
        sel.loc[(sel.type == "bloodpressure_mean") & (sel.type == "vital_NBP"), "type"] = "bp_mean_ninv"
        sel.loc[(sel.type == "bloodpressure_mean") & (sel.type == "vital_IBP"), "type"] = "bp_mean_inv"
        sel.loc[(sel.type == "bloodpressure_systolic") & (sel.type == "vital_NBP"), "type"] = "bp_sys_ninv"
        sel.loc[(sel.type == "bloodpressure_systolic") & (sel.type == "vital_IBP"), "type"] = "bp_sys_inv"
        sel.loc[(sel.type == "bloodpressure_diastolic") & (sel.type == "vital_NBP"), "type"] = "bp_dia_ninv"
        sel.loc[(sel.type == "bloodpressure_diastolic") & (sel.type == "vital_IBP"), "type"] = "bp_dia_inv"

        observations_exploded.loc[mask] = sel

        # Drop empty
        observations_exploded["type"] = observations_exploded["type"].replace("", np.nan)
        observations_exploded.dropna(subset=["type"], inplace=True)
        observations_preprocessed = observations_exploded

        # Get rid of messy unit data for now
        observations_preprocessed = drop_if_in(observations_preprocessed, ["unit", "meta"])
        observations_preprocessed.rename(columns={"timestamp_utc": "datetime"}, inplace=True)
        observations_preprocessed.value = pd.to_numeric(observations_preprocessed.value)

        observations_preprocessed = convert_to_utc(observations_preprocessed, "datetime", current_timezone)
        save_preprocessed(observations_preprocessed, "observations", output)

    @staticmethod
    def fluid_balance(fluid_balance, output):
        # ## Fluid Balance
        # We currently drop the metadata about which fluids are in the intake and output as we need domain knowledge input
        # about the important input/outputs. Fluid extended is supposed to contain the granular information.
        fluid_balance.rename(
            columns={
                "h_date_time": "datetime",
                "sum_output": "sum_output",
                "sum_input": "sum_input",
                "balance": "balance",
            },
            inplace=True,
        )
        fluid_balance_reduced = drop_if_in(fluid_balance, ["falnr", "value_decimal", "direction"])
        fluid_balance_melted = pd.melt(
            fluid_balance_reduced,
            id_vars=["id", "datetime"],
            var_name="type",
            value_name="value",
            value_vars=["sum_output", "sum_input", "balance"],
        )
        fluid_balance_melted["value"] = pd.to_numeric(fluid_balance_melted["value"])
        fluid_balance_melted = convert_to_utc(fluid_balance_melted, "datetime", current_timezone)
        save_preprocessed(fluid_balance_melted, "fluid_balance", output)

        # ### Fluid Balance Extended
        # Unpack and parse the c_value_decimal column to gather more granular information about the fluid balance.
        # Work in progress to create preprocessed values.
        # fluid_balance = fluid_balance.pivot(index="c_patnr", columns="c_katalog_leistungtext", values="c_wert")
        pd.melt(fluid_balance, id_vars=["id", "datetime"], value_vars=["value_decimal"])
        exploded_fluid = fluid_balance.explode("value_decimal")

        # Split the key-value pairs into two separate columns
        exploded_fluid[["key", "value"]] = exploded_fluid["value_decimal"].apply(
            lambda x: pd.Series([x[0], x[1]])
        )
        exploded_fluid.key.unique().tolist()

        import constants.fluid_balance as fb

        # Replace the keys with the translated values
        exploded_fluid.replace({"key": fb.drug_translation}, inplace=True)
        exploded_fluid = drop_if_in(exploded_fluid, ["falnr", "value_decimal", "direction", "balance"])

        exploded_fluid = convert_to_utc(exploded_fluid, "datetime", current_timezone)
        save_preprocessed(exploded_fluid, "fluid_balance_extended", output)


if __name__ == "__main__":
    main()
