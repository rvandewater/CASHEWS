import logging
import time
import numpy as np
import pandas as pd
from general import create_parser, general_preprocessing, load_data, save_preprocessed
from utils import convert_to_utc, current_timezone, drop_if_in


# # Basic preprocessing for ishmed data
def main():
    input, output, db_format = create_parser("ishmed")
    prefix = db_format
    ishmed_names = [
        f"{prefix}_ishmed_procedure",
        f"{prefix}_ishmed_patient",
        f"{prefix}_ishmed_labor",
        f"{prefix}_ishmed_fall",
        f"{prefix}_ishmed_diagnose",
        f"{prefix}_ishmed_bewegung",
    ]

    start = time.time()
    ishmed_dict = load_data(input, ishmed_names)
    ishmed_dict = general_preprocessing(ishmed_dict)

    ISHMEDPreprocessing.procedure(ishmed_dict[f"{prefix}_ishmed_procedure"], output)
    ISHMEDPreprocessing.case(ishmed_dict[f"{prefix}_ishmed_fall"], output)
    # ISHMEDPreprocessing.movement(ishmed_dict[f"{prefix}_ishmed_bewegung"], output)
    ISHMEDPreprocessing.patient_details(ishmed_dict[f"{prefix}_ishmed_patient"], output)
    ISHMEDPreprocessing.lab_values(ishmed_dict[f"{prefix}_ishmed_labor"], output)
    ISHMEDPreprocessing.diagnosis(ishmed_dict[f"{prefix}_ishmed_diagnose"], output)
    end = time.time()
    logging.info(f"Preprocessing took {format(end - start, '.2f')} seconds")


class ISHMEDPreprocessing:

    @staticmethod
    def procedure(input, output):
        # ## Procedure
        # @ Axel: map procedure code to description

        # What to do with the procedures? They are coded according to a certain catalogue and there can be 100 per patient? I would assume cutting them until 2020 and then perhaps taking the 30 most invasive procedures (that happened before the complication) as columns?

        procedure = input

        # procedure.procedure_code.value_counts()

        procedure.rename(
            columns={
                "lfd_prozedur": "procedure_number",  # check if correct
                "lfd_bewegung": "movement",
                "prozedur_begin": "datetime",
                "prozedur_ende": "end",
                "prozedur_code": "procedure_code",
                "prozedur_katalog": "procedure_codebook",
                "localisation": "location",  # check if correct
            },
            inplace=True,
        )
        procedure = drop_if_in(procedure, ["falnr"])

        procedure["duration"] = procedure.end - procedure.datetime
        save_preprocessed(procedure, "procedure_ext", output)
        # Extract surgery areas
        proc_filt = procedure.loc[procedure["procedure_code"].str.startswith("5-")].copy()

        def op_code_startswith(df, code):
            return df["procedure_code"].str.startswith(code)

        proc_filt = procedure.loc[procedure["procedure_code"].str.startswith("5-")].copy()
        # Esophagus
        val = 0
        codes = ["5-423", "5-424", "5-425", "5-426"]
        for item in codes:
            proc_filt.loc[op_code_startswith(proc_filt, item), "procedure_system"] = val

        # Stomach
        val = 1
        codes = ["5-434", "5-435", "5-436", "5-437", "5-438"]
        for item in codes:
            proc_filt.loc[op_code_startswith(proc_filt, item), "procedure_system"] = val

        # Intestine
        val = 2
        codes = ["5-454", "5-465", "5-466", "5-455", "5-456", "5-484", "5-485"]
        for item in codes:
            proc_filt.loc[op_code_startswith(proc_filt, item), "procedure_system"] = val

        # Liver
        val = 3
        codes = ["5-502", "5-501"]
        for item in codes:
            proc_filt.loc[op_code_startswith(proc_filt, item), "procedure_system"] = val

        # Pancreas
        val = 4
        codes = ["5-524", "5-525"]
        for item in codes:
            proc_filt.loc[op_code_startswith(proc_filt, item), "procedure_system"] = val
        surgery = proc_filt[~proc_filt["procedure_system"].isna()]
        surgery[surgery.datetime > pd.Timestamp("2021-01-01")].groupby("id").nunique()
        surgery = convert_to_utc(surgery, "datetime", timezone=current_timezone)
        save_preprocessed(surgery, "procedure", output)

    @staticmethod
    def case(input, output):
        # ## Case

        case = input
        case = case.rename(
            columns={
                "fall_status": "case_status",  # Check
                "aufnahme": "intake_date",
                "entlassung": "release_date",
                "aufnahmeart": "intake_type",
                "entlassungsart": "release_type",
                "age": "age",
                "dod": "date_of_death",
            }
        )
        # DOD means something different, but what?
        dropcolumns = ["falnr", "entlassung_planstatus", "aufnahme_planstatus"]
        case = drop_if_in(case, dropcolumns)

        case.head()

        print(max(case.intake_date))
        case = convert_to_utc(case, "intake_date", current_timezone)
        case = convert_to_utc(case, "release_date", current_timezone)
        save_preprocessed(case, "case", output)
        # Create case_reduced which should limit causal leakage
        case_reduced = drop_if_in(case, ["release_date", "date_of_death", "release_type"])
        save_preprocessed(case_reduced, "case_reduced", output)

    @staticmethod
    def movement(input, output):
        # ## Movement
        movement = input

        movement.rename(
            columns={
                "bewegungstyp": "movement_type",  # check if correct
                "bewegungsart": "movement_method",
                "bewegungskategory": "movement_category",
                "begin": "datetime_start",
                "ende": "datetime_end",
                "laufende_nummer": "running_number",
                "zimmer": "room",
                "bett": "bed",
                "fachliche_oe": "clinic",
                "pflege_oe": "nursing_station",
            },
            inplace=True,
        )
        movement = drop_if_in(movement, ["falnr"])
        movement  # Axel: can we make some kind of distinction between movements? Do we have a codebook that distinguishes what kind of movement is happening?

        movement.clinic.value_counts().head(50)

        print(f"Earliest value:{movement.datetime_start.min()}")
        print(f"Latest value:{movement.datetime_start.max()}")

        # movement.sort_values(by="datetime_end", ascending=False).head(100)
        movement.loc[movement["datetime_end"] == pd.Timestamp("9999-12-31 22:59:59"), "datetime_end"] = (
            movement.loc[movement["datetime_end"] == pd.Timestamp("9999-12-31 22:59:59"), "datetime_start"]
        )
        standard_time = pd.Timestamp("2022-01-01 0:0:0")
        movement = movement[movement["datetime_start"] > standard_time]
        # Correct for wrong entries
        movement.loc[movement["datetime_end"] < standard_time, "datetime_end"] = None
        movement.loc[movement["datetime_end"] > movement["datetime_start"]] = None
        movement["duration"] = movement.datetime_end - movement.datetime_start
        movement.sort_values(by="datetime_end", ascending=False)

        movement.movement_method.value_counts()

        movement["datetime_start"] = movement["datetime_start"].astype("datetime64[s]")
        movement["datetime_end"] = movement["datetime_end"].astype("datetime64[s]")
        movement = convert_to_utc(movement, "datetime_start", current_timezone)
        movement = convert_to_utc(movement, "datetime_end", current_timezone)

        save_preprocessed(movement, "movement", output)

    # Feature extraction per method, category, type, clinic
    @staticmethod
    def patient_details(input, output):
        patient_details = input
        patient_details.rename(
            columns={"gender": "sex", "birthdate": "birthdate", "datetimeofdeath": "datetime_of_death"},
            inplace=True,
        )
        death_endpoint = patient_details
        death_endpoint["label"] = ~death_endpoint["datetime_of_death"].isna()
        death_endpoint.sort_values(by="id", ascending=True, inplace=True)
        death_endpoint.reset_index(inplace=True)
        death_endpoint = convert_to_utc(death_endpoint, "datetime_of_death", current_timezone)
        save_preprocessed(death_endpoint, "in_hospital_death", output)
        save_preprocessed(patient_details, "patient_details", output)

    @staticmethod
    def lab_values(input, output):

        def append(
            name: str, data: pd.DataFrame, lab_data: pd.DataFrame, types: [str], units: [str] = []
        ) -> pd.DataFrame:
            subsets = []
            for type in types:
                type_data = data[data["type"] == type]
                logging.debug(f'{type}: {len(type_data)} {type_data["unit"].unique()}')
                subsets.append(type_data)

            subset = pd.concat(subsets, ignore_index=True)
            if len(units) > 0:
                subset = subset[subset["unit"].isin(units)]

            subset.loc[:, "type"] = name
            logging.debug(f'Total: {len(subset)} {subset["unit"].unique()}')

            return pd.concat([lab_data, subset], ignore_index=True)

        data = input

        data.rename(
            columns={
                "katalog_leistungtext": "type",
                "wert": "value",
                "wert_einheit": "unit",
                "wert_timestamp": "timestamp",
                "patnr": "id",
            },
            inplace=True,
        )

        # Replace unrealistic values with the timestamp of the document to include them in the data
        data["timestamp"] = np.where(
            data["timestamp"] < pd.Timestamp("2020-01-01"), data["dokument_timestamp"], data["timestamp"]
        )
        lab_data = pd.DataFrame(columns=["id", "timestamp", "type", "value"])

        # 25 OH Vitamin D3 = 25 OH Vitamin B3
        lab_data = append(
            "vd25",
            data,
            lab_data,
            ["25-Hydroxy-Vitamin D3", "25-OH-Vitamin D3", "25-OH-Vitamin D3 Se", "1.25-OH-Vitamin D3 Se"],
        )

        # Base Excess = Basenüberschuss
        lab_data = append("be", data, lab_data, ["ABE", "Base Excess", "Basenüberschuß" "SBE"])

        # Antithrombin = Antithrombin
        lab_data = append("at", data, lab_data, ["AT3", "Antithrombin", "Antithrombin  Aktivität"])

        # Albumin = Albumin
        lab_data = append(
            "alb",
            data,
            lab_data,
            ["Albumin", "Albumin (HP)", "Albumin HP", "Albumin Se", "Albumin i.Se"],
            ["g/l"],
        )

        # Alkaline Phosphatase = Alkalische Phosphatase
        lab_data = append(
            "alp",
            data,
            lab_data,
            [
                "Alk.Phosphatase HP",
                "Alk. Phosphatase (HP)",
                "Alk. Phosphatase",
                "alk.Knochenphosphatase/Ostase",
                "Alk.Phosphatase",
                "Alk.Phosphatase Se",
                "Alk. Phosphatase - neu",
            ],
        )

        # Ammonia = Ammoniak
        lab_data = append("nh3", data, lab_data, ["Ammoniak", "Ammoniak (EDTA)", "Ammoniak EDTA"])

        # Amylase = Amylase
        lab_data = append("ams", data, lab_data, ["Amylase", "Amylase HP", "Amylase Se"], ["U/l"])

        # Basophils Absolute = Basophile Absolut
        lab_data = append("baso", data, lab_data, ["Basophile absolut"])

        baso = data[(data["type"] == "Basophile") & (data["unit"] == "/nl")]
        baso.loc[:, "type"] = "baso"
        lab_data = pd.concat([lab_data, baso], ignore_index=True)

        # Basophils Relative = Basophile Relativ
        lab_data = append("baso_rel", data, lab_data, ["Basophile %"])

        baso_rel = data[(data["type"] == "Basophile") & (data["unit"] == "%")]
        baso_rel.loc[:, "type"] = "baso_rel"
        lab_data = pd.concat([lab_data, baso_rel], ignore_index=True)

        # Total Bilirubin = Gesamt Bilirubin
        lab_data = append(
            "tbil",
            data,
            lab_data,
            [
                "Bilirubin",
                "Bilirubin gesamt Se",
                "Bilirubin, gesamt",
                "Bilirubin, gesamt",
                "Bilirubin, total",
                "Bilirubin, ges.",
                "Bilirubin, total (HP)",
                "tBil",
            ],
            ["mg/dl", "mg/dL"],
        )

        # Bilirubin Direct = Bilirubin Direkt
        lab_data = append(
            "dbil",
            data,
            lab_data,
            [
                "Bilirubin direkt Se",
                "Bilirubin, conjugiert",
                "Bilirubin, direkt",
                "Bilirubin, direkt (HP)",
                "Bilirubin, direkt HP",
                "Bilirubin, ges. (DPD) ",
            ],
        )

        # Bilirubin Indirect = Bilirubin Indirekt
        lab_data = append("ibil", data, lab_data, ["Bilirubin indirekt"])

        # Total Bilirubin in drainage
        lab_data = append("tbil_drain", data, lab_data, ["ges.Bilirubin Pkt°°", "ges.Bilirubin i.Pkt°°"])
        # - = Potenzial von Wasserstoff
        lab_data = append("ph", data, lab_data, ["Blut-pH-Wert", "pH", "pH-Wert"])

        # Creatine Kinase = Kreatin Kinase
        lab_data = append(
            "ck",
            data,
            lab_data,
            ["CK", "CK (HP)", "Creatinkinase (CK)", "Creatinkinase (CK) HP", "Creatinkinase (CK) Se"],
        )

        # Creatine Kinase MB = Kreatin Kinase MB
        lab_data = append("ck_mb", data, lab_data, ["CK-MB", "CK-MB (HP)", "CK-MB HP", "CK-MB Se"])

        # Carboxyhemoglobin = Carboxyhämoglobin
        lab_data = append("cohb", data, lab_data, ["CO-Hb", "COHb"])

        # C Reactive Protein = C Reaktives Protein
        lab_data = append(
            "crp", data, lab_data, ["CRP HP", "CRP", "CRP (HP)", "hsCRP", "CRP2", "CRP Se", "CRPhs"], ["mg/l"]
        )

        # Calcium = Kalzium
        lab_data = append(
            "ca", data, lab_data, ["Ca++", "Calcium", "Calcium (HP)", "Calcium Se"], ["mmol/L", "mmol/l"]
        )

        # Chloride = Chlorid
        lab_data = append("cl", data, lab_data, ["Chlorid", "Chlorid (HP)", "Chlorid Se", "Cl-"])

        # Total Cholesterol = Gesamt Cholesterin
        lab_data = append(
            "tc",
            data,
            lab_data,
            ["Cholesterin", "ges.Cholesterin", "ges.Cholesterin HP", "ges.Cholesterin Se"],
        )

        # Creatinine = Kreatinin
        lab_data = append(
            "cr",
            data,
            lab_data,
            [
                "Creatinin",
                "Creatinin (enz)",
                "Creatinin (enzymat.)",
                "Kreatinin",
                "Kreatinin (JaffÃ©)",
                "Kreatinin (JaffÃ©) (HP)",
                "Kreatinin (JaffÃ©) HP",
                "Kreatinin (JaffÃ©) Se",
                "Kreatinin (enzym.)",
                "Kreatinin (enzym.) HP",
                "Kreatinin (enzym.) Se",
                "Kreatinin (Jaffé) HP",
                "Kreatinin (Jaffé) (HP)",
                "Kreatinin",
                "Kreatinin (Jaffé)",
                "Kreatinin (enzym.) HP",
                "Kreatinin (enzym.) (HP)",
                "KREATININ",
                "Kreatinin (enzym.)",
                "Kreatinin (Jaffé) Se",
                "Kreatinin Clearance",
                "Protein/Kreatinin",
            ],
            ["mg/dl"],
        )
        # Creatinine drainage
        # lab_data = append('cr_drain', data, lab_data, ['Kreatinin Pkt°°', 'Kreatinin i.Pkt°°'])

        # Cystatin C = Cystatin C
        lab_data = append("cys_c", data, lab_data, ["Cystatin C", "Cystatin C HP", "Cystatin C Se"], ["mg/l"])

        # D Dimer = D Dimere
        lab_data = append("d_dim", data, lab_data, ["D-Dimer", "D-Dimere"])

        # Iron = Eisen
        lab_data = append("fe", data, lab_data, ["Eisen", "Eisen (HP)", "Eisen Se"])

        # Eosinophils = Eosinophile
        lab_data = append("eos", data, lab_data, ["Eosinophile absolut"])

        eos = data[(data["type"] == "Eosinophile") & (data["unit"] == "/nl")]
        eos.loc[:, "type"] = "eos"
        lab_data = pd.concat([lab_data, eos], ignore_index=True)

        # Eosinophils Relative = Eosinophile Relativ
        lab_data = append("eos_rel", data, lab_data, ["Eosinophile %"])

        eos_rel = data[(data["type"] == "Eosinophile") & (data["unit"] == "%")]
        eos_rel.loc[:, "type"] = "eos_rel"
        lab_data = pd.concat([lab_data, eos_rel], ignore_index=True)

        # Erythroblasts = Erythroblasten
        lab_data = append("ebl", data, lab_data, ["Erythroblasten absolut"])

        ebl = data[(data["type"] == "Erythroblasten") & (data["unit"] == "/nl")]
        ebl.loc[:, "type"] = "ebl"
        lab_data = pd.concat([lab_data, ebl], ignore_index=True)

        # Erythroblasts Relative = Erythroblasten Relativ
        lab_data = append("ebl_rel", data, lab_data, ["Erythroblasten %"])

        ebl_rel = data[(data["type"] == "Erythroblasten") & (data["unit"] == "%")]
        ebl_rel.loc[:, "type"] = "ebl_rel"
        lab_data = pd.concat([lab_data, ebl_rel], ignore_index=True)

        # Erythrocytes = Erythrozyten
        lab_data = append("rbc", data, lab_data, ["Erythrozyten"], ["/pl"])
        # Fraction of Inspired Oxygen = Inspiratorische Sauerstofffraktion
        lab_data = append("fio2", data, lab_data, ["FIO2"])
        # Ferritin = Ferritin
        lab_data = append("fer", data, lab_data, ["Ferritin", "Ferritin HP", "Ferritin SE", "Ferritin Se"])
        # Fibrinogen = Fibrinogen
        lab_data = append("fg", data, lab_data, ["Fibrinogen"], ["g/l"])
        # Schistocytes = Fragmentozyten
        lab_data = append("schisto", data, lab_data, ["Fragmentozyten"], ["%"])
        # Gamma Glutamyltransferase = Gamma Glutamyltransferase
        lab_data = append(
            "ggt", data, lab_data, ["GGT", "GGT (HP)", "gamma-GT", "gamma-GT HP", "gamma-GT Se"]
        )
        # Glutamate Dehydrogenase = Glutamat Dehydrogenase
        lab_data = append("gdh", data, lab_data, ["GLDH", "GLDH HP", "GLDH Se"])
        # Glucose = Glukose
        lab_data = append(
            "glu", data, lab_data, ["GLU", "Glu", "Glucose", "Glucose HP", "Glucose Se"], ["mg/dl", "mg/dL"]
        )
        # ASAT (GOT) = ASAT (GOT)
        lab_data = append(
            "asat",
            data,
            lab_data,
            ["GOT (AST)", "GOT (AST) (HP)", "GOT (AST) HP", "GOT (AST) Se", "GOT (ASAT)"],
        )
        # ALAT (GPT) = ALAT (GPT)
        lab_data = append(
            "alat",
            data,
            lab_data,
            ["GPT (ALT)", "GPT (ALT) (HP)", "GPT (ALT) HP", "GPT (ALT) Se", "GPT (ALAT) "],
        )
        # Bicarbonate = Bikarbonat
        lab_data = append(
            "hco3",
            data,
            lab_data,
            [
                "HCO3-",
                "HCO3",
                "HCO3 ven.",
                "SBC",
                "Standard Bicarbonat",
                "Standardbicarbonat",
                "aktuelles Bicarbonat",
                "Bicarbonat Se°",
            ],
        )
        # High Density Lipoprotein = HDL Cholesterin
        lab_data = append(
            "hdl", data, lab_data, ["HDL-Cholesterin", "HDL-Cholesterin HP", "HDL-Cholesterin Se"]
        )
        # Deoxyhemoglobin = Desoxyhämoglobin
        lab_data = append("hhb", data, lab_data, ["HHb"])
        # Haptoglobin = Haptoglobin
        lab_data = append("hp", data, lab_data, ["Haptoglobin", "Haptoglobin HP", "Haptoglobin Se"], ["g/l"])
        # Urea = Harnstoff
        lab_data = append(
            "urea", data, lab_data, ["Harnstoff", "Harnstoff (HP)", "Harnstoff HP", "Harnstoff Se"], ["mg/dl"]
        )
        # Urea Drain = Harnstoff Drain
        # lab_data = append('urea_drain', data, lab_data, ['Harnstoff Pkt°°', 'Harnstoff i.Pkt°°'])
        # Uric Acid = Harnsäure
        lab_data = append(
            "ua",
            data,
            lab_data,
            [
                "Harnsäure HP",
                "Harnsäure (HP)",
                "Harnsäure",
                "Harnsäurekristalle",
                "Harnsäure SU",
                "Harnsäure SU/d",
                "Harnsäure Se",
            ],
            ["mg/dl"],
        )
        # Hemoglobin = Hämoglobin
        lab_data = append(
            "hb", data, lab_data, ["Hb", "Hämoglobin", "freies Hämoglobin HP", "tHb"], ["g/dl", "g/dL"]
        )
        # Glycated Hemoglobin = Glykosyliertes Hämoglobin
        lab_data = append("hba1c", data, lab_data, ["HbA1c", "HbA1c (EDTA)"])
        # Hematocrit = Hämatokrit
        lab_data = append(
            "hct", data, lab_data, ["Hct", "Hämatokrit", "Hämatokrit (l/l)", "BGA-Hämatokrit"], ["%"]
        )
        # I/T Ratio = I/T Quotient
        lab_data = append("it_ratio", data, lab_data, ["I/T Quotient maschinell"])
        # International Normalized Ratio = International Normalized Ratio
        lab_data = append("inr", data, lab_data, ["INR", "TPZ-INR"])
        # Immature Platelet Fraction = Unreife Thrombozytenfraktion
        lab_data = append("ipf", data, lab_data, ["Immature PlÃ¤ttchenfraktion"])
        # Immunoglobulin A = Immunoglobulin A
        lab_data = append(
            "iga", data, lab_data, ["Immunglobulin A", "Immunglobulin A HP", "Immunglobulin A Se"], ["g/l"]
        )
        # Immunoglobulin E = Immunoglobulin E
        lab_data = append(
            "ige", data, lab_data, ["Immunglobulin E", "Immunglobulin E HP", "Immunglobulin E Se"], ["kU/l"]
        )
        # Immunoglobulin G = Immunoglobulin G
        lab_data = append(
            "igg", data, lab_data, ["Immunglobulin G", "Immunglobulin G HP", "Immunglobulin G Se"], ["g/l"]
        )
        # Immunoglobulin M = Immunoglobulin M
        lab_data = append(
            "igm", data, lab_data, ["Immunglobulin M", "Immunglobulin M HP", "Immunglobulin M Se"], ["g/l"]
        )
        # Potassium = Kalium
        lab_data = append(
            "k",
            data,
            lab_data,
            ["K+", "Kalium", "Kalium HP", "Kalium Se", "KALIUM(BG) ven.", "KALIUM(BG)", "BGA-Kalium"],
            ["mmol/L", "mmol/l"],
        )
        # Lactate Dehydrogenase = Laktatdehydrogenase
        lab_data = append("ldh", data, lab_data, ["LDH", "LDH (HP)", "LDH HP", "LDH Se"])
        # Low Density Lipoprotein = LDL Cholesterin
        lab_data = append(
            "ldl", data, lab_data, ["LDL-Cholesterin", "LDL-Cholesterin HP", "LDL-Cholesterin Se"]
        )
        # Lactate = Laktat
        lab_data = append("lac", data, lab_data, ["Lac", "Lactat", "Laktat", "LACTAT(BG) ven."])
        # Leukocytes = Leukozyten
        lab_data = append("wbc", data, lab_data, ["Leukozyten"], ["/nl"])
        # Lipase = Lipase
        lab_data = append(
            "lps", data, lab_data, ["Lipase", "Lipase (HP)", "Lipase HP", "Lipase Se", "LIPASE"]
        )
        # Lipase Drain = Lipase Drain
        lab_data = append("lps_drain", data, lab_data, ["Lipase Pkt°°", "Lipase i.Pkt°°"])
        # Lymphocytes = Lymphocytes
        lab_data = append("lym", data, lab_data, ["Lymphozyten abs.", "Lymphozyten absolut"], ["/nl"])
        lym = data[(data["type"] == "Lymphozyten") & (data["unit"] == "/nl")]
        lym.loc[:, "type"] = "lym"
        lab_data = pd.concat([lab_data, lym], ignore_index=True)
        # Lymphocytes Relative = Lymphocytes Relativ
        lab_data = append("lym_rel", data, lab_data, ["Lymphozyten %", "Lymphozyten rel."])
        lym_rel = data[(data["type"] == "Lymphozyten") & (data["unit"] == "%")]
        lym_rel.loc[:, "type"] = "lym_rel"
        lab_data = pd.concat([lab_data, lym_rel], ignore_index=True)
        # Mean Corpuscular Hemoglobin = Mittleres Korpuskulares Hämoglobin
        lab_data = append("mch", data, lab_data, ["MCH"])
        # Mean Corpuscular Hemoglobin Concentration = Mittlere Korpusukuläre Hämoglobin Konzentration
        lab_data = append("mchc", data, lab_data, ["MCHC"])
        # Mean Corpuscular Volume = Mittleres Korpuskuläres Volumen
        lab_data = append("mcv", data, lab_data, ["MCV"])
        # Mean Platelet Volume = Mittleres Thrombozytenvolumen
        lab_data = append("mpv", data, lab_data, ["MPV"])
        # Magnesium = Magnesium
        lab_data = append("mg", data, lab_data, ["Magnesium", "Magnesium (HP)", "Magnesium Se"])
        # Methemoglobin = Methämoglobin
        lab_data = append("methb", data, lab_data, ["MetHb"])
        # Monocytes = Monocytes
        lab_data = append("mono", data, lab_data, ["Monozyten abs.", "Monozyten absolut"])
        mono = data[(data["type"] == "Monozyten") & (data["unit"] == "/nl")]
        mono.loc[:, "type"] = "mono"
        lab_data = pd.concat([lab_data, mono], ignore_index=True)
        # Monocytes Relative = Monocytes Relativ
        lab_data = append("mono_rel", data, lab_data, ["Monozyten %", "Monozyten rel."])
        mono_rel = data[(data.loc[:, "type"] == "Monozyten") & (data["unit"] == "%")]
        mono_rel.loc[:, "type"] = "mono_rel"
        lab_data = pd.concat([lab_data, mono_rel], ignore_index=True)
        # Myelocytes = Myelozyten
        lab_data = append("myelo", data, lab_data, ["Myelozyten"])
        # Myoglobin = Myoglobin
        lab_data = append("mb", data, lab_data, ["Myoglobin", "Myoglobin HP", "Myoglobin Se"], ["Âµg/l"])
        # N Terminal Pro B Type Natriuretic Peptide = N Terminal Pro B Type Natriuretic Peptide
        lab_data = append(
            "nt_probnp", data, lab_data, ["NT pro BNP", "NT-pro BNP", "NT-pro BNP (HP)"], ["ng/l"]
        )
        # Sodium = Natrium
        lab_data = append(
            "na",
            data,
            lab_data,
            [
                "Na+",
                "Natrium",
                "Natrium HP",
                "Natrium Se",
                "BGA-Natrium",
                "NATRIUM",
                "NATRIUM(BG)",
                "NATRIUM(BG) ven.",
            ],
            ["mmol/L", "mmol/l"],
        )
        # Neutrophils = Neutrophile
        lab_data = append("pmn", data, lab_data, ["Neutrophile absolut"])
        pmn = data[(data["type"] == "Neutrophile") & (data["unit"] == "/nl")]
        pmn.loc[:, "type"] = "pmn"
        lab_data = pd.concat([lab_data, pmn], ignore_index=True)
        # Neutrophils Relative = Neutrophile Relativ
        lab_data = append("pmn_rel", data, lab_data, ["Neutrophile %"])
        pmn_rel = data[(data["type"] == "Neutrophile") & (data["unit"] == "%")]
        pmn_rel.loc[:, "type"] = "pmn_rel"
        lab_data = pd.concat([lab_data, pmn_rel], ignore_index=True)
        # Oxygen Saturation = Sauerstoffsättigung
        lab_data = append(
            "so2",
            data,
            lab_data,
            [
                "O2-Sättigung",
                "sO2",
            ],
        )
        # Oxyhemoglobin = Oxyhämoglobin
        lab_data = append("o2hb", data, lab_data, ["O2Hb"])
        # Phosphorus = Phosphor
        lab_data = append("p", data, lab_data, ["Phosphor, anorg."], ["mmol/l"])
        # Procalcitonin = Procalcitonin
        lab_data = append(
            "pct",
            data,
            lab_data,
            ["Procalcitonin", "Procalcitonin (HP)", "Procalcitonin HP", "Procalcitonin Se", "PCT"],
        )
        # Protein = Protein
        lab_data = append("pro", data, lab_data, ["Protein", "Protein HP"], ["g/l"])
        # Pseudocholinesterase = Pseudocholinesterase
        lab_data = append(
            "pche",
            data,
            lab_data,
            [
                "PCHE",
                "PCHE (HP)",
                "Pseudo-Cholinesterase",
                "Pseudo-Cholinesterase HP",
                "Pseudo-Cholinesterase Se",
                "PCHE (37°)",
            ],
        )
        # Quick Value = Quick Wert
        lab_data = append("quick", data, lab_data, ["Quick (TPZ)", "TPZ-Wert"])
        # Red Cell Distribution Width = Erythrozytenverteilungsbreite
        lab_data = append("rdw", data, lab_data, ["RDW", "RDW-CV"])
        # Reticulocytes = Retikulozyten
        lab_data = append("rtic", data, lab_data, ["Retikulozyten"], ["/nl"])
        # Temperature = Temperatur
        lab_data = append("temp", data, lab_data, ["T", "Temperatur"])

        # Prothrombin Time = Thromboplastinzeit
        # lab_data = append('pt', data, lab_data, ['TPZ-Wert'])

        # Thyroid Stimulating Hormone = Schilddrüsenstimulierendes Hormon
        lab_data = append(
            "tsh",
            data,
            lab_data,
            [
                "TSH",
                "TSH bas.",
                "TSH bas. Se",
                "TSH bas. i.Se",
                "TSH basal",
                "TSH basal (HP)",
                "TSH basal Se",
            ],
        )
        # Platelets = Thrombozyten
        lab_data = append("plt", data, lab_data, ["Thrombozyten"])
        # Transferrin = Transferrin
        lab_data = append(
            "trans", data, lab_data, ["Transferrin", "Transferrin HP", "Transferrin Se"], ["g/l"]
        )
        # Transferrin Saturation = Transferrinsättigung
        lab_data = append(
            "ts",
            data,
            lab_data,
            [
                "Transferrin-Sättigung Se",
                "Transferrin-Sättigung HP",
                "Transferrin-Sättigung",
                "Transferrinsättigung",
            ],
        )
        # Total Triglycerides = Gesamt Triglyceride
        lab_data = append("tg", data, lab_data, ["Triglyceride", "Triglyceride HP", "Triglyceride Se"])
        # Triglycerides Drain = Triglyceride Drain
        # lab_data = append('tg_drain', data, lab_data, ['Triglyceride Pkt°°', 'Triglyceride i.Pkt°°'])
        #   = Partielle Thromboplastinzeit
        lab_data = append("aptt", data, lab_data, ["aPTT", "aPTT Pathromtin SL"])
        # Phosphate = Phosphat
        lab_data = append("po4", data, lab_data, ["anorg. PO4  HP", "anorg. PO4  Se"])
        # Carbon Dioxide Partial Pressure = Kohlendioxidpartialdruck
        lab_data = append("pco2", data, lab_data, ["pCO2", "pCO2(T)"])
        # Oxygen Partial Pressure = Sauerstoffpartialdruck
        lab_data = append("po2", data, lab_data, ["pO2"])
        # Immature Granulocytes = Unreife Granulozyten
        lab_data = append("ig", data, lab_data, ["unreife Granulozyten absolut"])
        ig = data[(data["type"] == "unreife Granulozyten") & (data["unit"] == "/nl")]
        ig.loc[:, "type"] = "ig"
        lab_data = pd.concat([lab_data, ig], ignore_index=True)
        # Immature Granulocytes Relative = Unreife Granulozyten Relativ
        lab_data = append("ig_rel", data, lab_data, ["unreife Granulozyten %"])
        ig_rel = data[(data.loc[:, "type"] == "unreife Granulozyten") & (data["unit"] == "%")]
        ig_rel.loc[:, "type"] = "immature_granulocytes_relative"
        lab_data = pd.concat([lab_data, ig_rel], ignore_index=True)

        lab_data["value"] = lab_data["value"].apply(pd.to_numeric, errors="coerce")
        lab_data["value"] = lab_data["value"].astype(float)
        # id id numeric
        # lab_data["id"] = lab_data["id"].str.replace('CASS1-', '')
        # lab_data["id"] = lab_data["id"].str.replace('Cass1-', '')
        # lab_data['id'] = lab_data['id'].astype(int)

        # event_ts
        lab_data["timestamp"] = pd.to_datetime(lab_data["timestamp"])

        # type
        lab_data["type"] = lab_data["type"].astype(str)

        # drop unit column
        lab_data.drop(columns=["unit"], inplace=True)

        # drop nan
        lab_data.dropna(inplace=True)

        # drop duplicates
        lab_data.drop_duplicates(keep="first", inplace=True)

        # sort by event_ts
        lab_data.sort_values(by=["timestamp"], inplace=True)

        lab_data = drop_if_in(lab_data, ["dokument_art", "p_year", "p_month", "month", "year"])

        lab_data.rename(
            columns={
                "labor_system": "lab_system",
                "leistungnr": "measurement_id",
                "labor_status": "lab_status",
                "timestamp": "datetime",
                "dokument_timestamp": "document_datetime",
            },
            inplace=True,
        )
        lab_data = convert_to_utc(lab_data, "datetime", current_timezone)
        save_preprocessed(lab_data, "lab_numeric", output)

    @staticmethod
    def diagnosis(input, output):
        # ## Diagnosis
        # How to process this: there is no date as far as I am aware. Does the diagnosis always happen before the complication?
        base_date = pd.Timestamp("2020-01-01 0:0:0")
        diagnosis = input
        diagnosis.rename(columns={"diagnose_timestamp": "datetime", "diagnose_1": "type"}, inplace=True)
        diagnosis = drop_if_in(diagnosis, ["falnr"])
        diagnosis = diagnosis[diagnosis["datetime"] > base_date]
        diagnosis = convert_to_utc(diagnosis, "datetime", current_timezone)
        save_preprocessed(diagnosis, "diagnosis", output)


if __name__ == "__main__":
    main()
