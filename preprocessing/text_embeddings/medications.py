import pandas as pd
from preprocessing.filters.filters import case_id_to_numeric, drop_hdl_columns


def add_prefix_to_columns(df, prefix, exclude_columns=['text', 'id', 'datetime']):
    """Add prefix to all columns except specified ones"""

    # Create mapping for column renaming
    rename_dict = {}
    for col in df.columns:
        if col not in exclude_columns:
            rename_dict[col] = f"{prefix}_{col}"

    # Rename columns
    df_renamed = df.rename(columns=rename_dict)

    return df_renamed
def remove_timezones(df):
    # Remove timezone if present
    if pd.api.types.is_datetime64tz_dtype(df['datetime']):
        df['datetime'] = df['datetime'].dt.tz_convert(None)
    else:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    return df

def load_ward_medications(cfg):
    medications_ward = pd.read_csv(cfg.embeddings.ward_medications, sep=";")
    medications_ward["cassandra1_id"] = medications_ward["cassandra1_id"].str.upper()
    medications_ward = case_id_to_numeric(medications_ward, column_name="cassandra1_id")
    medications_ward["start_time"] = pd.to_datetime(medications_ward["start_time"], errors="coerce")
    medications_ward.rename(columns={"cassandra1_id": "id", "medication_text": "text", "start_time": "datetime"},
                            inplace=True)
    medications_ward = add_prefix_to_columns(medications_ward, prefix="ward_med")
    medications_ward = remove_timezones(medications_ward)
    return medications_ward

def load_copra_medications(cfg):
    copra_medication = pd.read_parquet(cfg.embeddings.icu_medications)
    copra_medication = drop_hdl_columns(copra_medication)
    medical_terms_translation = {
        "Narkose, Sedation, Relaxation, Opiate, Schmerztherapie": "Anesthesia, Sedation, Relaxation, Opiates, Pain therapy",
        "pos. inotrope und vasoaktive Medikamente": "Positive inotropic and vasoactive medications",
        "sonstige Interna": "Other internal medications",
        "Antiinfectiva": "Anti-infectives",
        "Volumen incl. Humanalbumin/Plasmaexpander": "Volume including human albumin/plasma expanders",
        "Blutprodukte": "Blood products",
        "Ernährung parenteral, Vitamine, Ernährungszusätze, Alt": "Parenteral nutrition, vitamins, nutritional supplements, alternative",
        "inhalativ/bronchoskopisch": "Inhalation/bronchoscopic",
        "sonstiges": "Other/miscellaneous",
        "gerinnungsaktive Medikamente": "Coagulation-active medications",
        "Ernährung enteral, Vitamine, Ernährungszusätze": "Enteral nutrition, vitamins, nutritional supplements",
        "sonstige Externa AT/Klys/nasal": "Other external medications (AT/enema/nasal)",
        "sonstige Interna s.c.+i.m.": "Other internal medications subcutaneous + intramuscular",
        "sonstige Externa": "Other external medications"
    }
    copra_medication["substance_group"] = copra_medication["substance_group"].replace(medical_terms_translation)
    copra_medication["text"] = copra_medication["mixture_name"] + " " + copra_medication["drug_name"]
    copra_medication.drop(columns=["mixture_name", "drug_name", "falnr", "mixture_id", "generic_name",
                                   "datetime_order_start", "datetime_application_start", "datetime_application_end",
                                   "datetime_order_start", "datetime_order_end", "drug_id", "drug_name", "volume_unit",
                                   "type", "application_form", "application_location", "rate", "rate_unit",
                                   "concentration_unit", "amount_unit", "concentration", "volume_total", "amount_total",
                                   "application_id"],
                          inplace=True)
    copra_medication = add_prefix_to_columns(copra_medication, prefix="icu_med")
    if 'datetime' in copra_medication.columns:
        copra_medication['datetime'] = pd.to_datetime(copra_medication['datetime']).dt.tz_localize(None)
    copra_medication = remove_timezones(copra_medication)
    return copra_medication

def load_fluids(cfg):
    fluids = pd.read_parquet(cfg.embeddings.icu_fluids)
    fluids = drop_hdl_columns(fluids)
    fluids["text"] = fluids["key"]
    fluids.drop(columns=["key"], inplace=True)
    fluids = add_prefix_to_columns(fluids, prefix="icu_fluid")
    fluids = remove_timezones(fluids)
    return fluids