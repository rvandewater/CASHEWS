import math
import os
import sys
from datetime import datetime, timedelta
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re
from preprocessing.filters.filters import case_id_to_numeric
# from huggingface import AutoTokenizer, AutoModel
from faker import Faker

escape_characters_regex = r"\r|\n"
escape_characters_regex_obj = re.compile(escape_characters_regex)

def get_data_path(cfg):
    linux_path = cfg.root
    mac_path = "/Volumes/sc-proj-cc08-cassandra/"
    win_path = "T:/"

    if sys.platform == "linux":
        return linux_path  # HPC
    elif sys.platform == "darwin":
        return mac_path
    elif sys.platform == "win32":
        return win_path
    else:
        raise NotImplementedError(
            f"Unsupported platform: {sys.platform}. Please set the data path manually in the config file."
        )


def construct_path(*args):
    return os.path.join(*args)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def remove_escape_characters(text):
    return escape_characters_regex_obj.sub(" ", text)


def format_cass_id(value):
    prefix = "CASS1-"
    try:
        number = value.split(prefix)[1]
        formatted_number = number.zfill(4)  # Pad the number with leading zeros to ensure it has 4 digits
        return prefix + formatted_number
    except Exception as e:
        return f"ERROR {value} {e}"


def load_clinical_notes(cfg):
    # path of original file which was copied to temp_folder: transfer_learning\CASSANDRA\data\bot_data\processed_bot_data\texts_processed_with_range.parquet
    # df = pd.read_parquet(construct_path(root, temp_folder, "texts.parquet"))
    df = pd.read_parquet(cfg.clinical_notes.path)
    df["clinical_note"] = df["clinical_note"].apply(remove_escape_characters)
    df["clinical_note"] = df["clinical_note"].str.strip()

    df["cassandra1_id"] = df["cassandra1_id"].str.upper()
    df["cassandra1_id"] = df["cassandra1_id"].apply(format_cass_id)

    df = df[
        ["cassandra1_id", "DateTime_note", "clinical_note", "note_provider", "Stand"]
    ]  # reorder the columns
    df = case_id_to_numeric(df, "cassandra1_id")
    df.rename(columns={"cassandra1_id": "id", "DateTime_note": "datetime", "clinical_note": "text"}, inplace=True)
    return df


def load_base_data(cfg):
    df = pd.read_csv(cfg.base_data.path)  # pd.read_excel(cfg.base_data.path)

    # df["cassandra1_id"] = df["cassandra1_id"].str.upper()
    # df["cassandra1_id"] = df["cassandra1_id"].apply(format_cass_id)

    df = df[["cassandra1_id", "OP_Naht"]]

    return df


def find_overlong_notes(df, temp_tokenizer):
    result = []
    for index, row in df.iterrows():
        inputs = temp_tokenizer(row["clinical_note"], return_tensors="pt")
        if inputs.input_ids.size()[1] > 512:
            result.append(index)

    return result


def remove_footer(text):
    footer_regex = r"Verlaufsbericht,\s*Seite\s*\d{1,3}\s*von\s*\d{1,3}(?:(?:\s*|-)[a-zßäöüáéí]+){1,4},(?:(?:\s*|-)[a-zßäöüáéí]+\.?){1,4}\s*geb.\s*Datum\s*\d{1,2}\.\d{1,2}\.\d{4}, (?:weiblich|männlich),\s*\d{1,2}\s*J\s*Dokument\s*nur\s*zur\s*internen\s*Verwendung\s*Patienten\s*Nr\.:\s*\d{4,8}\s*gedruckt\s*am:\s*\d{1,2}\.\d{1,2}\.\d{4}"
    footer_regex_obj = re.compile(footer_regex, re.IGNORECASE)
    return footer_regex_obj.sub("", text)


def replace_names(text, cfg):
    root = cfg.root
    temp_folder = "temp_hpc_transfer"
    names = pd.read_csv(construct_path(root, temp_folder, "last_names_combined.csv"), encoding="utf-8")
    drop_names = [
        "Morgen",
        "Herz",
        "Fieber",
        "Tag",
        "Beutel",
        "Grund",
        "Weiß",
        "Finger",
        "Linke",
        "Röder",
        "Schmuck",
    ]
    names = names[~names.nachname.isin(drop_names)]  # remove names that have meaning in the notes

    names_regex = r"(?<=\b)(" + "|".join(re.escape(name) for name in names["nachname"]) + r")(?=\b)"
    names_regex_obj = re.compile(names_regex, re.IGNORECASE)

    fake = Faker(["de_DE"])

    return names_regex_obj.sub(fake.last_name(), text)
    # return names_regex_obj.sub("Anonymisiert", text)


def replace_abbreviations(text, abbreviations=None):
    if not text or pd.isna(text):
        return text
    # abbreviations_regex = r"\b(" + "|".join(re.escape(abbrev) for abbrev in abbreviations.keys()) + r")\b"
    # abbreviations_regex_obj = re.compile(abbreviations_regex, re.IGNORECASE)
    # Create regex pattern that matches whole words only
    abbreviations_regex = r'\b(?:' + '|'.join(re.escape(abbrev) for abbrev in abbreviations.keys()) + r')\b'
    abbreviations_regex_obj = re.compile(abbreviations_regex, re.IGNORECASE)

    def replace_match(match):
        key = match.group().lower()
        return abbreviations.get(key, match.group())  # Return original if not found

    text = abbreviations_regex_obj.sub(replace_match, text)
    return text


def combine_all_notes_by_time(df, interval_hours=6):
    rows_by_id = {}
    for cass_id, case_rows in df.groupby("id"):
        rows_by_id[cass_id] = case_rows.reset_index(drop=True)

    notes_combined = []
    for id, case_rows in rows_by_id.items():
        notes_combined.append(combine_case_notes_by_time(case_rows, interval_hours))

    notes_combined_df = pd.concat(notes_combined).reset_index(drop=True)
    return notes_combined_df


def combine_case_notes_by_time(case_rows, interval_hours=1):
    case_rows["OP_Naht_UTC"] = pd.to_datetime(case_rows["OP_Naht_UTC"])
    case_rows["DateTime_note"] = pd.to_datetime(case_rows["DateTime_note"])

    op_time = case_rows.iloc[0]["OP_Naht_UTC"]
    last_note_time = case_rows.sort_values("datetime", ascending=False).iloc[0]["datetime"]
    first_note_time = case_rows.sort_values("datetime").iloc[0]["datetime"]

    cass_id = case_rows.iloc[0]["id"]
    if pd.isnull(op_time):
        print(f"Removed entries for {cass_id} because timestamp of OP_Naht is {op_time}")
        return
    elif last_note_time < op_time:
        print(f"Removed entries for {cass_id} because last clinical note was recorded before OP_Naht")
        return

    case_rows = case_rows.sort_values(
        "datetime"
    )  # to ensure that clinical notes are concated in the order they were written in

    # determine time for start of first interval
    if first_note_time < op_time:
        time_diff = op_time - first_note_time
        current_start_time = op_time - math.ceil((time_diff) / timedelta(hours=interval_hours)) * timedelta(
            hours=interval_hours
        )
        # print(f"{cass_id}: op at {op_time}, starting intervals at {current_start_time}")   # uncomment for understanding/debugging
    else:
        current_start_time = op_time

    # identify notes in same interval and combine them
    case_rows_combined = []
    while current_start_time <= last_note_time:
        current_end_time = current_start_time + timedelta(
            hours=interval_hours, seconds=-1
        )  # substract one second to ensure that intervals do not overlap
        interval_rows = case_rows[
            (current_start_time <= case_rows["datetime"])
            & (case_rows["datetime"] <= current_end_time)
            ]
        if not interval_rows.empty:
            case_rows_combined.append(aggregate_rows(interval_rows))
        current_start_time = current_end_time + timedelta(seconds=1)

    case_rows_combined_df = pd.concat(case_rows_combined)
    return case_rows_combined_df


def aggregate_rows(rows):
    def keep_first(series):  # assumes that rows is ordered by time
        return series.iloc[0]

    def concat_strings(series):
        return "; ".join(series)

    # group by note_provider to combine notes from nurses/doctors separately
    rows_aggregated = rows.groupby("note_provider", as_index=False).agg(
        {col: (concat_strings if col == "text" else keep_first) for col in df.columns}
    )
    return rows_aggregated


# def get_embedding_multilingual(text):
#     def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#         last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#         return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#
#     # Each input text should start with "query: " or "passage: ", even for non-English texts.
#     # For tasks other than retrieval, you can simply use the "query: " prefix.
#     input_texts = [
#         "query: how much protein should a female eat",
#         "query: 南瓜的家常做法",
#         "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#         "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅",
#     ]
#
#     # multilingual_e5_path = "/home/winterah/.cache/huggingface/hub/intfloat/multilingual-e5-large"
#     tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
#     model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
#
#     # Tokenize the input texts
#     batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
#
#     outputs = model(**batch_dict)
#     embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
#
#     return embeddings


def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state
    mean_embedding = torch.mean(embedding, dim=1)
    # cls_embedding = embedding[:, 0, :]  # not used
    return mean_embedding.detach().numpy()  # result consists of 768 float values


def reduce_dimensionality(embeddings, target_dim=100):
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)  # standardize embeddings

    if embeddings.shape[0] < target_dim:
        print(
            """There are less data entries than desired dimensions (100), the embedding can therefore not be reduced.\nWARNING: The embeddings that are returned have not been reduced and therefore probably have a different structure than previous embeddings."""
        )
        return embeddings

    pca = PCA(
        n_components=target_dim
    )  # n_components specifies the number of dimensions a vector should be reduced to
    reduced_embeddings = pca.fit_transform(normalized_embeddings)

    return reduced_embeddings
