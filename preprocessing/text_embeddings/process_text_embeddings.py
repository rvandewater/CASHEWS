import logging
import os
import re
import sys
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel
import polars as pl
home_folder = "vandewrp"
project_path = f"/home/{home_folder}/projects/hpc-cassandra-analysis/"
from collections import Counter
sys.path.append(project_path)
from preprocessing.text_embeddings.clinical_notes_utils import construct_path, get_timestamp, \
    load_clinical_notes, remove_footer
from preprocessing.prepare_segmentation import prepare_interval_dates
from execute_data_segmentation import calculate_start_end_segment_for_cohort
from preprocessing.text_embeddings.medications import load_ward_medications, load_copra_medications, load_fluids



local_dir = Path("/sc-projects/sc-proj-cc08-cassandra/RW_Clinical_Notes/")
meta_path = local_dir / "umls_home/2025AA/META"

(local_dir / ".cache/huggingface").mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = f"{local_dir}/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = f"{local_dir}/.cache/huggingface"



def extract_nlp_features(df, text_column='Text', prefix="note", silent=False):
    """Extract comprehensive NLP features from clinical notes using pandas"""

    # First, let's check what's in the text column
    if not silent:
        print(f"Checking text column: {text_column}")
        print("Sample of text column:")
        print(df[[text_column, "id"]].head())
        print(f"Non-null count: {df[text_column].notna().sum()}")
        print(f"Non-empty count: {((df[text_column].notna()) & (df[text_column] != '')).sum()}")

    all_words = []  # Collect all words for frequency analysis

    def calculate_features(text):
        if text is None or text == "" or pd.isna(text):
            return pd.Series({
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0,
                'punctuation_count': 0,
                'digit_count': 0,
                'uppercase_count': 0,
                'unique_word_count': 0,
                'lexical_diversity': 0.0
            })

        # Convert to string if not already
        text = str(text)

        # Basic counts
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        # Clean words for frequency analysis
        clean_words = [word.lower().strip('.,!?;:()[]{}\"\'') for word in words if word.strip('.,!?;:()[]{}\"\'')]
        all_words.extend(clean_words)

        # Sentence count (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Average word length
        avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / word_count if word_count > 0 else 0

        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Punctuation count
        punctuation_count = len(re.findall(r'[.,!?;:\-()"\']', text))

        # Digit count
        digit_count = len(re.findall(r'\d', text))

        # Uppercase count
        uppercase_count = len(re.findall(r'[A-Z]', text))

        # Unique words and lexical diversity
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        unique_word_count = len(unique_words)
        lexical_diversity = unique_word_count / word_count if word_count > 0 else 0

        return pd.Series({
            f'{prefix}_feature_word_count': word_count,
            f'{prefix}_feature_char_count': char_count,
            f'{prefix}_feature_sentence_count': sentence_count,
            f'{prefix}_feature_avg_word_length': round(avg_word_length, 2),
            f'{prefix}_feature_avg_sentence_length': round(avg_sentence_length, 2),
            f'{prefix}_feature_punctuation_count': punctuation_count,
            f'{prefix}_feature_digit_count': digit_count,
            f'{prefix}_feature_uppercase_count': uppercase_count,
            f'{prefix}_feature_unique_word_count': unique_word_count,
            f'{prefix}_feature_lexical_diversity': round(lexical_diversity, 3)
        })

    # Apply feature extraction with progress bar suppressed
    if not silent:
        from tqdm import tqdm
        tqdm.pandas(desc="Extracting NLP features")
        nlp_features = df[text_column].progress_apply(calculate_features)
    else:
        nlp_features = df[text_column].apply(calculate_features)

    # Combine with original columns
    result_df = pd.concat([
        df,
        nlp_features,
    ], axis=1)
    # Calculate 100 most common words if we have any
    common_words_df = None
    if all_words:
        word_counter = Counter(all_words)
        most_common_words = word_counter.most_common(100)

        # if not silent:
            # print("100 Most Common Words in Clinical Notes:")
            # print("=" * 50)
            # for i, (word, count) in enumerate(most_common_words, 1):
            #     print(f"{i:3d}. {word:<15} ({count:,} occurrences)")

        # Create DataFrame with word frequencies
        common_words_df = pd.DataFrame({
            'rank': range(1, min(101, len(most_common_words) + 1)),
            'word': [word for word, count in most_common_words],
            'frequency': [count for word, count in most_common_words],
            'percentage': [count / sum(word_counter.values()) * 100 for word, count in most_common_words]
        })
    elif not silent:
        print("No words found in text data!")

    return result_df, common_words_df


# Apply NLP feature extraction with tqdm suppressed
# df_nlp_features, df_common_words = extract_nlp_features(df_notes_fixed, text_column="text", silent=True)

def replace_abbreviations(text, abbreviations):
    """
    Replace medical abbreviations with their full meanings in clinical text.

    Args:
        text: The clinical note text to process
        abbreviations: Dictionary mapping abbreviations to their meanings

    Returns:
        Text with abbreviations replaced by their full meanings
    """
    if not text or pd.isna(text):
        return text

    # Convert to string if not already
    text = str(text)

    # Create regex pattern that matches whole words only using word boundaries
    # Sort by length (longest first) to handle overlapping abbreviations
    sorted_abbrevs = sorted(abbreviations.keys(), key=len, reverse=True)
    abbreviations_regex = r'\b(?:' + '|'.join(re.escape(abbrev) for abbrev in sorted_abbrevs) + r')\b'
    abbreviations_regex_obj = re.compile(abbreviations_regex, re.IGNORECASE)

    def replace_match(match):
        matched_text = match.group()
        # Try exact case match first, then lowercase
        return abbreviations.get(matched_text, abbreviations.get(matched_text.lower(), matched_text))

    text = abbreviations_regex_obj.sub(replace_match, text)
    return text

def generate_combined_column(df, columns):
    """
    Combines column for embedding generation.
    :param df: Dataframe containing the columns to combine
    :param columns: Columns to combine
    :return: Combined dataframe with a new 'combined' column
    """
    result_df = df.clone()
    valid_columns = [col for col in columns if col in result_df.columns]
    if not valid_columns:
        raise ValueError("No valid columns found in the input dataframe. Please check the column names.")
    # Print non-valid columns
    non_valid_columns = [col for col in columns if col not in result_df.columns]
    if non_valid_columns:
        print(f"Warning: The following columns are not present in the input dataframe "
              f"and will be ignored: {non_valid_columns}")
    # Filter out rows with null values and report
    initial_count = len(result_df)
    result_df = result_df.filter(~pl.any_horizontal([pl.col(col).is_null() for col in valid_columns]))
    removed_count = initial_count - len(result_df)
    if removed_count > 0:
        print(
            f"Removed {removed_count} rows ({removed_count / initial_count * 100:.1f}%) "
            f"with null values in {valid_columns}")
        if removed_count == initial_count:
            raise ValueError("All rows were removed due to null values in the specified columns. "
                             "Please check the input dataframe and column names.")
    # Process each column separately if not only_combined
    # Create a combined column from all supplied columns
    print(f"Combining columns {columns}")
    # Create a combined column from all supplied columns using Polars expressions
    if len(valid_columns) == 0:
        raise ValueError("No valid columns found for combining. Please check the input dataframe and column names.")
    elif len(valid_columns) == 1:
        print(f"Only one valid column found: {valid_columns[0]}. No need to combine. Will rename it to 'combined'.")
        result_df = result_df.with_columns(pl.col(valid_columns[0]).alias("combined"))
        valid_columns = ["combined"]
    else:
        combined_expr = pl.concat_str([pl.col(col).cast(pl.String) for col in valid_columns], separator=" ")
        result_df = result_df.with_columns(combined_expr.alias("combined"))
    return result_df, valid_columns

def get_embeddings(df, model_string="jinaai/jina-embeddings-v3", column="text",dims=128):
    logging.info(f"getting embeddings for column: {column} using model: {model_string} with dims: {dims}")
    logging.info(df.shape)
    # Initialize the model
    model = AutoModel.from_pretrained(model_string, trust_remote_code=True)
    # model.to("cuda")
    all_embeddings = []
    for text in tqdm(df[column], desc="Getting embeddings", ):
        try:
            # with logging_redirect_tqdm():
            embedding = model.encode(text, truncate_dim=dims, show_progress_bar=False)  # get_embedding(text)
            all_embeddings.append(embedding)
        except Exception as e:
            all_embeddings.append(np.zeros((1, dims)))  # if an error occurs, add a zero vector to the list
            print(f"Error for text: {text}, error: {e}")
    all_embeddings_array = np.vstack(all_embeddings)
    # reduced_embeddings = reduce_dimensionality(all_embeddings_array)
    if isinstance(df, pl.DataFrame):
        df = df.with_columns(pl.Series("embedding", all_embeddings_array))
    else:
        df["embedding"] = list(all_embeddings_array)
    return df, all_embeddings_array

def filter_items_by_time(df_notes, start_time, end_time):
    """
    Filter notes in the DataFrame based on start and end time.
    Args:
        df_notes: DataFrame containing notes with 'id' and 'datetime' columns.
        start_time: Dictionary mapping patient IDs to start times.
        end_time: Dictionary mapping patient IDs to end times.
    Returns:
        DataFrame containing notes within the specified time range for all patients.
    """
    # Filter to only include patients that have both start and end times
    valid_patients = [pid for pid in start_time.keys()
                     if start_time.get(pid) is not None and end_time.get(pid) is not None]

    # Filter dataframe to only include valid patients
    df_filtered = df_notes[df_notes['id'].isin(valid_patients)].copy()

    # Create boolean mask for notes within time periods
    mask = pd.Series(False, index=df_filtered.index)

    for patient_id in valid_patients:
        patient_mask = (
            (df_filtered['id'] == patient_id) &
            (df_filtered['datetime'] >= start_time[patient_id]) &
            (df_filtered['datetime'] <= end_time[patient_id])
        )
        mask = mask | patient_mask
    print(f"Filtered {len(df_filtered) - mask.sum()} notes outside of the specified time range. Remaining notes: {mask.sum()}")
    return df_filtered[mask]





def parse_arguments():
    """Parse command line arguments for clinical notes processing."""
    parser = argparse.ArgumentParser(
        description="Process clinical notes and generate embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data_type", type=str, default="clinical_notes",
                        choices=["clinical_notes", "medications"],
                        help="Type of data to process")

    # Configuration and paths
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Override project path (defaults to value in script)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="temp_hpc_transfer",
        help="Output directory for results"
    )

    # Cohort and time filtering
    parser.add_argument(
        "--cohort",
        type=str,
        default="ICU_and_normal_ward",
        choices=["ICU_and_normal_ward", "real_life_set"],
        help="Cohort to process"
    )
    parser.add_argument(
        "--culling",
        action="store_true",
        help="Enable culling during interval preparation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    # Embedding parameters
    parser.add_argument(
        "--model",
        type=str,
        default="jinaai/jina-embeddings-v3",
        help="Hugging Face model for embeddings"
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=128,
        help="Embedding dimensions, currently supported with jinai-3: 32, 64, 128, 256, 512"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text to embed"
    )

    # Processing options
    parser.add_argument(
        "--skip-abbreviations",
        action="store_true",
        help="Skip abbreviation replacement"
    )
    parser.add_argument(
        "--skip-footer-removal",
        action="store_true",
        help="Skip footer removal from notes"
    )
    parser.add_argument(
        "--skip-extract-nlp-features",
        action="store_true",
        help="Skip extract NLP features from text"
    )

    # Output options
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        default=True,
        help="Save embeddings as numpy array"
    )
    parser.add_argument(
        "--save-notes",
        action="store_true",
        default=True,
        help="Save processed notes as parquet"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output filenames"
    )

    return parser.parse_args()


def __main__():
    """Main function with argparse integration."""
    logging.basicConfig(
        level=logging.INFO,
    )

    args = parse_arguments()

    # Use project path from args if provided
    if args.project_path:
        global project_path
        project_path = args.project_path
        sys.path.insert(0, project_path)

    # Load configuration
    config_path = args.config if os.path.isabs(args.config) else f"{project_path}/{args.config}"
    cfg = OmegaConf.load(config_path)
    dims = args.dims if args.dims else 128
    print(f"Type of data to process: {args.data_type}")
    print(f"Config loaded: {config_path}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    print(f"Processing with model: {args.model}")
    print(f"Embedding dimensions: {dims}")
    print(f"Cohort: {args.cohort}")

    # Load clinical notes
    if args.data_type == "medications":
        icu_meds = load_copra_medications(cfg)
        ward_meds = load_ward_medications(cfg)
        icu_fluids = load_fluids(cfg)
        data_to_embed = pd.concat([icu_meds, ward_meds, icu_fluids], ignore_index=True)
        data_to_embed['datetime'] = pd.to_datetime(data_to_embed['datetime']).dt.tz_localize(None)
        prefix = "med"
    elif args.data_type == "clinical_notes":
        data_to_embed = load_clinical_notes(cfg)
        prefix = "note"
    else:
        raise ValueError(f"Unsupported data type: {args.data_type}. Choose 'clinical_notes' or 'medications'.")

    # Prepare interval dates
    admission_time, surgery_time, icu_transfer_time, ward_transfer_time, complication_time, discharge_time = (
        prepare_interval_dates(cfg, culling=args.culling, verbose=args.verbose, cohort="real_life_set")
    )

    start_time, end_time = calculate_start_end_segment_for_cohort(
        args.cohort, discharge_time, complication_time,
        icu_transfer_time, surgery_time, ward_transfer_time
    )
    data_to_embed = filter_items_by_time(data_to_embed, start_time, end_time)
    if args.debug:
        data_to_embed = data_to_embed.head(100)  # Limit to 100 notes for debugging
        logging.info("Debug mode enabled: processing only first 100 notes.")

    if args.data_type == "clinical_notes":
        # Note specific processing

        # Process abbreviations if not skipped
        if not args.skip_abbreviations:
            abbreviations_df = pd.read_csv(
                cfg.clinical_notes.medical_abbreviations,
                sep=";", decimal=",", encoding="utf-8"
            )
            abbreviations = abbreviations_df.set_index("Abbreviation")["Meaning"].to_dict()
            data_to_embed["text"] = data_to_embed["text"].apply(
                replace_abbreviations, abbreviations=abbreviations
            )

        # Remove footers if not skipped
        if not args.skip_footer_removal:
            data_to_embed["text"] = data_to_embed["text"].apply(remove_footer)
            logging.info(f"Notes after footer removal: {len(data_to_embed)}")
        dummies = pd.get_dummies(data_to_embed["note_provider"], prefix='note_feature_provider', dtype=int)
    else:
        dummies = pd.get_dummies(data_to_embed["icu_med_substance_group"], prefix='icu_med_substance_group', dtype=int)
    data_to_embed = pd.concat([data_to_embed, dummies], axis=1)

    # Extract NLP features if not skipped
    if not args.skip_extract_nlp_features:
        data_to_embed, common_words_df = extract_nlp_features(data_to_embed, args.text_column,prefix)
        if common_words_df is not None:
            common_words_df.to_csv(
                construct_path(cfg.root, args.output_dir, f"common_words_{get_timestamp()}.csv")
            )

    # Get embeddings
    processed_text, all_embeddings_array = get_embeddings(
        data_to_embed,
        model_string=args.model,
        column=args.text_column,
        dims=dims
    )

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if args.save_embeddings:
        output_file = f"{args.output_prefix}_{args.data_type}_{timestamp}_dims_{dims}_debug_{args.debug}"
        np.save(
            construct_path(cfg.root, args.output_dir, f"{output_file}.npy"),
            all_embeddings_array
        )

        processed_text.to_parquet(
            construct_path(cfg.root, args.output_dir, f"{output_file}.parquet")
        )

    logging.info("Processing complete: embeddings generated and saved.")


if __name__ == "__main__":
    __main__()
