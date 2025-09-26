import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import polars as pl
import os
import numpy as np
home_folder = "vandewrp"
project_path = f"/home/{home_folder}/projects/hpc-cassandra-analysis/"
sys.path.append(project_path)
from preprocessing.text_embeddings.process_text_embeddings import get_embeddings, generate_combined_column

local_dir = Path("/sc-projects/sc-proj-cc08-cassandra/RW_Clinical_Notes/")
meta_path = local_dir / "umls_home/2025AA/META"

(local_dir / ".cache/huggingface").mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = f"{local_dir}/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = f"{local_dir}/.cache/huggingface"

def parse_arguments():
    """Parse command line arguments for clinical notes processing."""
    parser = argparse.ArgumentParser(
        description="Process clinical notes and generate embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data-path", type=str, default=None, help="Path which contains the parquet "
                                                                    "file with text")

    # Configuration and paths
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Override project path (defaults to value in script)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./output",
        help="Output path for results"
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
        "--text-columns",
        type=str,
        nargs="+",
        default=["text"],
        help="Column name(s) containing text to embed (space-separated list)"
    )

    # Output options
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        default=True,
        help="Save embeddings as numpy array"
    )

    return parser.parse_args()

def __main__():
    args = parse_arguments()
    data = pl.read_parquet(args.data_path)
    columns = args.text_columns
    # logging.info(f"Collecting {len(data)} rows")
    logging.info(f"Data head: {data.head(5)}")
    data = data.head(5) if args.debug else data
    data, valid_columns = generate_combined_column(data, columns)
    logging.info(f"Generated combined column with {len(data)} rows and columns: {valid_columns}")
    # Set project path if provided
    if args.project_path:
        import sys
        sys.path.append(args.project_path)

    # Validate dimensions
    dims = args.dims
    if dims not in [32, 64, 128, 256, 512]:
        raise ValueError(f"Invalid dimensions {dims}. Supported values are: 32, 64, 128, 256, 512")

    # Get embeddings
    processed_df, all_embeddings_array = get_embeddings(
        data,
        model_string=args.model,
        column="combined",
        dims=dims
    )
    datetime.now().strftime("%Y-%m-%d_%H-%M")
    # output_file = f"{args.output_prefix}_{args.data_type}_{timestamp}_dims_{dims}_debug_{args.debug}"
    output_file = os.path.splitext(args.output_path)[0]
    np.save(
        f"{output_file}.npy",
        all_embeddings_array)
    processed_df.write_parquet(
        f"{output_file}.parquet",
    )
    logging.info(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if parse_arguments().debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    __main__()