import torch
import random
import argparse
import gc
import re

from pathlib import Path

import numpy as np


def get_model_directory_name(model_name):

    directory_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(model_name),
    ).strip("._")

    if not directory_name:
        raise ValueError("model_name must contain at least one valid character")

    return directory_name


def get_model_results_dir(results_dir, model_name):
    return Path(results_dir) / get_model_directory_name(model_name)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save the results",
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        required=True,
        help="Directory containing the prompt files",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the balanced KeSpeech subset",
    )
    parser.add_argument(
        "--dialects",
        type=str,
        nargs="*",
        help="Dialect names to evaluate. If omitted, all available dialects are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the LALM model to use",
    )
    parser.add_argument(
        "--quantization_config",
        type=str,
        default="none",
        help="Quantization configuration for the model (4bits, 8bits, none)",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate (default: 512)",
    )
    return parser.parse_args()

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def clear_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

##### Filtering functions

def get_audio_speaker_id(audio_path):
    try:
        parts = Path(audio_path).parts
        audio_index = parts.index("Audio")
        return parts[audio_index + 1]

    except (IndexError, ValueError, TypeError):
        return None

def filter_valid_speakers(df):

    dialect_speaker = df["dialect_audio"].apply(get_audio_speaker_id)
    mandarin_speaker = df["mandarin_audio"].apply(get_audio_speaker_id)

    subdialect = df.iloc[0]["subdialect"]

    mask = (
        (df["speaker_id"].astype(str) == dialect_speaker) &
        (df["speaker_id"].astype(str) == mandarin_speaker) &
        (df["subdialect"] == subdialect)
    )

    return df[mask].copy()
