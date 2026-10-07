import os
import json
import time
from pathlib import Path

import pandas as pd

from utils import (
    parse_args,
    set_seed,
    filter_valid_speakers,
    clear_gpu,
    get_model_results_dir,
)
from ZeroShotAudio import ZeroShotAudio


def resolve_manifest_audio_path(audio_path, dialect_path):

    audio_path = Path(audio_path)
    if audio_path.is_absolute():
        return str(audio_path)

    manifest_dir = Path(dialect_path).resolve().parent
    try:
        audio_index = audio_path.parts.index("Audio")
    except ValueError:
        # Non-KeSpeech manifests may keep a path relative to the JSONL itself.
        return str(manifest_dir / audio_path)

    audio_relative_path = Path(*audio_path.parts[audio_index:])
    dataset_dir = (
        manifest_dir.parent
        if manifest_dir.name == "parallel_pairs_by_dialect"
        else manifest_dir
    )
    return str(dataset_dir / audio_relative_path)


def resolve_dialect_paths(dataset_dir, dialects):

    pairs_dir = Path(dataset_dir) / "parallel_pairs_by_dialect"

    if dialects:
        paths = [pairs_dir / f"{dialect}.jsonl" for dialect in dialects]
        missing_paths = [path for path in paths if not path.is_file()]

        if missing_paths:
            missing_dialects = ", ".join(path.stem for path in missing_paths)
            available_dialects = ", ".join(
                path.stem for path in sorted(pairs_dir.glob("*.jsonl"))
            )
            raise FileNotFoundError(
                f"No manifest found for: {missing_dialects}. "
                f"Available dialects: {available_dialects}."
            )
    else:
        paths = sorted(pairs_dir.glob("*.jsonl"))

    if not paths:
        raise FileNotFoundError(
            f"No JSONL manifests were found in {pairs_dir}."
        )

    return [str(path) for path in paths]


def get_processed_pairs(output_path):

    processed = set()

    if not os.path.exists(output_path):
        return processed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)

                mandarin_audio = result.get("mandarin_audio")
                dialect_audio = result.get("dialect_audio")
                if mandarin_audio and dialect_audio:
                    processed.add((mandarin_audio, dialect_audio))

    return processed


def update_metadata(metadata_path, subdialect, args, num_samples, run_time, prompt_times):

    if os.path.exists(metadata_path):

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    else:

        metadata = {
            "subdialect": subdialect,
            "model_name": args.model_name,
            "quantization_config": args.quantization_config,
            "num_samples": num_samples,
            "total_experiment_time_sec": 0.0,
            "runs": 0,
            "prompts": {},
        }

    metadata["total_experiment_time_sec"] += run_time

    metadata["runs"] += 1
    
    metadata["num_samples"] = max(
        metadata.get("num_samples", 0),
        num_samples,
    )

    for prompt_name, prompt_data in prompt_times.items():

        if prompt_name not in metadata["prompts"]:

            metadata["prompts"][prompt_name] = {
                "path": prompt_data["path"],
                "total_time_sec": 0.0,
            }

        metadata["prompts"][prompt_name]["total_time_sec"] += (
            prompt_data["time_sec"]
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4,
        )


def evaluate_dialect(dialect_path, args, prompt_paths, zero_shot):
    experiment_start = time.time()

    df = pd.read_json(dialect_path, lines=True)
    required_columns = {
        "mandarin_audio",
        "dialect_audio",
        "text",
        "speaker_id",
        "normalized_text",
        "subdialect",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{dialect_path} is not a valid Chinese pair manifest; "
            f"missing columns: {', '.join(sorted(missing_columns))}."
        )
    df = filter_valid_speakers(df)
    if df.empty:
        print(f"No valid samples found in {dialect_path}.")
        return

    subdialect = str(df.iloc[0]["subdialect"])
    output_group = subdialect
    output_dir = get_model_results_dir(
        args.results_dir,
        args.model_name,
    ) / output_group
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    prompt_times = {}

    print(f"Starting evaluation for subdialect: {subdialect}")

    for prompt_path in prompt_paths:
        prompt_start = time.time()
        prompt_full_path = f"{args.prompts_dir}/{prompt_path}"

        with open(prompt_full_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        prompt_name = os.path.splitext(os.path.basename(prompt_path))[0]
        output_path = output_dir / f"{output_group}_{prompt_name}.jsonl"
        processed_pairs = get_processed_pairs(output_path)

        print(
            f"\n{subdialect} / {prompt_name}: "
            f"{len(processed_pairs)} already processed"
        )

        with open(output_path, "a", encoding="utf-8") as output_file:
            for _, row in df.iterrows():
                mandarin_audio = resolve_manifest_audio_path(
                    row["mandarin_audio"], dialect_path
                )
                dialect_audio = resolve_manifest_audio_path(
                    row["dialect_audio"], dialect_path
                )
                pair = (mandarin_audio, dialect_audio)

                if pair in processed_pairs:
                    continue

                mandarin_result = zero_shot.ask(
                    prompt=prompt,
                    audio_path=mandarin_audio,
                    max_new_tokens=args.max_new_tokens,
                )
                dialect_result = zero_shot.ask(
                    prompt=prompt,
                    audio_path=dialect_audio,
                    max_new_tokens=args.max_new_tokens,
                )

                result = {
                    "speaker_id": row["speaker_id"],
                    "text": row["text"],
                    "normalized_text": row["normalized_text"],
                    "dialect_audio": dialect_audio,
                    "mandarin_audio": mandarin_audio,
                    "dialect_response": dialect_result["response"],
                    "dialect_token_probabilities": dialect_result[
                        "token_probabilities"
                    ],
                    "mandarin_response": mandarin_result["response"],
                    "mandarin_token_probabilities": mandarin_result[
                        "token_probabilities"
                    ],
                    "dialect_inference_time_sec": dialect_result[
                        "inference_time_sec"
                    ],
                    "mandarin_inference_time_sec": mandarin_result[
                        "inference_time_sec"
                    ],
                    "dialect_total_time_sec": dialect_result[
                        "total_time_sec"
                    ],
                    "mandarin_total_time_sec": mandarin_result[
                        "total_time_sec"
                    ],
                }

                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
                processed_pairs.add(pair)
                clear_gpu()

        prompt_time = time.time() - prompt_start
        prompt_times[prompt_name] = {
            "path": prompt_path,
            "time_sec": prompt_time,
        }
        print(f"{prompt_name} finished in {prompt_time:.2f} seconds")

    run_time = time.time() - experiment_start
    update_metadata(
        metadata_path=metadata_path,
        subdialect=subdialect,
        args=args,
        num_samples=len(df),
        run_time=run_time,
        prompt_times=prompt_times,
    )
    print(f"\n{subdialect} finished in {run_time:.2f} seconds")


def main():
    args = parse_args()
    set_seed(args.seed)
    dialect_paths = resolve_dialect_paths(args.dataset_dir, args.dialects)

    prompt_paths = sorted(
        filename
        for filename in os.listdir(args.prompts_dir)
        if filename.endswith(".txt")
    )

    if not prompt_paths:
        print(f"No .txt prompt files found in {args.prompts_dir}")
        return

    zero_shot = ZeroShotAudio(
        model_name=args.model_name,
        quantization_config=args.quantization_config,
    )

    try:
        for dialect_path in dialect_paths:
            evaluate_dialect(
                dialect_path=dialect_path,
                args=args,
                prompt_paths=prompt_paths,
                zero_shot=zero_shot,
            )
    finally:
        del zero_shot
        clear_gpu()


if __name__ == "__main__":
    main()
