#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from _stats_shared import DIALECT_DISPLAY_ORDER

TIME_FIELDS = (
    "dialect_inference_time_sec",
    "mandarin_inference_time_sec",
    "dialect_total_time_sec",
    "mandarin_total_time_sec",
)


def input_files(results_dir):
    if not results_dir.is_dir():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    files = sorted(results_dir.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No JSONL files found in {results_dir}")
    return files


def group_from_path(jsonl_path, results_dir):
    relative_parts = jsonl_path.relative_to(results_dir).parts
    if len(relative_parts) < 3:
        raise ValueError(
            f"Expected model/.../dialect/file.jsonl beneath {results_dir}: {jsonl_path}"
        )
    model = relative_parts[0]
    dialect = relative_parts[-2]
    if dialect not in DIALECT_DISPLAY_ORDER:
        raise ValueError(f"Unknown dialect directory {dialect!r}: {jsonl_path}")
    return model, dialect


def read_times(jsonl_path):
    with jsonl_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path}:{line_number}: {error.msg}"
                ) from error
            missing = [field for field in TIME_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"Missing time fields in {jsonl_path}:{line_number}: "
                    + ", ".join(missing)
                )
            times = {}
            for field in TIME_FIELDS:
                value = record[field]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Invalid {field} in {jsonl_path}:{line_number}: {value!r}"
                    )
                times[field] = float(value)
            yield times


def collect_timings(results_dir):
    totals = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for jsonl_path in input_files(results_dir):
        model, dialect = group_from_path(jsonl_path, results_dir)
        for times in read_times(jsonl_path):
            totals[model][dialect]["pairs"] += 1
            for field, value in times.items():
                totals[model][dialect][field] += value
    return totals


def model_time_summary(totals):
    models = {}
    for model in sorted(totals):
        dialects = {}
        for dialect in DIALECT_DISPLAY_ORDER:
            values = totals[model].get(dialect)
            if not values:
                continue
            pairs = int(values["pairs"])
            regional_inference = values["dialect_inference_time_sec"]
            standard_inference = values["mandarin_inference_time_sec"]
            regional_total = values["dialect_total_time_sec"]
            standard_total = values["mandarin_total_time_sec"]
            dialects[dialect] = {
                "pairs": pairs,
                "regional": {
                    "inference_seconds_total": regional_inference,
                    "inference_seconds_mean_per_pair": regional_inference / pairs,
                    "total_seconds_total": regional_total,
                    "total_seconds_mean_per_pair": regional_total / pairs,
                },
                "standard": {
                    "inference_seconds_total": standard_inference,
                    "inference_seconds_mean_per_pair": standard_inference / pairs,
                    "total_seconds_total": standard_total,
                    "total_seconds_mean_per_pair": standard_total / pairs,
                },
                "combined": {
                    "inference_seconds_total": regional_inference + standard_inference,
                    "inference_seconds_mean_per_pair": (
                        regional_inference + standard_inference
                    ) / pairs,
                    "total_seconds_total": regional_total + standard_total,
                    "total_seconds_mean_per_pair": (regional_total + standard_total) / pairs,
                },
            }
        models[model] = dialects
    return models


def time_summary(totals):
    return {
        "unit": "seconds",
        "description": (
            "Cumulative recorded inference and end-to-end times by model and "
            "regional variety. These are sums of per-audio durations, not batch "
            "wall-clock times when processing was parallel."
        ),
        "models": model_time_summary(totals),
    }


def merge_totals(*totals_by_order):
    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for totals in totals_by_order:
        for model, dialects in totals.items():
            for dialect, values in dialects.items():
                for field, value in values.items():
                    merged[model][dialect][field] += value
    return merged


def total_seconds_by_model(totals):
    return {
        model: sum(
            values["dialect_total_time_sec"] + values["mandarin_total_time_sec"]
            for values in dialects.values()
        )
        for model, dialects in sorted(totals.items())
    }


def total_seconds_across_tasks(summary_totals):
    totals = defaultdict(float)
    for model_totals in summary_totals.values():
        for model, seconds in model_totals.items():
            totals[model] += seconds
    return dict(sorted(totals.items()))


def paired_prompt_time_summary(normal_totals, reverse_totals, task_description):
    description = (
        f"Cumulative recorded inference and end-to-end times for {task_description}. "
        "Values are sums of per-audio durations, not "
        "batch wall-clock times when processing was parallel."
    )
    return {
        "unit": "seconds",
        "description": description,
        "prompt_orders": {
            "normal": {"models": model_time_summary(normal_totals)},
            "reverse": {"models": model_time_summary(reverse_totals)},
            "combined": {
                "models": model_time_summary(
                    merge_totals(normal_totals, reverse_totals)
                )
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--transcription-results",
        type=Path,
        help="Root directory containing CER transcription results.",
    )
    parser.add_argument(
        "--bod-results",
        type=Path,
        nargs=2,
        metavar=("NORMAL_RESULTS", "REVERSE_RESULTS"),
        help=(
            "Normal and reverse base-versus-dialect result directories. "
            "Writes a separate JSON summary."
        ),
    )
    parser.add_argument(
        "--adj-results",
        type=Path,
        nargs=2,
        metavar=("NORMAL_RESULTS", "REVERSE_RESULTS"),
        help=(
            "Normal and reverse individual-adjective result directories. "
            "Writes a separate JSON summary."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the generated JSON summary.",
    )
    args = parser.parse_args()

    try:
        if not any(
            (args.transcription_results, args.bod_results, args.adj_results)
        ):
            raise ValueError(
                "Provide at least one of --transcription-results, --bod-results, "
                "or --adj-results."
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        summary_totals = {}
        if args.transcription_results:
            transcription_totals = collect_timings(args.transcription_results)
            transcription_path = args.output_dir / "transcription_time_summary.json"
            transcription_path.write_text(
                json.dumps(
                    time_summary(transcription_totals),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            output_paths.append(transcription_path)
            summary_totals["transcription"] = total_seconds_by_model(
                transcription_totals
            )
        if args.bod_results:
            normal_dir, reverse_dir = args.bod_results
            normal_totals = collect_timings(normal_dir)
            reverse_totals = collect_timings(reverse_dir)
            bod_path = args.output_dir / "base_or_dialect_time_summary.json"
            bod_path.write_text(
                json.dumps(
                    paired_prompt_time_summary(
                        normal_totals,
                        reverse_totals,
                        "base-versus-dialect classification",
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            output_paths.append(bod_path)
            summary_totals["base_or_dialect"] = total_seconds_by_model(
                merge_totals(normal_totals, reverse_totals)
            )
        if args.adj_results:
            normal_dir, reverse_dir = args.adj_results
            normal_totals = collect_timings(normal_dir)
            reverse_totals = collect_timings(reverse_dir)
            adjective_path = args.output_dir / "adjective_time_summary.json"
            adjective_path.write_text(
                json.dumps(
                    paired_prompt_time_summary(
                        normal_totals,
                        reverse_totals,
                        "individual-adjective selection",
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            output_paths.append(adjective_path)
            summary_totals["adjective_selection"] = total_seconds_by_model(
                merge_totals(normal_totals, reverse_totals)
            )
        total_summary_path = args.output_dir / "total_time_by_model.json"
        total_summary_path.write_text(
            json.dumps(
                {
                    "unit": "seconds",
                    "total_seconds_by_model": summary_totals,
                    "total_seconds_across_included_tasks_by_model": (
                        total_seconds_across_tasks(summary_totals)
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        output_paths.append(total_summary_path)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    for output_path in output_paths:
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
