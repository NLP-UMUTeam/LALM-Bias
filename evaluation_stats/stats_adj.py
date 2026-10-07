#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from _stats_shared import (
    MODEL_DISPLAY_ORDER,
    collect_response_pairs,
    display_model_name,
    model_filename_label,
    summarise_intersection_stability,
    summarise_response_pairs,
    summarise_stability_bias,
    validate_paired_speaker_cluster_bootstrap,
    write_full_results_latex,
    write_stability_bias_latex,
    write_stability_summary_latex,
)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("normal_results", type=Path)
    parser.add_argument("reverse_results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        normal_records = collect_response_pairs(args.normal_results)
        reverse_records = collect_response_pairs(args.reverse_results)
        if not normal_records:
            raise ValueError("No recognised individual-adjective JSONL files were found.")
        output_dir = args.output_dir / "adj"
        output_dir.mkdir(parents=True, exist_ok=True)
        validate_paired_speaker_cluster_bootstrap()
        intersection_rows = summarise_response_pairs(
            normal_records, reverse_records, include_ci=True
        )
        stability_rows = summarise_intersection_stability(
            normal_records, reverse_records
        )
        output_paths = []
        summary_path = output_dir / "0_stability_bias_summary.tex"
        write_stability_bias_latex(
            summarise_stability_bias(stability_rows, intersection_rows), summary_path
        )
        output_paths.append(summary_path)
        dimension_path = output_dir / "1_prompt_order_stability_by_dimension.tex"
        variety_path = output_dir / "1_prompt_order_stability_by_variety.tex"
        write_stability_summary_latex(stability_rows, dimension_path, variety_path)
        output_paths.extend((dimension_path, variety_path))
        model_names = sorted(
            {model for model, _, _, _ in normal_records},
            key=lambda model: (
                MODEL_DISPLAY_ORDER.get(display_model_name(model), len(MODEL_DISPLAY_ORDER)),
                model,
            ),
        )
        for model in model_names:
            model_rows = [row for row in intersection_rows if row["model"] == model]
            if model_rows:
                output_path = output_dir / f"2_full_results_{model_filename_label(model)}.tex"
                write_full_results_latex(model_rows, output_path)
                output_paths.append(output_path)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for output_path in output_paths:
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
