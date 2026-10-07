#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from _stats_shared import (
    collect_bod_response_pairs,
    summarise_bod_stability,
    write_bod_stability_latex,
    write_bod_summary_latex,
)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("normal_results", type=Path)
    parser.add_argument("reverse_results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        normal_records = collect_bod_response_pairs(args.normal_results)
        reverse_records = collect_bod_response_pairs(args.reverse_results)
        if not normal_records:
            raise ValueError("No recognised base-or-dialect JSONL files were found.")
        output_dir = args.output_dir / "bod"
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = summarise_bod_stability(normal_records, reverse_records)
        summary_path = output_dir / "0_bod_recognition_summary.tex"
        stability_path = output_dir / "1_bod_stability.tex"
        write_bod_summary_latex(rows, summary_path)
        write_bod_stability_latex(rows, stability_path)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {summary_path}")
    print(f"Wrote {stability_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
