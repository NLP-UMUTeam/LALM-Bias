#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from _stats_shared import collect_cer_rows, write_cer_latex


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        rows = collect_cer_rows(args.results)
        output_dir = args.output_dir / "cer"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "0_cer.tex"
        write_cer_latex(rows, output_path)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
