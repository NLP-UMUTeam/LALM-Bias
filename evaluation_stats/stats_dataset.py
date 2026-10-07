#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from _stats_shared import DIALECT_ABBREVIATIONS, DIALECT_DISPLAY_ORDER

CITY_DISPLAY_NAMES = {
    "XiAn": "Xi'an",
}


def latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def display_city(city):
    return CITY_DISPLAY_NAMES.get(city, city)


def resolve_summary_path(path):
    if path.is_file():
        return path
    if path.is_dir() and (path / "selection_summary.json").is_file():
        return path / "selection_summary.json"
    raise ValueError(
        "Expected selection_summary.json or a directory containing it: "
        f"{path}"
    )


def load_summary(path):
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error.msg}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return summary


def required_mapping(summary, name):
    value = summary.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Selection summary is missing the {name} mapping.")
    return value


def pairs_per_variety(summary):
    counts = required_mapping(summary, "pairs_per_dialect")
    values = []
    for dialect in DIALECT_DISPLAY_ORDER:
        value = counts.get(dialect)
        if not isinstance(value, int):
            raise ValueError(f"Missing pair count for {dialect}.")
        values.append(value)
    if len(set(values)) != 1:
        raise ValueError("Pair counts are not balanced across regional varieties.")
    return values[0]


def sampling_targets(summary):
    total = pairs_per_variety(summary)
    by_gender = required_mapping(summary, "pairs_per_gender")
    by_gender_age = required_mapping(summary, "pairs_per_gender_and_age")
    female = by_gender.get("Female")
    male = by_gender.get("Male")
    if not isinstance(female, int) or not isinstance(male, int):
        raise ValueError("Selection summary has incomplete gender counts.")
    age_counts = {
        age_group: sum(
            by_gender_age.get(f"{gender}_{age_group}", 0)
            for gender in ("Female", "Male")
        )
        for age_group in ("18-29", "30-39", "40+")
    }
    if female + male != total or sum(age_counts.values()) != total:
        raise ValueError("Gender or age counts do not sum to pairs per variety.")
    return (
        ("Female speakers", female),
        ("Male speakers", male),
        ("Age 18--29", age_counts["18-29"]),
        ("Age 30--39", age_counts["30-39"]),
        ("Age 40+", age_counts["40+"]),
        ("Total", total),
    )


def city_counts(summary):
    total = pairs_per_variety(summary)
    by_dialect = required_mapping(summary, "pairs_per_city")
    result = {}
    for dialect in DIALECT_DISPLAY_ORDER:
        counts = by_dialect.get(dialect)
        if not isinstance(counts, dict):
            raise ValueError(f"Missing city counts for {dialect}.")
        if any(not isinstance(value, int) for value in counts.values()):
            raise ValueError(f"Invalid city counts for {dialect}.")
        if sum(counts.values()) != total:
            raise ValueError(f"City counts do not sum to {total} for {dialect}.")
        result[dialect] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return result


def repeated_speaker_statistics(summary):
    statistics = required_mapping(summary, "repeated_speaker_statistics")
    required = (
        "mean_pairs_per_speaker", "median_pairs_per_speaker",
        "speakers_with_multiple_pairs_percentage", "maximum_pairs_per_speaker",
    )
    missing = [name for name in required if name not in statistics]
    if missing:
        raise ValueError(
            "Selection summary is missing repeated-speaker statistics: "
            + ", ".join(missing)
        )
    return statistics


def write_sampling_table(rows, output_path):
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{lr}",
        r"\toprule", r"\textbf{Sampling target} & \textbf{Pairs per variety} \\",
        r"\midrule",
    ]
    for index, (label, value) in enumerate(rows):
        if index == 2:
            lines.append(r"\midrule")
        if index == len(rows) - 1:
            lines.extend((r"\midrule", rf"\textbf{{{label}}} & \textbf{{{value}}} \\") )
        else:
            lines.append(rf"{label} & {value} \\")
    lines.extend((
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Target distribution reproduced for each of the eight regional",
        r"Mandarin varieties in the balanced evaluation subset.}",
        r"\label{tab:balanced-sampling}", r"\end{table}", "",
    ))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_city_table(counts, output_path):
    lines = [
        r"\begin{table}[!htb]", r"\centering", r"\small",
        r"\begin{tabular}{llr}", r"\toprule",
        r"\textbf{Var.} & \textbf{City} & \textbf{Pairs} \\", r"\midrule",
    ]
    for dialect_index, dialect in enumerate(DIALECT_DISPLAY_ORDER):
        cities = counts[dialect]
        for city_index, (city, pair_count) in enumerate(cities):
            variety_cell = (
                rf"\multirow{{{len(cities)}}}{{*}}{{{DIALECT_ABBREVIATIONS[dialect]}}}"
                if city_index == 0 else ""
            )
            lines.append(
                f"{variety_cell} & {latex_escape(display_city(city))} & {pair_count} "
                + r"\\"
            )
        if dialect_index < len(DIALECT_DISPLAY_ORDER) - 1:
            lines.append(r"\midrule")
    lines.extend((
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Number of matched regional--Standard Mandarin pairs by regional variety and city in the final evaluation subset. Variety abbreviations are defined in Table~\ref{tab:variety-abbreviations}.}",
        r"\label{tab:dataset-cities}", r"\end{table}", "",
    ))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_repeated_speakers_table(statistics, output_path):
    median_value = statistics["median_pairs_per_speaker"]
    median_display = str(int(median_value)) if median_value == int(median_value) else str(median_value)
    rows = (
        ("Mean pairs per speaker", f"{statistics['mean_pairs_per_speaker']:.2f}"),
        ("Median pairs per speaker", median_display),
        ("Speakers with $>1$ pair", f"{statistics['speakers_with_multiple_pairs_percentage']:.1f}\\%"),
        ("Maximum pairs from one speaker", str(statistics["maximum_pairs_per_speaker"])),
    )
    lines = [
        r"\begin{table}[!htb]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{lr}",
        r"\toprule", r"\textbf{Statistic} & \textbf{Value} \\", r"\midrule",
    ]
    lines.extend(f"{label} & {value} " + r"\\" for label, value in rows)
    lines.extend((
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Repeated-speaker statistics in the final evaluation subset.}",
        r"\label{tab:repeated-speakers}", r"\end{table}", "",
    ))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Directory generated by the selector or its selection_summary.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for generated LaTeX tables.",
    )
    args = parser.parse_args()
    try:
        summary = load_summary(resolve_summary_path(args.dataset))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        sampling_path = args.output_dir / "0_balanced_sampling.tex"
        city_path = args.output_dir / "1_dataset_cities.tex"
        repeated_path = args.output_dir / "2_repeated_speakers.tex"
        write_sampling_table(sampling_targets(summary), sampling_path)
        write_city_table(city_counts(summary), city_path)
        write_repeated_speakers_table(repeated_speaker_statistics(summary), repeated_path)
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    for output_path in (sampling_path, city_path, repeated_path):
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
