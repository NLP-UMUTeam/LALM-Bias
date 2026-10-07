#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _stats_shared import (
    ADJECTIVE_DIMENSIONS,
    DIALECT_ABBREVIATIONS,
    DIALECT_DISPLAY_ORDER,
    DIMENSION_DISPLAY_ORDER,
    collect_response_pairs,
    display_model_name,
    model_filename_label,
    summarise_response_pairs,
)


_PAIR_BY_DIMENSION = {
    dimension: pair_name for pair_name, dimension in ADJECTIVE_DIMENSIONS.items()
}
RADAR_PAIR_ORDER = tuple(
    _PAIR_BY_DIMENSION[dimension] for dimension in DIMENSION_DISPLAY_ORDER
)


def set_radar_category_labels(axis, angles, labels, radial_range, fontsize):
    axis.set_xticks(angles)
    axis.set_xticklabels([])
    for angle, label in zip(angles, labels):
        horizontal = math.sin(angle)
        vertical = math.cos(angle)
        margin = radial_range * (0.025 + 0.075 * abs(horizontal))
        axis.text(
            angle,
            radial_range + margin,
            label,
            fontsize=fontsize,
            ha="center" if abs(horizontal) < 0.2 else ("left" if horizontal > 0 else "right"),
            va="center" if abs(vertical) < 0.2 else ("bottom" if vertical > 0 else "top"),
            clip_on=False,
        )


def write_radar_by_adjective_pair(rows, output_path, pairs):
    delta_by_key = {
        (row["dialect"], row["pair_name"]): row["delta_percentage_points"]
        for row in rows
    }
    values = [
        value
        for (_, pair_name), value in delta_by_key.items()
        if pair_name in pairs and value is not None
    ]
    if not values:
        raise ValueError("The model has no selection gaps for the radar figure.")

    lower_limit = min(-20, math.floor(min(values) / 5) * 5)
    upper_limit = max(5, math.ceil(max(values) / 10) * 10)
    radial_range = upper_limit - lower_limit
    angles = [2 * math.pi * index / len(pairs) for index in range(len(pairs))]
    closed_angles = angles + [angles[0]]
    colors = plt.get_cmap("tab10").colors
    figure, axis = plt.subplots(figsize=(9.2, 7.4), subplot_kw={"projection": "polar"})
    axis.set_theta_offset(math.pi / 2)
    axis.set_theta_direction(-1)
    set_radar_category_labels(
        axis,
        angles,
        [ADJECTIVE_DIMENSIONS[pair_name] for pair_name in pairs],
        radial_range,
        13,
    )
    axis.set_ylim(0, radial_range)
    tick_values = [
        value
        for value in [-10, 0, *range(20, int(upper_limit) + 1, 20)]
        if lower_limit <= value <= upper_limit
    ]
    if upper_limit not in tick_values:
        tick_values.append(upper_limit)
    axis.set_yticks([value - lower_limit for value in tick_values])
    axis.set_yticklabels(
        ["0 pp" if value == 0 else f"{value:.0f}" for value in tick_values], fontsize=13
    )
    for tick_label, value in zip(axis.get_yticklabels(), tick_values):
        if value == 0:
            tick_label.set_fontweight("bold")
        elif value < 0:
            tick_label.set_color("#b22222")
    axis.set_rlabel_position(18)
    axis.xaxis.grid(color="#9a9a9a", alpha=0.6, linewidth=0.7)
    axis.yaxis.grid(False)
    axis.spines["polar"].set_color("#4a4a4a")
    axis.spines["polar"].set_linewidth(0.9)
    for reference_value in tick_values:
        axis.axhline(
            reference_value - lower_limit,
            color="#b22222" if reference_value < 0 else "#303030",
            linestyle="--",
            linewidth=1.1 if reference_value == 0 else 0.75,
            alpha=1 if reference_value == 0 else 0.65,
            zorder=1,
        )

    for index, dialect in enumerate(DIALECT_DISPLAY_ORDER):
        selection_gaps = [delta_by_key.get((dialect, pair_name)) for pair_name in pairs]
        radial_values = [
            None if value is None else value - lower_limit for value in selection_gaps
        ]
        color = colors[index % len(colors)]
        if any(value is not None for value in radial_values):
            axis.plot(
                closed_angles,
                radial_values + [radial_values[0]],
                color=color,
                linewidth=1.6,
                marker="o",
                markersize=3.8,
                alpha=0.72,
                label=DIALECT_ABBREVIATIONS[dialect],
                zorder=2,
            )
        else:
            axis.plot(
                [],
                [],
                color=color,
                marker="o",
                markersize=3.8,
                alpha=0.72,
                label=DIALECT_ABBREVIATIONS[dialect],
            )

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.2, 1.05),
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#777777",
        framealpha=0.95,
        fontsize=13,
        handlelength=2.2,
        borderpad=0.6,
    )
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "normal_results",
        type=Path,
        help="Directory with normal-order adjective-selection JSONL results.",
    )
    parser.add_argument(
        "reverse_results",
        type=Path,
        help="Directory with reverse-order adjective-selection JSONL results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory; one radar PDF is written for each model.",
    )
    args = parser.parse_args()

    try:
        normal_records = collect_response_pairs(args.normal_results)
        reverse_records = collect_response_pairs(args.reverse_results)
        if not normal_records:
            raise ValueError("No recognised individual-adjective JSONL files were found.")
        intersection_rows = summarise_response_pairs(normal_records, reverse_records)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        model_names = sorted(
            {row["model"] for row in intersection_rows}, key=display_model_name
        )
        for model_name in model_names:
            model_rows = [row for row in intersection_rows if row["model"] == model_name]
            output_path = args.output_dir / f"adjective_{model_filename_label(model_name)}_radar_by_pair.pdf"
            write_radar_by_adjective_pair(model_rows, output_path, RADAR_PAIR_ORDER)
            output_paths.append(output_path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for output_path in output_paths:
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
