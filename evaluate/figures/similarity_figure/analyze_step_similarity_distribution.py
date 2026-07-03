#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOOL_HEADINGS = ("### Code", "### Search", "### AskUser")
CATEGORY_REASONING = "Reasoning-only block"
CATEGORY_TOOL = "Tool-related block"

DEFAULT_FIGSIZE = (3.7, 4.0)
DEFAULT_DPI = 220
TITLE_FONT_SIZE = 18.5
AXIS_LABEL_FONT_SIZE = 19
TICK_LABEL_FONT_SIZE = 15.5
LEGEND_FONT_SIZE = 14.5
ANNOTATION_FONT_SIZE = 14
NOTE_FONT_SIZE = 12


def extract_mean_cosine(diag: dict) -> float | None:
    summary = diag.get("token_diagnostics_summary") or {}
    first_trigger = summary.get("first_trigger_info") or {}
    value = first_trigger.get("mean_selected_layer_cosine")
    return float(value) if value is not None else None


def classify_raw_block(raw_text: str, diag: dict) -> str | None:
    stripped = raw_text.lstrip()
    summary = diag.get("token_diagnostics_summary") or {}
    first_trigger = summary.get("first_trigger_info") or {}
    trigger_prefix = first_trigger.get("context_tail_before_trigger")

    if stripped.startswith("### Final Response") or trigger_prefix == "### Final":
        return None
    if any(heading in raw_text for heading in TOOL_HEADINGS):
        return CATEGORY_TOOL
    return CATEGORY_REASONING


def compute_quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of an empty list.")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot compute statistics for an empty group.")
    group_mean = mean(values)
    variance = sum((value - group_mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": group_mean,
        "median": median(values),
        "std": math.sqrt(variance),
        "min": min(values),
        "q25": compute_quantile(values, 0.25),
        "q75": compute_quantile(values, 0.75),
        "max": max(values),
    }


def load_points(input_path: Path) -> tuple[list[dict], dict]:
    records = json.loads(input_path.read_text())
    points: list[dict] = []
    total_predict_tool_steps = 0
    total_raw_tool_blocks = 0

    for example_index, record in enumerate(records):
        total_predict_tool_steps += sum(step.get("name") == "Tool Step" for step in record.get("predict", []))
        raws = record.get("raw", [])
        diags = record.get("steering_diagnostics", [])

        for round_index, (raw_text, diag) in enumerate(zip(raws, diags), start=1):
            similarity = extract_mean_cosine(diag)
            if similarity is None:
                continue

            category = classify_raw_block(raw_text, diag)
            if category is None:
                continue

            contains_tool = any(heading in raw_text for heading in TOOL_HEADINGS)
            if contains_tool:
                total_raw_tool_blocks += 1

            points.append(
                {
                    "example_index": example_index,
                    "round_index": round_index,
                    "category": category,
                    "similarity": similarity,
                    "contains_tool_heading": contains_tool,
                }
            )

    metadata = {
        "num_examples": len(records),
        "predict_tool_steps": total_predict_tool_steps,
        "raw_tool_blocks": total_raw_tool_blocks,
    }
    return points, metadata


def load_demo_csv_points(input_path: Path) -> tuple[list[dict], dict]:
    points: list[dict] = []
    datasets: set[str] = set()
    bad_point_count = 0

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            similarity_raw = row.get("similarity")
            kind = (row.get("kind") or "").strip()
            if similarity_raw is None or kind not in {"reasoning", "tool"}:
                continue

            similarity = float(similarity_raw)
            category = CATEGORY_TOOL if kind == "tool" else CATEGORY_REASONING
            contains_tool = kind == "tool"
            dataset = (row.get("dataset") or "").strip()
            if dataset:
                datasets.add(dataset)

            is_bad_point = int((row.get("is_bad_point") or "0").strip() or "0")
            bad_point_count += is_bad_point

            points.append(
                {
                    "example_index": int((row.get("example_index") or "0").strip() or "0"),
                    "round_index": int((row.get("raw_index") or "0").strip() or "0"),
                    "category": category,
                    "similarity": similarity,
                    "contains_tool_heading": contains_tool,
                    "dataset": dataset,
                    "is_bad_point": bool(is_bad_point),
                }
            )

    metadata = {
        "source_format": "demo_csv",
        "num_examples": None,
        "predict_tool_steps": None,
        "raw_tool_blocks": sum(point["contains_tool_heading"] for point in points),
        "datasets": sorted(datasets),
        "bad_point_count": bad_point_count,
        "selected_subset_count": len(points),
    }
    return points, metadata


def group_values(points: list[dict]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {CATEGORY_REASONING: [], CATEGORY_TOOL: []}
    for point in points:
        grouped[point["category"]].append(point["similarity"])
    return grouped


def describe_plot_transform(transform_metadata: dict) -> str | None:
    if not transform_metadata:
        return None

    notes: list[str] = []
    if "reasoning_shift_applied" in transform_metadata:
        peak_x = transform_metadata["reasoning_peak_x_before_shift"]
        shift = transform_metadata["reasoning_shift_applied"]
        notes.append(
            f"Reasoning similarities are shifted by {shift:+.4f} so the density peak moves from x={peak_x:.4f} to x=0."
        )
    if transform_metadata.get("tool_mirrored_about_zero"):
        notes.append("Tool similarities are mirrored about x=0.")
    if "tool_peak_x_before_shift" in transform_metadata:
        tool_peak_x = transform_metadata["tool_peak_x_before_shift"]
        tool_peak_shift = transform_metadata["tool_peak_shift_applied"]
        notes.append(
            f"Tool similarities are shifted by {tool_peak_shift:+.4f} so the density peak moves from x={tool_peak_x:.4f} to x=0."
        )
    if "tool_shift_applied" in transform_metadata:
        tool_shift = transform_metadata["tool_shift_applied"]
        notes.append(f"Tool similarities are shifted by {tool_shift:+.4f} along the x-axis.")
    if transform_metadata.get("reasoning_mirrored_about_zero"):
        notes.append("Reasoning similarities are mirrored about x=0.")
    if transform_metadata.get("reasoning_density_symmetrized_about_zero"):
        notes.append("Reasoning density is symmetrized about x=0 after alignment.")
    return " ".join(notes) if notes else None


def build_summary(points: list[dict], input_path: Path, metadata: dict) -> dict:
    grouped = group_values(points)
    reasoning_stats = build_stats(grouped[CATEGORY_REASONING])
    tool_stats = build_stats(grouped[CATEGORY_TOOL])
    source_format = metadata.get("source_format", "inference_json")
    transform_note = describe_plot_transform(metadata.get("plot_transform") or {})

    summary = {
        "input_path": str(input_path),
        "source_format": source_format,
        "reasoning_only_block_stats": reasoning_stats,
        "tool_related_block_stats": tool_stats,
        "difference_reasoning_minus_tool": {
            "mean": reasoning_stats["mean"] - tool_stats["mean"],
            "median": reasoning_stats["median"] - tool_stats["median"],
        },
    }

    if source_format == "demo_csv":
        summary.update(
            {
                "selected_subset_count": metadata["selected_subset_count"],
                "raw_tool_blocks": metadata["raw_tool_blocks"],
                "datasets": metadata["datasets"],
                "bad_point_count": metadata["bad_point_count"],
                "note": (
                    "This plot is drawn from a curated demo CSV subset such as "
                    "threshold_separation_demo_t010_categorical.csv, not the full inference distribution."
                ),
            }
        )
        if metadata.get("plot_transform"):
            summary["plot_transform"] = metadata["plot_transform"]
            if transform_note:
                summary["note"] += f" {transform_note}"
        return summary

    summary.update(
        {
            "num_examples": metadata["num_examples"],
            "num_points_excluding_final_response": len(points),
            "predict_tool_steps": metadata["predict_tool_steps"],
            "raw_tool_blocks": metadata["raw_tool_blocks"],
            "note": (
                "Cosine similarity is logged at the first trigger inside each raw generation block. "
                "Tool similarity here is approximated by raw blocks that contain a tool heading."
            ),
        }
    )
    if metadata.get("plot_transform"):
        summary["plot_transform"] = metadata["plot_transform"]
        if transform_note:
            summary["note"] += f" {transform_note}"
    return summary


def compute_kde_bandwidth(values: np.ndarray, bandwidth_scale: float) -> float:
    if values.size == 1:
        return max(0.015 * bandwidth_scale, 1e-3)

    std = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    iqr_sigma = float((q75 - q25) / 1.34) if q75 > q25 else 0.0
    sigma = min(item for item in (std, iqr_sigma) if item > 0) if (std > 0 or iqr_sigma > 0) else 0.0
    if sigma <= 0:
        sigma = max(abs(float(values.mean())) * 0.05, 0.01)

    bandwidth = 0.9 * sigma * (values.size ** (-1.0 / 5.0)) * bandwidth_scale
    return max(float(bandwidth), 1e-3)


def compute_density_curve(
    values: list[float],
    grid: np.ndarray,
    bandwidth_scale: float,
) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=float)
    bandwidth = compute_kde_bandwidth(array, bandwidth_scale)
    z_values = (grid[:, None] - array[None, :]) / bandwidth
    density = np.exp(-0.5 * z_values * z_values).sum(axis=1)
    density /= array.size * bandwidth * math.sqrt(2.0 * math.pi)
    return density, bandwidth


def build_grid_from_values(values: list[float], grid_size: int, center_on_zero: bool = False) -> np.ndarray:
    all_values = np.asarray(values, dtype=float)
    minimum = float(all_values.min())
    maximum = float(all_values.max())
    span = maximum - minimum
    padding = max(span * 0.08, 0.02)

    minimum -= padding
    maximum += padding

    if center_on_zero:
        bound = max(abs(minimum), abs(maximum))
        if grid_size % 2 == 0:
            grid_size += 1
        return np.linspace(-bound, bound, grid_size)

    return np.linspace(minimum, maximum, grid_size)


def build_grid(grouped: dict[str, list[float]], grid_size: int, center_on_zero: bool = False) -> np.ndarray:
    return build_grid_from_values(grouped[CATEGORY_REASONING] + grouped[CATEGORY_TOOL], grid_size, center_on_zero)


def estimate_density_peak_x(values: list[float], bandwidth_scale: float) -> float:
    peak_grid = build_grid_from_values(values, grid_size=4097)
    density, _ = compute_density_curve(values, peak_grid, bandwidth_scale)
    return float(peak_grid[int(np.argmax(density))])


def transform_points(
    points: list[dict],
    bandwidth_scale: float,
    align_reasoning_peak_to_zero: bool,
    mirror_reasoning_around_zero: bool,
    align_tool_peak_to_zero: bool,
    mirror_tool_around_zero: bool,
    tool_shift: float,
) -> tuple[list[dict], dict]:
    if not (
        align_reasoning_peak_to_zero
        or mirror_reasoning_around_zero
        or align_tool_peak_to_zero
        or mirror_tool_around_zero
        or tool_shift != 0.0
    ):
        return points, {}

    transformed_points = [dict(point) for point in points]
    for point in transformed_points:
        point["original_similarity"] = point["similarity"]

    transform_metadata: dict[str, float | bool] = {}

    if align_reasoning_peak_to_zero:
        reasoning_values = [point["similarity"] for point in transformed_points if point["category"] == CATEGORY_REASONING]
        reasoning_peak_x = estimate_density_peak_x(reasoning_values, bandwidth_scale)
        reasoning_shift = -reasoning_peak_x
        for point in transformed_points:
            if point["category"] == CATEGORY_REASONING:
                point["similarity"] += reasoning_shift
        transform_metadata["reasoning_peak_x_before_shift"] = reasoning_peak_x
        transform_metadata["reasoning_shift_applied"] = reasoning_shift

    if mirror_reasoning_around_zero:
        for point in transformed_points:
            if point["category"] == CATEGORY_REASONING:
                point["similarity"] = -point["similarity"]
        transform_metadata["reasoning_mirrored_about_zero"] = True

    if mirror_tool_around_zero:
        for point in transformed_points:
            if point["category"] == CATEGORY_TOOL:
                point["similarity"] = -point["similarity"]
        transform_metadata["tool_mirrored_about_zero"] = True

    if align_tool_peak_to_zero:
        tool_values = [point["similarity"] for point in transformed_points if point["category"] == CATEGORY_TOOL]
        tool_peak_x = estimate_density_peak_x(tool_values, bandwidth_scale)
        tool_peak_shift = -tool_peak_x
        for point in transformed_points:
            if point["category"] == CATEGORY_TOOL:
                point["similarity"] += tool_peak_shift
        transform_metadata["tool_peak_x_before_shift"] = tool_peak_x
        transform_metadata["tool_peak_shift_applied"] = tool_peak_shift

    if tool_shift != 0.0:
        for point in transformed_points:
            if point["category"] == CATEGORY_TOOL:
                point["similarity"] += tool_shift
        transform_metadata["tool_shift_applied"] = tool_shift

    return transformed_points, transform_metadata


def plot_distribution(
    points: list[dict],
    summary: dict,
    output_path: Path,
    title: str | None,
    x_label: str | None,
    y_label: str | None,
    grid_size: int,
    bandwidth_scale: float,
    note_text: str | None,
    threshold_line: float | None,
    zero_line: bool,
    zero_line_label: str,
    symmetrize_reasoning_density: bool,
    hide_legend: bool,
) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = group_values(points)
    grid = build_grid(
        grouped,
        grid_size,
        center_on_zero=bool(summary.get("plot_transform")) or symmetrize_reasoning_density,
    )

    categories = [CATEGORY_REASONING, CATEGORY_TOOL]
    colors = {
        CATEGORY_REASONING: "#2E6F95",
        CATEGORY_TOOL: "#D1495B",
    }
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
    bandwidths: dict[str, float] = {}

    for category in categories:
        density, bandwidth = compute_density_curve(grouped[category], grid, bandwidth_scale)
        if symmetrize_reasoning_density and category == CATEGORY_REASONING:
            density = 0.5 * (density + density[::-1])
        bandwidths[category] = bandwidth

        ax.plot(
            grid,
            density,
            color=colors[category],
            linewidth=2.7,
            label=category,
        )
        ax.fill_between(grid, density, color=colors[category], alpha=0.12)

    if threshold_line is not None:
        ax.axvline(
            threshold_line,
            color="#D1495B",
            linewidth=1.9,
            linestyle="--",
            alpha=0.95,
        )

    if zero_line:
        ax.axvline(
            0.0,
            color="#4A4A4A",
            linewidth=1.8,
            linestyle=":",
            alpha=0.95,
        )
        ax.annotate(
            zero_line_label,
            xy=(0.0, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            color="#4A4A4A",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONT_SIZE)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    if not hide_legend:
        ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)

    if note_text:
        fig.text(0.5, 0.01, note_text, ha="center", va="bottom", fontsize=NOTE_FONT_SIZE)
        fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.55)
    else:
        fig.tight_layout(pad=0.55)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return bandwidths


def write_points_csv(points: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_original_similarity = any("original_similarity" in point for point in points)
    if has_original_similarity:
        header = (
            "example_index,round_index,category,similarity,original_similarity,contains_tool_heading\n"
        )
    else:
        header = "example_index,round_index,category,similarity,contains_tool_heading\n"
    lines = [header]
    for point in points:
        if has_original_similarity:
            lines.append(
                f"{point['example_index']},{point['round_index']},{point['category']},"
                f"{point['similarity']:.12f},{point['original_similarity']:.12f},"
                f"{int(point['contains_tool_heading'])}\n"
            )
        else:
            lines.append(
                f"{point['example_index']},{point['round_index']},{point['category']},"
                f"{point['similarity']:.12f},{int(point['contains_tool_heading'])}\n"
            )
    output_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reasoning vs tool-related cosine-similarity distributions as two curves."
    )
    parser.add_argument("input_path", type=Path, help="Path to an inference result JSON file or a demo CSV.")
    parser.add_argument(
        "--input-format",
        choices=("auto", "inference_json", "demo_csv"),
        default="auto",
        help="How to interpret the input. Defaults to auto-detect by file suffix.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix without extension. Defaults to evaluate/figures/<input_stem>_similarity_distribution.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Omit the plot title entirely.",
    )
    parser.add_argument(
        "--hide-note",
        action="store_true",
        help="Omit the bottom explanatory note.",
    )
    parser.add_argument(
        "--threshold-line",
        type=float,
        default=None,
        help="Optional x-position for a vertical threshold line.",
    )
    parser.add_argument(
        "--zero-line",
        action="store_true",
        help="Draw a vertical reference line at x=0.",
    )
    parser.add_argument(
        "--zero-line-label",
        type=str,
        default="x=0",
        help="Label shown next to the x=0 reference line.",
    )
    parser.add_argument(
        "--align-reasoning-peak-to-zero",
        action="store_true",
        help="Shift the reasoning curve so its KDE peak sits at x=0.",
    )
    parser.add_argument(
        "--mirror-reasoning-around-zero",
        action="store_true",
        help="Mirror the reasoning similarities about x=0 after any reasoning shift.",
    )
    parser.add_argument(
        "--mirror-tool-around-zero",
        action="store_true",
        help="Mirror the tool-related curve about x=0.",
    )
    parser.add_argument(
        "--align-tool-peak-to-zero",
        action="store_true",
        help="Shift the tool-related curve so its KDE peak sits at x=0 after any tool mirroring.",
    )
    parser.add_argument(
        "--tool-shift",
        type=float,
        default=0.0,
        help="Add a constant shift to the tool-related similarities after any mirroring step.",
    )
    parser.add_argument(
        "--symmetrize-reasoning-density",
        action="store_true",
        help="Make the plotted reasoning density symmetric about x=0 after any reasoning shift.",
    )
    parser.add_argument(
        "--x-label",
        type=str,
        default=None,
        help="Override the x-axis label.",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default=None,
        help="Override the y-axis label.",
    )
    parser.add_argument(
        "--hide-x-label",
        action="store_true",
        help="Omit the x-axis label.",
    )
    parser.add_argument(
        "--hide-y-label",
        action="store_true",
        help="Omit the y-axis label.",
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Omit the legend from the plot.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=400,
        help="Number of x-axis points used to draw the density curves.",
    )
    parser.add_argument(
        "--bandwidth-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to KDE bandwidth. Larger values produce smoother curves.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    default_prefix = Path("SteeringMark/evaluate/figures") / f"{input_path.stem}_similarity_distribution"
    output_prefix = args.output_prefix or default_prefix
    default_title = f"{input_path.stem}: reasoning vs tool-related similarity distribution"
    title = None if args.hide_title else (args.title or default_title)

    input_format = args.input_format
    if input_format == "auto":
        input_format = "demo_csv" if input_path.suffix.lower() == ".csv" else "inference_json"

    if input_format == "demo_csv":
        points, metadata = load_demo_csv_points(input_path)
    else:
        points, metadata = load_points(input_path)
        metadata["source_format"] = "inference_json"

    points, transform_metadata = transform_points(
        points=points,
        bandwidth_scale=args.bandwidth_scale,
        align_reasoning_peak_to_zero=args.align_reasoning_peak_to_zero,
        mirror_reasoning_around_zero=args.mirror_reasoning_around_zero,
        align_tool_peak_to_zero=args.align_tool_peak_to_zero,
        mirror_tool_around_zero=args.mirror_tool_around_zero,
        tool_shift=args.tool_shift,
    )
    if args.symmetrize_reasoning_density:
        transform_metadata["reasoning_density_symmetrized_about_zero"] = True
    if transform_metadata:
        metadata["plot_transform"] = transform_metadata

    summary = build_summary(points, input_path, metadata)
    default_x_label = "Aligned Similarity" if transform_metadata else "Cosine Similarity"
    x_label = None if args.hide_x_label else (args.x_label or default_x_label)
    y_label = None if args.hide_y_label else (args.y_label or "Distribution")

    plot_path = output_prefix.with_suffix(".png")
    summary_path = output_prefix.with_suffix(".summary.json")
    csv_path = output_prefix.with_suffix(".points.csv")

    bandwidths = plot_distribution(
        points=points,
        summary=summary,
        output_path=plot_path,
        title=title,
        x_label=x_label,
        y_label=y_label,
        grid_size=args.grid_size,
        bandwidth_scale=args.bandwidth_scale,
        note_text=None if args.hide_note else summary["note"],
        threshold_line=args.threshold_line,
        zero_line=args.zero_line,
        zero_line_label=args.zero_line_label,
        symmetrize_reasoning_density=args.symmetrize_reasoning_density,
        hide_legend=args.hide_legend,
    )
    summary["kde_bandwidths"] = bandwidths

    summary_path.write_text(json.dumps(summary, indent=2))
    write_points_csv(points, csv_path)

    print(f"Saved distribution plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved point data to: {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
