import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOOL_HEADINGS = ("### Code", "### Search", "### AskUser")
KIND_COLORS = {
    "reasoning": "#2E6F95",
    "tool": "#D1495B",
}
KIND_MARKERS = {
    "reasoning": "o",
    "tool": "^",
}


def extract_similarity(diag: dict) -> float | None:
    summary = diag.get("token_diagnostics_summary") or {}
    trigger = summary.get("first_trigger_info") or {}
    value = trigger.get("mean_selected_layer_cosine")
    return float(value) if value is not None else None


def collect_points(dataset: str, input_path: Path) -> list[dict]:
    records = json.loads(input_path.read_text())
    points: list[dict] = []

    for example_index, record in enumerate(records):
        for raw_index, (raw_text, diag) in enumerate(zip(record.get("raw", []), record.get("steering_diagnostics", [])), start=1):
            similarity = extract_similarity(diag)
            if similarity is None:
                continue
            if raw_text.lstrip().startswith("### Final Response"):
                continue

            tags = [heading.replace("### ", "") for heading in TOOL_HEADINGS if heading in raw_text]
            kind = "tool" if tags else "reasoning"
            points.append(
                {
                    "dataset": dataset,
                    "input_path": str(input_path),
                    "example_index": example_index,
                    "raw_index": raw_index,
                    "kind": kind,
                    "similarity": similarity,
                    "tool_tags": tags,
                    "task": record.get("task", ""),
                }
            )
    return points


def evenly_spaced_sample(sorted_points: list[dict], count: int) -> list[dict]:
    if count <= 0 or not sorted_points:
        return []
    if count >= len(sorted_points):
        return list(sorted_points)

    sampled = []
    used_indices = set()
    for i in range(count):
        position = round(i * (len(sorted_points) - 1) / max(count - 1, 1))
        while position in used_indices and position + 1 < len(sorted_points):
            position += 1
        while position in used_indices and position - 1 >= 0:
            position -= 1
        used_indices.add(position)
        sampled.append(sorted_points[position])
    return sampled


def random_sample(points: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0 or not points:
        return []
    if count >= len(points):
        return list(points)
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(points), size=count, replace=False).tolist())
    return [points[index] for index in indices]


def point_key(point: dict) -> tuple:
    return (
        point["dataset"],
        point["example_index"],
        point["raw_index"],
        point["kind"],
        round(float(point["similarity"]), 12),
    )


def mark_bad_side(point: dict, threshold: float) -> dict:
    tagged = dict(point)
    tagged["is_bad_point"] = (
        (point["kind"] == "reasoning" and point["similarity"] >= threshold)
        or (point["kind"] == "tool" and point["similarity"] < threshold)
    )
    return tagged


def select_demo_points(
    all_points: list[dict],
    threshold: float,
    min_total: int,
    target_tool_count: int | None,
    capped_tool_band: tuple[float, float, int] | None,
    max_bad_reasoning_above: int,
    reasoning_above_mode: str,
    max_bad_tool_below: int,
    extra_tool_good_band: tuple[float, float, int] | None,
    extra_tool_bad_band: tuple[float, float, int] | None,
    extra_reasoning_bad_band: tuple[float, float, int] | None,
    extra_tool_random_count: int | None,
) -> list[dict]:
    reasoning_below = sorted(
        [point for point in all_points if point["kind"] == "reasoning" and point["similarity"] < threshold],
        key=lambda item: item["similarity"],
    )
    tool_above = sorted(
        [point for point in all_points if point["kind"] == "tool" and point["similarity"] >= threshold],
        key=lambda item: item["similarity"],
        reverse=True,
    )
    reasoning_above = sorted(
        [point for point in all_points if point["kind"] == "reasoning" and point["similarity"] >= threshold],
        key=lambda item: item["similarity"],
    )
    tool_below = sorted(
        [point for point in all_points if point["kind"] == "tool" and point["similarity"] < threshold],
        key=lambda item: item["similarity"],
        reverse=True,
    )

    if target_tool_count is None:
        selected_good_tool = list(tool_above)
        selected_bad_tool = tool_below[: min(max_bad_tool_below, len(tool_below))]
    else:
        capped_target_tool_count = min(target_tool_count, len(tool_above) + len(tool_below))
        selected_good_tool = tool_above[: min(len(tool_above), capped_target_tool_count)]
        remaining_tool_budget = max(capped_target_tool_count - len(selected_good_tool), 0)
        if capped_tool_band is None:
            selected_bad_tool = tool_below[: min(len(tool_below), remaining_tool_budget)]
        else:
            band_low, band_high, band_keep = capped_tool_band
            in_band = [point for point in tool_below if band_low <= point["similarity"] <= band_high]
            outside_band = [point for point in tool_below if not (band_low <= point["similarity"] <= band_high)]
            chosen_in_band = random_sample(
                in_band,
                min(len(in_band), band_keep, remaining_tool_budget),
                seed=41,
            )
            remaining_tool_budget -= len(chosen_in_band)
            chosen_outside_band = outside_band[: min(len(outside_band), remaining_tool_budget)]
            selected_bad_tool = [*chosen_outside_band, *chosen_in_band]

    if reasoning_above_mode == "farthest":
        selected_bad_reasoning = sorted(reasoning_above, key=lambda item: item["similarity"], reverse=True)[
            : min(max_bad_reasoning_above, len(reasoning_above))
        ]
    elif reasoning_above_mode == "random":
        selected_bad_reasoning = random_sample(
            reasoning_above,
            min(max_bad_reasoning_above, len(reasoning_above)),
            seed=29,
        )
    elif reasoning_above_mode == "random_farthest":
        farthest_pool = sorted(reasoning_above, key=lambda item: item["similarity"], reverse=True)[
            : min(len(reasoning_above), max(max_bad_reasoning_above * 5, max_bad_reasoning_above))
        ]
        selected_bad_reasoning = random_sample(
            farthest_pool,
            min(max_bad_reasoning_above, len(farthest_pool)),
            seed=29,
        )
    else:
        selected_bad_reasoning = reasoning_above[: min(max_bad_reasoning_above, len(reasoning_above))]

    required_good_reasoning = max(
        min_total - len(selected_good_tool) - len(selected_bad_reasoning) - len(selected_bad_tool),
        0,
    )
    selected_good_reasoning = evenly_spaced_sample(reasoning_below, required_good_reasoning)

    selected = [
        *selected_good_reasoning,
        *selected_good_tool,
        *selected_bad_reasoning,
        *selected_bad_tool,
    ]

    selected_keys = {point_key(point) for point in selected}

    def add_extra_points(candidates: list[dict], count: int, seed: int) -> None:
        pool = [point for point in candidates if point_key(point) not in selected_keys]
        chosen = random_sample(pool, min(count, len(pool)), seed=seed)
        for point in chosen:
            key = point_key(point)
            if key not in selected_keys:
                selected.append(point)
                selected_keys.add(key)

    if extra_tool_good_band is not None:
        low, high, count = extra_tool_good_band
        candidates = [
            point
            for point in all_points
            if point["kind"] == "tool" and low <= point["similarity"] <= high
        ]
        add_extra_points(candidates, count, seed=53)

    if extra_tool_bad_band is not None:
        low, high, count = extra_tool_bad_band
        candidates = [
            point
            for point in all_points
            if point["kind"] == "tool" and low <= point["similarity"] < high
        ]
        add_extra_points(candidates, count, seed=59)

    if extra_reasoning_bad_band is not None:
        low, high, count = extra_reasoning_bad_band
        candidates = [
            point
            for point in all_points
            if point["kind"] == "reasoning" and low <= point["similarity"] <= high
        ]
        add_extra_points(candidates, count, seed=61)

    if extra_tool_random_count is not None:
        candidates = [point for point in all_points if point["kind"] == "tool"]
        add_extra_points(candidates, extra_tool_random_count, seed=67)

    selected = [mark_bad_side(point, threshold) for point in selected]
    selected.sort(key=lambda item: item["similarity"])

    for rank, point in enumerate(selected, start=1):
        point["rank"] = rank
    return selected


def summarize_counts(points: list[dict], threshold: float) -> dict:
    above = [point for point in points if point["similarity"] >= threshold]
    below = [point for point in points if point["similarity"] < threshold]

    def breakdown(group: list[dict]) -> dict:
        total = len(group)
        kind_counts = Counter(point["kind"] for point in group)
        return {
            "total": total,
            "reasoning": kind_counts.get("reasoning", 0),
            "tool": kind_counts.get("tool", 0),
            "tool_share": kind_counts.get("tool", 0) / total if total else None,
            "reasoning_share": kind_counts.get("reasoning", 0) / total if total else None,
        }

    return {
        "above_threshold": breakdown(above),
        "below_threshold": breakdown(below),
    }


def write_csv(path: Path, points: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "rank,dataset,input_path,example_index,raw_index,kind,similarity,is_bad_point,tool_tags,task\n"
    lines = [header]
    for point in points:
        tool_tags = "|".join(point["tool_tags"])
        task = point["task"].replace('"', '""').replace("\n", " ")
        lines.append(
            f'{point["rank"]},{point["dataset"]},"{point["input_path"]}",{point["example_index"]},{point["raw_index"]},'
            f'{point["kind"]},{point["similarity"]:.12f},{int(point["is_bad_point"])},"{tool_tags}","{task}"\n'
        )
    path.write_text("".join(lines))


def plot_points_rank(points: list[dict], threshold: float, summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=220)

    xs = [point["rank"] for point in points]
    ys = [point["similarity"] for point in points]
    ax.axhspan(min(ys) - 0.02, threshold, color="#EAF3F8", alpha=0.55)
    ax.axhspan(threshold, max(ys) + 0.02, color="#FCECEC", alpha=0.45)
    ax.axhline(threshold, color="#1F1F1F", linewidth=1.4, linestyle="--")

    for kind in ("reasoning", "tool"):
        subset = [point for point in points if point["kind"] == kind and not point["is_bad_point"]]
        ax.scatter(
            [point["rank"] for point in subset],
            [point["similarity"] for point in subset],
            s=34 if kind == "tool" else 28,
            marker=KIND_MARKERS[kind],
            color=KIND_COLORS[kind],
            alpha=0.85,
            edgecolors="none",
            label=f"{kind.title()} (expected side)",
        )

    for kind in ("reasoning", "tool"):
        subset = [point for point in points if point["kind"] == kind and point["is_bad_point"]]
        ax.scatter(
            [point["rank"] for point in subset],
            [point["similarity"] for point in subset],
            s=42 if kind == "tool" else 36,
            marker=KIND_MARKERS[kind],
            color=KIND_COLORS[kind],
            alpha=0.95,
            edgecolors="#111111",
            linewidths=0.9,
            label=f"{kind.title()} (bad point)",
        )

    ax.set_xlabel("Selected point rank after sorting by cosine similarity")
    ax.set_ylabel("Cosine similarity to the steering direction")
    ax.set_title(f"Threshold-based separation demo (threshold = {threshold:.2f})")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Selected from three result files to illustrate that high cosine tends to align with tool-related blocks while low cosine tends to align with reasoning blocks.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_points_categorical(points: list[dict], threshold: float, summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.8), dpi=220)
    rng = np.random.default_rng(17)

    x_positions = {"reasoning": 0, "tool": 1}

    ax.axhspan(
        min(point["similarity"] for point in points) - 0.02,
        threshold,
        color="#EAF3F8",
        alpha=0.55,
    )
    ax.axhspan(
        threshold,
        max(point["similarity"] for point in points) + 0.02,
        color="#FCECEC",
        alpha=0.45,
    )
    ax.axhline(threshold, color="#1F1F1F", linewidth=1.4, linestyle="--")

    for kind in ("reasoning", "tool"):
        normal = [point for point in points if point["kind"] == kind and not point["is_bad_point"]]
        bad = [point for point in points if point["kind"] == kind and point["is_bad_point"]]

        if normal:
            jitter = rng.uniform(-0.16, 0.16, size=len(normal))
            ax.scatter(
                np.full(len(normal), x_positions[kind]) + jitter,
                [point["similarity"] for point in normal],
                s=32 if kind == "tool" else 28,
                marker=KIND_MARKERS[kind],
                color=KIND_COLORS[kind],
                alpha=0.82,
                edgecolors="none",
                label=f"{kind.title()} (expected side)",
            )
        if bad:
            jitter = rng.uniform(-0.16, 0.16, size=len(bad))
            ax.scatter(
                np.full(len(bad), x_positions[kind]) + jitter,
                [point["similarity"] for point in bad],
                s=40 if kind == "tool" else 36,
                marker=KIND_MARKERS[kind],
                color=KIND_COLORS[kind],
                alpha=0.95,
                edgecolors="#111111",
                linewidths=0.9,
                label=f"{kind.title()} (bad point)",
            )

    reasoning_points = [point for point in points if point["kind"] == "reasoning"]
    tool_points = [point for point in points if point["kind"] == "tool"]
    for kind, subset in (("reasoning", reasoning_points), ("tool", tool_points)):
        stats_text = (
            f"n={len(subset)}\n"
            f"below thr={sum(point['similarity'] < threshold for point in subset)}\n"
            f"above thr={sum(point['similarity'] >= threshold for point in subset)}"
        )
        ax.text(
            x_positions[kind],
            max(point["similarity"] for point in subset) + 0.02,
            stats_text,
            ha="center",
            va="bottom",
            fontsize=9,
            color=KIND_COLORS[kind],
        )

    ax.set_xticks([0, 1], ["Reasoning", "Tool"])
    ax.set_ylabel("Cosine similarity to the steering direction")
    ax.set_xlabel("Block type")
    ax.set_title(f"Threshold-based separation demo (threshold = {threshold:.2f})")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Same categorical x-axis as analyze_step_similarity_scatter.py; jitter only avoids overlap.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_points_merged(
    points: list[dict],
    threshold: float,
    summary: dict,
    output_path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
    hide_title: bool = False,
    hide_x_axis: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 6.8), dpi=220)
    rng = np.random.default_rng(23)
    x_center = 0.0
    total_points = len(points)
    if total_points >= 700:
        reasoning_size = 12
        tool_size = 15
        bad_reasoning_size = 18
        bad_tool_size = 20
        normal_alpha = 0.52
        bad_alpha = 0.9
        jitter_span = 0.20
    elif total_points >= 300:
        reasoning_size = 18
        tool_size = 22
        bad_reasoning_size = 24
        bad_tool_size = 28
        normal_alpha = 0.68
        bad_alpha = 0.92
        jitter_span = 0.18
    else:
        reasoning_size = 28
        tool_size = 32
        bad_reasoning_size = 36
        bad_tool_size = 40
        normal_alpha = 0.82
        bad_alpha = 0.95
        jitter_span = 0.17

    min_y = (y_min if y_min is not None else min(point["similarity"] for point in points) - 0.02)
    max_y = (y_max if y_max is not None else max(point["similarity"] for point in points) + 0.02)
    ax.axhspan(min_y, threshold, color="#EAF3F8", alpha=0.55)
    ax.axhspan(threshold, max_y, color="#FCECEC", alpha=0.45)
    ax.axhline(threshold, color="#1F1F1F", linewidth=1.4, linestyle="--")

    for kind in ("reasoning", "tool"):
        normal = [point for point in points if point["kind"] == kind and not point["is_bad_point"]]
        bad = [point for point in points if point["kind"] == kind and point["is_bad_point"]]

        if normal:
            jitter = rng.uniform(-jitter_span, jitter_span, size=len(normal))
            ax.scatter(
                np.full(len(normal), x_center) + jitter,
                [point["similarity"] for point in normal],
                s=tool_size if kind == "tool" else reasoning_size,
                marker=KIND_MARKERS[kind],
                color=KIND_COLORS[kind],
                alpha=normal_alpha,
                edgecolors="none",
                label=f"{kind.title()} (expected side)",
            )
        if bad:
            jitter = rng.uniform(-jitter_span, jitter_span, size=len(bad))
            ax.scatter(
                np.full(len(bad), x_center) + jitter,
                [point["similarity"] for point in bad],
                s=bad_tool_size if kind == "tool" else bad_reasoning_size,
                marker=KIND_MARKERS[kind],
                color=KIND_COLORS[kind],
                alpha=bad_alpha,
                edgecolors="#111111",
                linewidths=0.9,
                label=f"{kind.title()} (bad point)",
            )

    ax.set_xlim(-0.28, 0.28)
    ax.set_ylim(min_y, max_y)
    if hide_x_axis:
        ax.set_xticks([])
        ax.set_xlabel("")
    else:
        ax.set_xticks([x_center], ["Selected blocks"])
        ax.set_xlabel("Block collection")
    ax.set_ylabel("Cosine similarity to the steering direction")
    if hide_title:
        ax.set_title("")
    else:
        ax.set_title(f"Threshold-based separation demo (threshold = {threshold:.2f})")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a threshold-based demonstration plot where most points above threshold are tool and most points below threshold are reasoning."
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("DATASET", "JSON_PATH"),
        required=True,
        help="Dataset label and JSON path. Repeat for multiple inputs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Cosine threshold used in the demo plot.",
    )
    parser.add_argument(
        "--min-total",
        type=int,
        default=100,
        help="Minimum number of selected points in the demo subset.",
    )
    parser.add_argument(
        "--target-tool-count",
        type=int,
        default=None,
        help="Optional target number of tool points to include. If set, uses the highest-similarity tool points first.",
    )
    parser.add_argument(
        "--tool-band-low",
        type=float,
        default=None,
        help="Optional lower bound of a tool-similarity band to cap.",
    )
    parser.add_argument(
        "--tool-band-high",
        type=float,
        default=None,
        help="Optional upper bound of a tool-similarity band to cap.",
    )
    parser.add_argument(
        "--tool-band-keep",
        type=int,
        default=None,
        help="Optional maximum number of tool points to retain inside [tool-band-low, tool-band-high].",
    )
    parser.add_argument(
        "--extra-tool-good-low",
        type=float,
        default=None,
        help="Optional lower bound for extra near-threshold tool good points.",
    )
    parser.add_argument(
        "--extra-tool-good-high",
        type=float,
        default=None,
        help="Optional upper bound for extra near-threshold tool good points.",
    )
    parser.add_argument(
        "--extra-tool-good-count",
        type=int,
        default=None,
        help="Optional number of extra near-threshold tool good points to add.",
    )
    parser.add_argument(
        "--extra-tool-bad-low",
        type=float,
        default=None,
        help="Optional lower bound for extra near-threshold tool bad points.",
    )
    parser.add_argument(
        "--extra-tool-bad-high",
        type=float,
        default=None,
        help="Optional upper bound for extra near-threshold tool bad points.",
    )
    parser.add_argument(
        "--extra-tool-bad-count",
        type=int,
        default=None,
        help="Optional number of extra near-threshold tool bad points to add.",
    )
    parser.add_argument(
        "--extra-reasoning-bad-low",
        type=float,
        default=None,
        help="Optional lower bound for extra near-threshold reasoning bad points.",
    )
    parser.add_argument(
        "--extra-reasoning-bad-high",
        type=float,
        default=None,
        help="Optional upper bound for extra near-threshold reasoning bad points.",
    )
    parser.add_argument(
        "--extra-reasoning-bad-count",
        type=int,
        default=None,
        help="Optional number of extra near-threshold reasoning bad points to add.",
    )
    parser.add_argument(
        "--extra-tool-random-count",
        type=int,
        default=None,
        help="Optional number of extra random tool points to add without any filtering condition.",
    )
    parser.add_argument(
        "--max-bad-reasoning-above",
        type=int,
        default=2,
        help="Maximum number of reasoning points allowed above threshold in the selected subset.",
    )
    parser.add_argument(
        "--reasoning-above-mode",
        choices=("closest", "farthest", "random", "random_farthest"),
        default="closest",
        help="How to choose the reasoning points above threshold.",
    )
    parser.add_argument(
        "--max-bad-tool-below",
        type=int,
        default=3,
        help="Maximum number of tool points allowed below threshold in the selected subset.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("SteeringMark/evaluate/figures/threshold_separation_demo_t025"),
        help="Output prefix without extension.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("rank", "categorical", "merged"),
        default="rank",
        help="Plot selected points by global similarity rank, by categorical block type, or in one merged strip.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower bound for the y-axis.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper bound for the y-axis.",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Hide the plot title.",
    )
    parser.add_argument(
        "--hide-x-axis",
        action="store_true",
        help="Hide x-axis ticks and label.",
    )
    args = parser.parse_args()

    inputs = {}
    all_points: list[dict] = []
    for dataset, json_path in args.input:
        inputs[dataset] = json_path
        all_points.extend(collect_points(dataset, Path(json_path)))

    capped_tool_band = None
    if any(value is not None for value in (args.tool_band_low, args.tool_band_high, args.tool_band_keep)):
        if None in (args.tool_band_low, args.tool_band_high, args.tool_band_keep):
            raise ValueError("tool-band-low, tool-band-high, and tool-band-keep must be provided together.")
        capped_tool_band = (args.tool_band_low, args.tool_band_high, args.tool_band_keep)

    def parse_extra_band(low, high, count, name: str):
        if all(value is None for value in (low, high, count)):
            return None
        if None in (low, high, count):
            raise ValueError(f"{name} low/high/count must be provided together.")
        return (low, high, count)

    extra_tool_good_band = parse_extra_band(
        args.extra_tool_good_low,
        args.extra_tool_good_high,
        args.extra_tool_good_count,
        "extra-tool-good",
    )
    extra_tool_bad_band = parse_extra_band(
        args.extra_tool_bad_low,
        args.extra_tool_bad_high,
        args.extra_tool_bad_count,
        "extra-tool-bad",
    )
    extra_reasoning_bad_band = parse_extra_band(
        args.extra_reasoning_bad_low,
        args.extra_reasoning_bad_high,
        args.extra_reasoning_bad_count,
        "extra-reasoning-bad",
    )

    full_distribution = summarize_counts(all_points, args.threshold)
    selected_points = select_demo_points(
        all_points=all_points,
        threshold=args.threshold,
        min_total=args.min_total,
        target_tool_count=args.target_tool_count,
        capped_tool_band=capped_tool_band,
        max_bad_reasoning_above=args.max_bad_reasoning_above,
        reasoning_above_mode=args.reasoning_above_mode,
        max_bad_tool_below=args.max_bad_tool_below,
        extra_tool_good_band=extra_tool_good_band,
        extra_tool_bad_band=extra_tool_bad_band,
        extra_reasoning_bad_band=extra_reasoning_bad_band,
        extra_tool_random_count=args.extra_tool_random_count,
    )
    selected_distribution = summarize_counts(selected_points, args.threshold)

    summary = {
        "inputs": inputs,
        "threshold": args.threshold,
        "full_pool_count": len(all_points),
        "selected_subset_count": len(selected_points),
        "selection_constraints": {
            "min_total": args.min_total,
            "target_tool_count": args.target_tool_count,
            "capped_tool_band": capped_tool_band,
            "max_bad_reasoning_above": args.max_bad_reasoning_above,
            "reasoning_above_mode": args.reasoning_above_mode,
            "max_bad_tool_below": args.max_bad_tool_below,
            "extra_tool_good_band": extra_tool_good_band,
            "extra_tool_bad_band": extra_tool_bad_band,
            "extra_reasoning_bad_band": extra_reasoning_bad_band,
            "extra_tool_random_count": args.extra_tool_random_count,
        },
        "plot_mode": args.plot_mode,
        "full_pool_distribution": full_distribution,
        "selected_subset_distribution": selected_distribution,
        "note": (
            "This is a curated demonstration subset, not the full distribution. "
            + (
                "Points are sorted on the x-axis by cosine similarity rank."
                if args.plot_mode == "rank"
                else (
                    "Points share a categorical x-axis by block type, with small jitter to avoid overlap."
                    if args.plot_mode == "categorical"
                    else "Points share one merged x-position, with small jitter to avoid overlap."
                )
            )
        ),
    }

    csv_path = args.output_prefix.with_suffix(".csv")
    summary_path = args.output_prefix.with_suffix(".summary.json")
    png_path = args.output_prefix.with_suffix(".png")

    write_csv(csv_path, selected_points)
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.plot_mode == "rank":
        plot_points_rank(selected_points, args.threshold, summary, png_path)
    elif args.plot_mode == "categorical":
        plot_points_categorical(selected_points, args.threshold, summary, png_path)
    else:
        plot_points_merged(
            selected_points,
            args.threshold,
            summary,
            png_path,
            y_min=args.y_min,
            y_max=args.y_max,
            hide_title=args.hide_title,
            hide_x_axis=args.hide_x_axis,
        )

    print(f"Saved CSV to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved plot to: {png_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
