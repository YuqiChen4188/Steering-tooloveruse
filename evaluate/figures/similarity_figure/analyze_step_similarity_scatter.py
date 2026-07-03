import argparse
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


def build_summary(points: list[dict], input_path: Path, metadata: dict) -> dict:
    grouped: dict[str, list[float]] = {CATEGORY_REASONING: [], CATEGORY_TOOL: []}
    for point in points:
        grouped[point["category"]].append(point["similarity"])

    reasoning_stats = build_stats(grouped[CATEGORY_REASONING])
    tool_stats = build_stats(grouped[CATEGORY_TOOL])

    return {
        "input_path": str(input_path),
        "num_examples": metadata["num_examples"],
        "num_points_excluding_final_response": len(points),
        "predict_tool_steps": metadata["predict_tool_steps"],
        "raw_tool_blocks": metadata["raw_tool_blocks"],
        "reasoning_only_block_stats": reasoning_stats,
        "tool_related_block_stats": tool_stats,
        "difference_reasoning_minus_tool": {
            "mean": reasoning_stats["mean"] - tool_stats["mean"],
            "median": reasoning_stats["median"] - tool_stats["median"],
        },
        "note": (
            "Cosine similarity is logged at the first trigger inside each raw generation block. "
            "This file does not contain a separately logged cosine for parsed Tool Step headings, "
            "so tool similarity here is approximated by raw blocks that contain a tool heading."
        ),
    }


def plot_scatter(points: list[dict], summary: dict, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    categories = [CATEGORY_REASONING, CATEGORY_TOOL]
    x_positions = {CATEGORY_REASONING: 0, CATEGORY_TOOL: 1}
    colors = {
        CATEGORY_REASONING: "#2E6F95",
        CATEGORY_TOOL: "#D1495B",
    }

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for category in categories:
        y_values = [point["similarity"] for point in points if point["category"] == category]
        x_center = x_positions[category]
        jitter = rng.uniform(-0.16, 0.16, size=len(y_values))
        ax.scatter(
            np.full(len(y_values), x_center) + jitter,
            y_values,
            s=22,
            alpha=0.55,
            c=colors[category],
            edgecolors="none",
            label=category,
        )

    reasoning_stats = summary["reasoning_only_block_stats"]
    tool_stats = summary["tool_related_block_stats"]
    stats_by_category = {
        CATEGORY_REASONING: reasoning_stats,
        CATEGORY_TOOL: tool_stats,
    }

    for category in categories:
        x_center = x_positions[category]
        stats = stats_by_category[category]
        ax.hlines(
            y=stats["mean"],
            xmin=x_center - 0.22,
            xmax=x_center + 0.22,
            colors=colors[category],
            linewidth=2.0,
        )
        ax.hlines(
            y=stats["median"],
            xmin=x_center - 0.18,
            xmax=x_center + 0.18,
            colors=colors[category],
            linewidth=1.5,
            linestyles="--",
        )
        ax.text(
            x_center,
            stats["max"] + 0.018,
            f"n={stats['count']}\nmean={stats['mean']:.4f}\nmedian={stats['median']:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors[category],
        )

    ax.set_xticks([x_positions[category] for category in categories], categories)
    ax.set_ylabel("Mean cosine similarity at first trigger")
    ax.set_xlabel("Block type")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    note = (
        "Excludes blocks that start with Final Response. "
        "Tool group is approximated by raw blocks containing a tool heading."
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_points_csv(points: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "example_index,round_index,category,similarity,contains_tool_heading\n"
    lines = [header]
    for point in points:
        lines.append(
            f"{point['example_index']},{point['round_index']},{point['category']},"
            f"{point['similarity']:.12f},{int(point['contains_tool_heading'])}\n"
        )
    output_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cosine similarity between reasoning-only and tool-related blocks."
    )
    parser.add_argument("input_json", type=Path, help="Path to an inference result JSON file.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix without extension. Defaults to evaluate/figures/<input_stem>_similarity_scatter.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    args = parser.parse_args()

    input_path = args.input_json
    default_prefix = Path("SteeringMark/evaluate/figures") / f"{input_path.stem}_similarity_scatter"
    output_prefix = args.output_prefix or default_prefix
    title = args.title or f"{input_path.stem}: reasoning vs tool-related cosine similarity"

    points, metadata = load_points(input_path)
    summary = build_summary(points, input_path, metadata)

    plot_path = output_prefix.with_suffix(".png")
    summary_path = output_prefix.with_suffix(".summary.json")
    csv_path = output_prefix.with_suffix(".points.csv")

    plot_scatter(points, summary, plot_path, title)
    summary_path.write_text(json.dumps(summary, indent=2))
    write_points_csv(points, csv_path)

    print(f"Saved scatter plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved point data to: {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
