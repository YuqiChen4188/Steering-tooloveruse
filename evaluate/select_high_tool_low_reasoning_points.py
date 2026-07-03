import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOOL_HEADINGS = ("### Code", "### Search", "### AskUser")
DATASET_COLORS = {
    "math": "#D1495B",
    "intention": "#2E6F95",
    "time": "#3C8D5A",
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


def select_points(points: list[dict], reasoning_max: float, tool_min: float) -> tuple[list[dict], list[dict]]:
    selected_reasoning = [point for point in points if point["kind"] == "reasoning" and point["similarity"] <= reasoning_max]
    selected_tool = [point for point in points if point["kind"] == "tool" and point["similarity"] >= tool_min]
    selected_reasoning.sort(key=lambda item: item["similarity"])
    selected_tool.sort(key=lambda item: item["similarity"], reverse=True)
    return selected_reasoning, selected_tool


def write_csv(path: Path, reasoning_points: list[dict], tool_points: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "dataset,input_path,example_index,raw_index,kind,similarity,tool_tags,task\n"
    lines = [header]
    for point in [*reasoning_points, *tool_points]:
        tool_tags = "|".join(point["tool_tags"])
        task = point["task"].replace('"', '""').replace("\n", " ")
        lines.append(
            f'{point["dataset"]},"{point["input_path"]}",{point["example_index"]},{point["raw_index"]},'
            f'{point["kind"]},{point["similarity"]:.12f},"{tool_tags}","{task}"\n'
        )
    path.write_text("".join(lines))


def build_summary(
    inputs: dict[str, str],
    reasoning_points: list[dict],
    tool_points: list[dict],
    reasoning_max: float,
    tool_min: float,
) -> dict:
    counts = defaultdict(lambda: {"reasoning": 0, "tool": 0})
    tool_tag_counts = Counter()
    for point in reasoning_points:
        counts[point["dataset"]]["reasoning"] += 1
    for point in tool_points:
        counts[point["dataset"]]["tool"] += 1
        for tag in point["tool_tags"]:
            tool_tag_counts[tag] += 1

    return {
        "inputs": inputs,
        "selection_rule": {
            "reasoning_similarity_leq": reasoning_max,
            "tool_similarity_geq": tool_min,
        },
        "selected_reasoning_count": len(reasoning_points),
        "selected_tool_count": len(tool_points),
        "selected_total_count": len(reasoning_points) + len(tool_points),
        "counts_by_dataset": counts,
        "selected_tool_tag_counts": dict(tool_tag_counts),
        "note": (
            "Similarity is read from steering_diagnostics[].token_diagnostics_summary.first_trigger_info.mean_selected_layer_cosine. "
            "These are raw generation block trigger points; tool points are blocks containing ### Code / ### Search / ### AskUser."
        ),
    }


def plot_selected_points(
    reasoning_points: list[dict],
    tool_points: list[dict],
    reasoning_max: float,
    tool_min: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=220)

    x_reasoning = list(range(1, len(reasoning_points) + 1))
    gap = 8
    tool_start = len(reasoning_points) + gap
    x_tool = list(range(tool_start, tool_start + len(tool_points)))

    for x_value, point in zip(x_reasoning, reasoning_points):
        ax.scatter(
            x_value,
            point["similarity"],
            s=28,
            marker="o",
            color=DATASET_COLORS.get(point["dataset"], "#6C757D"),
            alpha=0.8,
            edgecolors="none",
        )

    for x_value, point in zip(x_tool, tool_points):
        ax.scatter(
            x_value,
            point["similarity"],
            s=34,
            marker="^",
            color=DATASET_COLORS.get(point["dataset"], "#6C757D"),
            alpha=0.85,
            edgecolors="none",
        )

    ax.axhline(0.0, color="#808080", linewidth=1.0, linestyle=":")
    ax.axhline(reasoning_max, color="#2E6F95", linewidth=1.2, linestyle="--")
    ax.axhline(tool_min, color="#D1495B", linewidth=1.2, linestyle="--")
    ax.axvline(len(reasoning_points) + gap / 2, color="#999999", linewidth=1.0, linestyle="--")

    ax.text(
        (1 + len(reasoning_points)) / 2 if reasoning_points else 1,
        reasoning_max - 0.015,
        f"Selected reasoning points: {len(reasoning_points)}\n(similarity <= {reasoning_max:.2f})",
        ha="center",
        va="top",
        fontsize=9,
        color="#2E6F95",
    )
    ax.text(
        tool_start + max(len(tool_points) - 1, 0) / 2 if tool_points else tool_start,
        tool_min + 0.015,
        f"Selected tool points: {len(tool_points)}\n(similarity >= {tool_min:.2f})",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#D1495B",
    )

    xticks = []
    xticklabels = []
    if reasoning_points:
        xticks.append((1 + len(reasoning_points)) / 2)
        xticklabels.append("Low-sim reasoning")
    if tool_points:
        xticks.append(tool_start + max(len(tool_points) - 1, 0) / 2)
        xticklabels.append("High-sim tool")
    ax.set_xticks(xticks, xticklabels)

    ax.set_ylabel("Cosine similarity to tool-minus-reasoning steering direction")
    ax.set_xlabel("Selected points ranked within each group")
    ax.set_title("Selected points showing high-similarity tool blocks and low-similarity reasoning blocks")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=dataset, markerfacecolor=color, markersize=8)
        for dataset, color in DATASET_COLORS.items()
    ]
    kind_handles = [
        plt.Line2D([0], [0], marker="o", color="#444444", linestyle="None", label="Reasoning", markersize=7),
        plt.Line2D([0], [0], marker="^", color="#444444", linestyle="None", label="Tool", markersize=7),
    ]
    threshold_handles = [
        plt.Line2D([0], [0], color="#2E6F95", linestyle="--", label=f"Reasoning threshold ({reasoning_max:.2f})"),
        plt.Line2D([0], [0], color="#D1495B", linestyle="--", label=f"Tool threshold ({tool_min:.2f})"),
    ]
    ax.legend(handles=[*dataset_handles, *kind_handles, *threshold_handles], loc="lower right", fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Tool points are raw blocks containing a tool heading; Final Response-only blocks are excluded.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select high-similarity tool points and low-similarity reasoning points from multiple result JSON files."
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
        "--reasoning-max",
        type=float,
        default=-0.10,
        help="Keep reasoning points with similarity <= this threshold.",
    )
    parser.add_argument(
        "--tool-min",
        type=float,
        default=0.10,
        help="Keep tool points with similarity >= this threshold.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("SteeringMark/evaluate/figures/high_tool_low_reasoning_points"),
        help="Output prefix without extension.",
    )
    args = parser.parse_args()

    all_points: list[dict] = []
    inputs = {}
    for dataset, json_path in args.input:
        inputs[dataset] = json_path
        all_points.extend(collect_points(dataset, Path(json_path)))

    reasoning_points, tool_points = select_points(
        all_points,
        reasoning_max=args.reasoning_max,
        tool_min=args.tool_min,
    )

    csv_path = args.output_prefix.with_suffix(".csv")
    summary_path = args.output_prefix.with_suffix(".summary.json")
    png_path = args.output_prefix.with_suffix(".png")

    write_csv(csv_path, reasoning_points, tool_points)
    summary = build_summary(
        inputs=inputs,
        reasoning_points=reasoning_points,
        tool_points=tool_points,
        reasoning_max=args.reasoning_max,
        tool_min=args.tool_min,
    )
    summary_path.write_text(json.dumps(summary, indent=2))
    plot_selected_points(
        reasoning_points=reasoning_points,
        tool_points=tool_points,
        reasoning_max=args.reasoning_max,
        tool_min=args.tool_min,
        output_path=png_path,
    )

    print(f"Saved CSV to: {csv_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved plot to: {png_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
