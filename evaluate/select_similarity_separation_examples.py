import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOOL_HEADINGS = ("### Code", "### Search", "### AskUser")
TOOL_LABELS = tuple(tag.replace("### ", "") for tag in TOOL_HEADINGS)


@dataclass
class ExampleSummary:
    dataset: str
    input_path: str
    example_index: int
    task: str
    reasoning_values: list[float]
    tool_values: list[float]
    tool_tags: list[str]
    reasoning_mean: float
    tool_mean: float
    mean_diff_tool_minus_reasoning: float
    separation_direction: str
    separation_margin: float


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def extract_mean_cosine(diag: dict) -> float | None:
    summary = diag.get("token_diagnostics_summary") or {}
    first_trigger = summary.get("first_trigger_info") or {}
    value = first_trigger.get("mean_selected_layer_cosine")
    return float(value) if value is not None else None


def classify_block(raw_text: str) -> tuple[str, list[str]]:
    tags = [label for heading, label in zip(TOOL_HEADINGS, TOOL_LABELS) if heading in raw_text]
    return ("tool" if tags else "reasoning"), tags


def summarize_examples(dataset: str, input_path: Path) -> list[ExampleSummary]:
    records = load_json(input_path)
    summaries: list[ExampleSummary] = []

    for example_index, record in enumerate(records):
        reasoning_values: list[float] = []
        tool_values: list[float] = []
        tool_tags: set[str] = set()

        for raw_text, diag in zip(record.get("raw", []), record.get("steering_diagnostics", [])):
            similarity = extract_mean_cosine(diag)
            if similarity is None:
                continue
            if raw_text.lstrip().startswith("### Final Response"):
                continue

            block_type, tags = classify_block(raw_text)
            if block_type == "tool":
                tool_values.append(similarity)
                tool_tags.update(tags)
            else:
                reasoning_values.append(similarity)

        if not reasoning_values or not tool_values:
            continue

        reasoning_max = max(reasoning_values)
        reasoning_min = min(reasoning_values)
        tool_max = max(tool_values)
        tool_min = min(tool_values)

        if tool_min > reasoning_max:
            direction = "tool_above_reasoning"
            margin = tool_min - reasoning_max
        elif tool_max < reasoning_min:
            direction = "tool_below_reasoning"
            margin = reasoning_min - tool_max
        else:
            direction = "overlap"
            margin = -min(reasoning_max - tool_min, tool_max - reasoning_min)

        summaries.append(
            ExampleSummary(
                dataset=dataset,
                input_path=str(input_path),
                example_index=example_index,
                task=record.get("task", ""),
                reasoning_values=reasoning_values,
                tool_values=tool_values,
                tool_tags=sorted(tool_tags),
                reasoning_mean=mean(reasoning_values),
                tool_mean=mean(tool_values),
                mean_diff_tool_minus_reasoning=mean(tool_values) - mean(reasoning_values),
                separation_direction=direction,
                separation_margin=margin,
            )
        )

    return summaries


def rank_examples(examples: list[ExampleSummary]) -> list[ExampleSummary]:
    def sort_key(item: ExampleSummary) -> tuple:
        non_overlap = item.separation_direction != "overlap"
        return (
            1 if non_overlap else 0,
            item.separation_margin,
            abs(item.mean_diff_tool_minus_reasoning),
            len(item.tool_values) + len(item.reasoning_values),
        )

    return sorted(examples, key=sort_key, reverse=True)


def serialize_example(item: ExampleSummary) -> dict:
    return {
        "dataset": item.dataset,
        "input_path": item.input_path,
        "example_index": item.example_index,
        "task": item.task,
        "tool_tags": item.tool_tags,
        "reasoning_values": item.reasoning_values,
        "tool_values": item.tool_values,
        "reasoning_mean": item.reasoning_mean,
        "tool_mean": item.tool_mean,
        "mean_diff_tool_minus_reasoning": item.mean_diff_tool_minus_reasoning,
        "separation_direction": item.separation_direction,
        "separation_margin": item.separation_margin,
    }


def make_wrapped_label(dataset: str, example_index: int, task: str, width: int = 28) -> str:
    prefix = f"{dataset}#{example_index}"
    wrapped = textwrap.fill(task.strip(), width=width)
    first_line = wrapped.splitlines()[0] if wrapped else ""
    if len(first_line) >= width - 1:
        first_line = first_line[: width - 2] + "…"
    return f"{prefix}\n{first_line}"


def plot_selected_examples(selected_by_dataset: dict[str, list[ExampleSummary]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(selected_by_dataset.keys())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(15, 4.8 * len(datasets)), dpi=180, squeeze=False)

    reasoning_color = "#2E6F95"
    tool_color = "#D1495B"
    rng = np.random.default_rng(11)

    for row_idx, dataset in enumerate(datasets):
        ax = axes[row_idx, 0]
        selected = selected_by_dataset[dataset]
        for x_idx, item in enumerate(selected):
            reasoning_x = np.full(len(item.reasoning_values), x_idx - 0.13) + rng.uniform(-0.03, 0.03, len(item.reasoning_values))
            tool_x = np.full(len(item.tool_values), x_idx + 0.13) + rng.uniform(-0.03, 0.03, len(item.tool_values))
            ax.scatter(reasoning_x, item.reasoning_values, s=34, c=reasoning_color, alpha=0.8, edgecolors="none")
            ax.scatter(tool_x, item.tool_values, s=34, c=tool_color, alpha=0.8, edgecolors="none")
            ax.plot([x_idx - 0.16, x_idx - 0.10], [item.reasoning_mean, item.reasoning_mean], color=reasoning_color, linewidth=2.1)
            ax.plot([x_idx + 0.10, x_idx + 0.16], [item.tool_mean, item.tool_mean], color=tool_color, linewidth=2.1)
            ax.text(
                x_idx,
                max(max(item.reasoning_values), max(item.tool_values)) + 0.02,
                f"{item.separation_direction}\nmargin={item.separation_margin:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax.set_title(f"{dataset}: selected examples with reasoning/tool cosine separation")
        ax.set_ylabel("Mean cosine similarity")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xticks(
            range(len(selected)),
            [make_wrapped_label(dataset, item.example_index, item.task) for item in selected],
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w", label="Reasoning-only blocks", markerfacecolor=reasoning_color, markersize=8),
                plt.Line2D([0], [0], marker="o", color="w", label="Tool-related blocks", markerfacecolor=tool_color, markersize=8),
            ],
            loc="upper right",
        )

    axes[-1, 0].set_xlabel("Selected examples")
    fig.text(
        0.5,
        0.01,
        "Tool-related blocks are raw generation blocks that contain ### Code / ### Search / ### AskUser. Final Response-only blocks are excluded.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select question examples that show cosine-similarity separation between reasoning and tool-related blocks."
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("DATASET", "JSON_PATH"),
        required=True,
        help="Dataset label and JSON path. Repeat this flag for multiple inputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of examples to keep per dataset.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("SteeringMark/evaluate/figures/cosine_separation_examples"),
        help="Output prefix without extension.",
    )
    args = parser.parse_args()

    all_selected: dict[str, list[ExampleSummary]] = {}
    serialized: dict[str, list[dict]] = {}

    for dataset, json_path in args.input:
        summaries = summarize_examples(dataset, Path(json_path))
        ranked = rank_examples(summaries)
        selected = ranked[: args.top_k]
        all_selected[dataset] = selected
        serialized[dataset] = [serialize_example(item) for item in selected]

    output_json = args.output_prefix.with_suffix(".json")
    output_png = args.output_prefix.with_suffix(".png")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(serialized, indent=2))
    plot_selected_examples(all_selected, output_png)

    print(f"Saved selected examples to: {output_json}")
    print(f"Saved comparison plot to: {output_png}")
    print(json.dumps(serialized, indent=2))


if __name__ == "__main__":
    main()
