import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


MODELS = [
    "Mistral-7B",
    "Llama-3.1-8B",
    "Mistral-Nemo(12B)",
    "Mistral-Small(24B)",
    "Llama-3.1-70B",
]

TASKS = ["Math", "Time", "Intention"]

METHODS = ["Base Model Tool Prompt", "Activation Addition"]

# Hard-coded ToolAvgUse values from the comparison sheet.
# Edit these values directly when the source numbers change.
TOOL_USAGE = {
    "Mistral-7B": {
        "Math": {"Base Model Tool Prompt": 3.90, "Activation Addition": 0.56},
        "Time": {"Base Model Tool Prompt": 1.67, "Activation Addition": 1.12},
        "Intention": {"Base Model Tool Prompt": 3.80, "Activation Addition": 2.34},
    },
    "Llama-3.1-8B": {
        "Math": {"Base Model Tool Prompt": 1.93, "Activation Addition": 0.35},
        "Time": {"Base Model Tool Prompt": 2.05, "Activation Addition": 1.50},
        "Intention": {"Base Model Tool Prompt": 3.77, "Activation Addition": 1.32},
    },
    "Mistral-Nemo(12B)": {
        "Math": {"Base Model Tool Prompt": 2.35, "Activation Addition": 0.52},
        "Time": {"Base Model Tool Prompt": 1.19, "Activation Addition": 1.20},
        "Intention": {"Base Model Tool Prompt": 1.80, "Activation Addition": 1.64},
    },
    "Mistral-Small(24B)": {
        "Math": {"Base Model Tool Prompt": 1.55, "Activation Addition": 0.13},
        "Time": {"Base Model Tool Prompt": 1.73, "Activation Addition": 1.48},
        "Intention": {"Base Model Tool Prompt": 2.52, "Activation Addition": 1.81},
    },
    "Llama-3.1-70B": {
        "Math": {"Base Model Tool Prompt": 3.53, "Activation Addition": 0.82},
        "Time": {"Base Model Tool Prompt": 2.08, "Activation Addition": 1.90},
        "Intention": {"Base Model Tool Prompt": 2.71, "Activation Addition": 2.47},
    },
}

DOMAIN_STYLES = {
    "Math": {
        "base_facecolor": "#D7E8F7",
        "activation_facecolor": "#4C78A8",
        "edgecolor": "#2F5D8A",
    },
    "Time": {
        "base_facecolor": "#F9DABF",
        "activation_facecolor": "#F58518",
        "edgecolor": "#B85F0A",
    },
    "Intention": {
        "base_facecolor": "#D7EED0",
        "activation_facecolor": "#54A24B",
        "edgecolor": "#397235",
    },
}


def build_layout():
    task_centers = []
    task_labels = []
    model_centers = []
    model_bounds = []

    task_spacing = 0.78
    group_gap = 0.28
    cursor = 0.0

    for model in MODELS:
        start = cursor
        centers_for_model = []
        for task in TASKS:
            task_centers.append(cursor)
            task_labels.append(task)
            centers_for_model.append(cursor)
            cursor += task_spacing
        end = centers_for_model[-1]
        model_centers.append(np.mean(centers_for_model))
        model_bounds.append((start - 0.5 * task_spacing, end + 0.5 * task_spacing))
        cursor += group_gap

    return (
        np.array(task_centers),
        task_labels,
        np.array(model_centers),
        model_bounds,
    )


def add_model_group_labels(ax, model_centers):
    for center, model in zip(model_centers, MODELS):
        ax.text(
            center,
            -0.11,
            model,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            fontweight="semibold",
        )


def add_group_separators(ax, model_bounds):
    for idx, (_, right) in enumerate(model_bounds[:-1]):
        next_left, _ = model_bounds[idx + 1]
        ax.axvline((right + next_left) / 2, color="#D7DCE2", linewidth=1.0, zorder=0)


def apply_base_shadow(bars):
    shadow_effect = pe.SimplePatchShadow(
        offset=(1.4, -1.4),
        shadow_rgbFace="#8A97A5",
        alpha=0.22,
        rho=0.96,
    )
    for bar in bars:
        bar.set_path_effects([shadow_effect, pe.Normal()])


def plot_grouped_bar_chart(output_path: Path):
    task_centers, task_labels, model_centers, model_bounds = build_layout()
    bar_width = 0.56

    fig, ax = plt.subplots(figsize=(15.2, 5.7))
    ax.set_facecolor("#FCFCFD")
    fig.patch.set_facecolor("white")

    base_values = [
        TOOL_USAGE[model][task]["Base Model Tool Prompt"]
        for model in MODELS
        for task in TASKS
    ]
    activation_values = [
        TOOL_USAGE[model][task]["Activation Addition"]
        for model in MODELS
        for task in TASKS
    ]

    base_colors = [DOMAIN_STYLES[task]["base_facecolor"] for model in MODELS for task in TASKS]
    activation_colors = [
        DOMAIN_STYLES[task]["activation_facecolor"] for model in MODELS for task in TASKS
    ]
    edge_colors = [DOMAIN_STYLES[task]["edgecolor"] for model in MODELS for task in TASKS]

    ax.bar(
        task_centers + 0.035,
        base_values,
        width=bar_width,
        color="#9CA8B5",
        alpha=0.20,
        linewidth=0,
        zorder=2,
    )

    base_bars = ax.bar(
        task_centers,
        base_values,
        width=bar_width,
        color=base_colors,
        edgecolor=edge_colors,
        linewidth=1.15,
        hatch="///",
        zorder=3,
    )
    apply_base_shadow(base_bars)

    ax.bar(
        task_centers,
        activation_values,
        width=bar_width,
        color=activation_colors,
        edgecolor=edge_colors,
        linewidth=1.1,
        alpha=0.96,
        zorder=4,
    )

    add_group_separators(ax, model_bounds)
    add_model_group_labels(ax, model_centers)

    ax.set_xticks(task_centers)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.tick_params(axis="x", length=0, pad=8)

    ax.set_ylabel("Average Tool Usage Per Query", fontsize=12)
    ax.set_ylim(0, 4.35)
    right_limit = task_centers[-1] + bar_width / 2 + 0.08
    ax.set_xlim(model_bounds[0][0], right_limit)
    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.28)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AEB6C1")
    ax.spines["bottom"].set_color("#AEB6C1")

    style_handles = [
        mpatches.Patch(
            facecolor="#EAF1F8",
            edgecolor="#5C6F82",
            linewidth=1.0,
            hatch="///",
            label="Base Model Tool Prompt",
        ),
        mpatches.Patch(
            facecolor="#5C6F82",
            edgecolor="#455463",
            linewidth=1.0,
            label="Activation Addition",
        ),
    ]
    ax.legend(
        handles=style_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.89, bottom=0.17)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-model grouped ToolAvgUse bars for base prompting vs activation addition."
    )
    parser.add_argument(
        "--output",
        default="/data/yuqi/SteeringMark/evaluate/figures/model_type_bar_chart.png",
        help="Path to save the generated chart.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    plot_grouped_bar_chart(output_path)
    print(f"Saved chart to: {output_path}")


if __name__ == "__main__":
    main()
