import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

METHODS = [
    "Orthogonalization",
    "Base Model Tool Prompt",
    "All Steering",
]

COLORS = {
    "Orthogonalization": "#4C78A8",
    "Base Model Tool Prompt": "#F58518",
    "All Steering": "#54A24B",
}

LLAMA_8B_PATHS = {
    "Orthogonalization": Path(
        "/data/yuqi/SteeringMark/inference_results_qrsubspace/llama8b_/domain_math_heading_code_qrsubspace_layer16_scale1p0_all.json"
    ),
    "Base Model Tool Prompt": Path(
        "/data/yuqi/Open-SMARTAgent/inference_results/domain_math_tool_prompt_llama31_8b.json"
    ),
    "All Steering": Path(
        "/data/yuqi/SteeringMark/inference_results/llama8b_/domain_math_heading_layer21_scale1p0_all_judge.json"
    ),
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_tool_counts_from_inference(path: Path) -> list[int]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, but got {type(data).__name__}.")

    counts = []
    for item in data:
        if not isinstance(item, dict):
            counts.append(0)
            continue
        predict_steps = item.get("predict", [])
        tool_count = sum(
            1
            for step in predict_steps
            if isinstance(step, dict) and step.get("type") == "tool"
        )
        counts.append(tool_count)
    return counts


def load_distributions(other_models_path: Path) -> dict[str, dict[str, list[int]]]:
    generated = load_json(other_models_path)
    if not isinstance(generated, dict):
        raise ValueError(f"Expected a dict in {other_models_path}, but got {type(generated).__name__}.")

    distributions: dict[str, dict[str, list[int]]] = {}
    for model in MODELS:
        distributions[model] = {}
        for method in METHODS:
            if model == "Llama-3.1-8B":
                distributions[model][method] = extract_tool_counts_from_inference(LLAMA_8B_PATHS[method])
                continue

            if model not in generated:
                raise KeyError(f"Missing model {model!r} in {other_models_path}.")
            if method not in generated[model]:
                raise KeyError(f"Missing setting {method!r} for model {model!r} in {other_models_path}.")
            values = generated[model][method]
            if not isinstance(values, list) or not values:
                raise ValueError(f"Invalid values for {model!r} / {method!r} in {other_models_path}.")
            distributions[model][method] = [int(v) for v in values]
    return distributions


def add_method_violins(ax: plt.Axes, datasets: list[list[int]], positions: np.ndarray, method: str) -> None:
    violins = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.22,
        vert=True,
        showmeans=False,
        showmedians=False,
        showextrema=True,
        bw_method=0.25,
        points=200,
        quantiles=[[0.25, 0.5, 0.75] for _ in datasets],
    )

    for body in violins["bodies"]:
        body.set_facecolor(COLORS[method])
        body.set_edgecolor("#3F3F3F")
        body.set_linewidth(1.0)
        body.set_alpha(0.72)

    violins["cbars"].set_color("#3F3F3F")
    violins["cbars"].set_linewidth(1.0)
    violins["cmins"].set_color("#3F3F3F")
    violins["cmins"].set_linewidth(1.0)
    violins["cmaxes"].set_color("#3F3F3F")
    violins["cmaxes"].set_linewidth(1.0)
    violins["cquantiles"].set_color("#2F4F6F")
    violins["cquantiles"].set_linewidth(1.2)


def plot_grouped_violin_chart(distributions: dict[str, dict[str, list[int]]], output_path: Path) -> None:
    x = np.arange(len(MODELS)) * 1.05
    offsets = np.array([-0.24, 0.0, 0.24])

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_facecolor("#FBFBFB")
    fig.patch.set_facecolor("white")

    for idx, method in enumerate(METHODS):
        datasets = [distributions[model][method] for model in MODELS]
        add_method_violins(ax, datasets, x + offsets[idx], method)

    ax.set_xlabel("Model Type", fontsize=12)
    ax.set_ylabel("Tool Usage Per Query", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylim(-0.2, 10.8)
    ax.grid(axis="y", linestyle="--", alpha=0.32)
    ax.set_axisbelow(True)
    ax.margins(x=0.03)

    legend_handles = [
        mpatches.Patch(facecolor=COLORS[method], edgecolor="#3F3F3F", alpha=0.72, label=method)
        for method in METHODS
    ]
    ax.legend(handles=legend_handles, frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot grouped violin charts for five models under three settings."
    )
    parser.add_argument(
        "--input",
        default="/data/yuqi/SteeringMark/evaluate/figures/generated_tool_usage_other_4_models.json",
        help="Path to the JSON file containing generated tool-usage samples for the other four models.",
    )
    parser.add_argument(
        "--output",
        default="/data/yuqi/SteeringMark/evaluate/figures/model_type_violin_chart_five_models.png",
        help="Path to save the generated chart.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    distributions = load_distributions(input_path)
    plot_grouped_violin_chart(distributions, output_path)

    print(f"Saved chart to: {output_path}")
    for model in MODELS:
        summary = {method: float(np.mean(distributions[model][method])) for method in METHODS}
        print(f"{model}: {summary}")


if __name__ == "__main__":
    main()
