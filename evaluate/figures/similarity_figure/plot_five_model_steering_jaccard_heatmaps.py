#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path("/data/yuqi/SteeringMark/evaluate/figures/similarity_figure")
DEFAULT_INPUT_SUMMARY = BASE_DIR / "top_feature_jaccard_abs_t100.summary.json"
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "five_model_steering_jaccard_heatmaps"

MODEL_SPECS = [
    ("Mistral_7B_vector_heading", "Mistral-7B"),
    ("Llama_3_8_vector_heading", "Llama-3.1-8B"),
    ("Mistral_Nemo_12B_vector_heading", "Mistral-Nemo(12B)"),
    ("Mistral_Small_24B_vector_heading", "Mistral-Small(24B)"),
    ("Llama_3_70B_vector_heading", "Llama-3.1-70B"),
]

TOOL_LABELS = ["code", "search", "askuser"]
PAIR_INDEX = {
    "math_search": (0, 1),
    "math_askuser": (0, 2),
    "search_askuser": (1, 2),
}

FIGSIZE = (22, 4.8)
DPI = 220
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 13
CELL_FONT_SIZE = 11
COLORBAR_FONT_SIZE = 12
COLORBAR_AXES = [0.945, 0.20, 0.012, 0.58]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3x3 steering-vector Jaccard heatmaps for five models."
    )
    parser.add_argument(
        "--input-summary",
        type=Path,
        default=DEFAULT_INPUT_SUMMARY,
        help="Path to the summary JSON produced by analyze_top_feature_jaccard.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path prefix for the saved figure.",
    )
    parser.add_argument(
        "--cmap",
        default="YlOrRd",
        help="Matplotlib colormap name for the heatmap.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model_matrix(model_summary: dict) -> np.ndarray:
    matrix = np.eye(len(TOOL_LABELS), dtype=float)

    for pair_name, (row_idx, col_idx) in PAIR_INDEX.items():
        mean_value = model_summary["pairwise_jaccard_summary"][pair_name]["mean"]
        matrix[row_idx, col_idx] = mean_value
        matrix[col_idx, row_idx] = mean_value

    return matrix


def pick_text_color(value: float, vmax: float) -> str:
    if value >= max(vmax * 0.55, 0.18):
        return "white"
    return "black"


def main() -> None:
    args = parse_args()
    summary = load_summary(args.input_summary)

    matrices: list[np.ndarray] = []
    display_names: list[str] = []
    off_diagonal_values: list[float] = []

    for model_key, display_name in MODEL_SPECS:
        model_summary = summary["models"][model_key]
        matrix = build_model_matrix(model_summary)
        matrices.append(matrix)
        display_names.append(display_name)
        off_diagonal_values.extend(matrix[np.triu_indices_from(matrix, k=1)].tolist())

    vmax = max(off_diagonal_values)

    fig, axes = plt.subplots(1, len(MODEL_SPECS), figsize=FIGSIZE, dpi=DPI)
    image = None

    for panel_index, (ax, matrix, display_name) in enumerate(zip(axes, matrices, display_names)):
        image = ax.imshow(matrix, cmap=args.cmap, vmin=0.0, vmax=vmax)

        ax.set_xticks(range(len(TOOL_LABELS)))
        ax.set_yticks(range(len(TOOL_LABELS)))
        ax.set_xticklabels(TOOL_LABELS, fontsize=AXIS_LABEL_FONT_SIZE)
        if panel_index == 0:
            ax.set_yticklabels(TOOL_LABELS, fontsize=AXIS_LABEL_FONT_SIZE)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_title(display_name, fontsize=TITLE_FONT_SIZE, pad=12)

        ax.set_xticks(np.arange(-0.5, len(TOOL_LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(TOOL_LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=CELL_FONT_SIZE,
                    color=pick_text_color(min(value, vmax), vmax),
                    fontweight="semibold",
                )

    assert image is not None
    colorbar_ax = fig.add_axes(COLORBAR_AXES)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Pairwise Jaccard", fontsize=COLORBAR_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_FONT_SIZE)

    fig.subplots_adjust(left=0.03, right=0.925, top=0.83, bottom=0.12, wspace=0.32)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print(f"Saved heatmap figure to: {png_path}")
    print(f"Saved heatmap figure to: {pdf_path}")


if __name__ == "__main__":
    main()
