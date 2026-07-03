#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(
    "/data/yuqi/SteeringMark/evaluate/figures/similarity_figure/"
    "three_questions_llama8b_layer16_code_prompt_token_similarity.json"
)
DEFAULT_OUTPUT_PREFIX = Path(
    "/data/yuqi/SteeringMark/evaluate/figures/similarity_figure/"
    "three_questions_llama8b_layer16_code_question_token_similarity_trends"
)

LINE_COLOR = "#1F77B4"
PEAK_COLOR = "#D1495B"
VALLEY_COLOR = "#2A9D8F"
GRID_COLOR = "#D9D9D9"
ZERO_LINE_COLOR = "#6C757D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token-position cosine similarity trends for the three questions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Token-level similarity JSON produced by extract_prompt_token_steering_similarity.py",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path prefix without extension.",
    )
    parser.add_argument(
        "--top-k-annotate",
        type=int,
        default=4,
        help="How many highest-similarity question tokens to annotate per subplot.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_token_label(text: str) -> str:
    normalized = text.replace("\n", "\\n")
    normalized = normalized.strip()
    if not normalized:
        return "<space>"
    return normalized


def clean_question_title(text: str, width: int = 88) -> str:
    normalized = text.replace("\n", " ")
    replacements = {
        "\\le": "<=",
        "\\ge": ">=",
        "\\cdot": "*",
        "\\sum": "sum",
        "\\dotsm": "...",
        "\\textsuperscript": "^",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\[", "").replace("\\]", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return textwrap.fill(normalized, width=width)


def build_question_token_series(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in result["token_records"] if record.get("in_question_span")]


def annotate_top_points(ax, token_records: list[dict[str, Any]], top_k: int) -> None:
    ranked = sorted(token_records, key=lambda item: item["cosine_similarity"], reverse=True)[:top_k]
    for rank, item in enumerate(ranked):
        x = item["question_token_index"]
        y = item["cosine_similarity"]
        label = clean_token_label(item["token_text"])
        x_offset = 6 if rank % 2 == 0 else -6
        ha = "left" if x_offset > 0 else "right"
        ax.scatter([x], [y], color=PEAK_COLOR, s=28, zorder=4)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x_offset, 8),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=9.5,
            color=PEAK_COLOR,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": PEAK_COLOR,
                "alpha": 0.92,
            },
        )


def add_extrema_guides(ax, token_records: list[dict[str, Any]]) -> None:
    if not token_records:
        return
    max_item = max(token_records, key=lambda item: item["cosine_similarity"])
    min_item = min(token_records, key=lambda item: item["cosine_similarity"])
    ax.scatter([max_item["question_token_index"]], [max_item["cosine_similarity"]], color=PEAK_COLOR, s=30, zorder=4)
    ax.scatter([min_item["question_token_index"]], [min_item["cosine_similarity"]], color=VALLEY_COLOR, s=30, zorder=4)


def extract_code_trigger_similarity(result: dict[str, Any]) -> float | None:
    code_generation = result.get("code_heading_generation") or {}
    trigger = code_generation.get("code_heading_trigger")
    if not trigger:
        return None
    return trigger.get("cosine_similarity")


def plot_results(payload: dict[str, Any], output_prefix: Path, top_k_annotate: int) -> None:
    results = payload["results"]
    fig, axes = plt.subplots(
        nrows=len(results),
        ncols=1,
        figsize=(13.5, 9.6),
        dpi=220,
        sharex=False,
    )
    if len(results) == 1:
        axes = [axes]

    global_min = min(
        min(record["cosine_similarity"] for record in build_question_token_series(result))
        for result in results
    )
    global_max = max(
        max(record["cosine_similarity"] for record in build_question_token_series(result))
        for result in results
    )
    trigger_values = [
        value
        for result in results
        for value in [extract_code_trigger_similarity(result)]
        if value is not None
    ]
    if trigger_values:
        global_min = min(global_min, min(trigger_values))
        global_max = max(global_max, max(trigger_values))
    y_padding = max((global_max - global_min) * 0.14, 0.01)
    y_limits = (global_min - y_padding, global_max + y_padding)

    for ax, result in zip(axes, results):
        token_records = build_question_token_series(result)
        for idx, record in enumerate(token_records, start=1):
            record["question_token_index"] = idx

        x = np.array([record["question_token_index"] for record in token_records], dtype=float)
        y = np.array([record["cosine_similarity"] for record in token_records], dtype=float)
        code_trigger_similarity = extract_code_trigger_similarity(result)

        ax.plot(
            x,
            y,
            color=LINE_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
        )
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.3, linestyle="--", alpha=0.9)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_COLOR, alpha=0.85)
        ax.set_ylim(*y_limits)
        x_max = len(token_records) + (1.8 if code_trigger_similarity is not None else 0.0)
        ax.set_xlim(1, x_max)
        ax.set_ylabel("Cosine", fontsize=11.5)

        summary = result["question_cosine_summary"]
        clean_question = clean_question_title(result["question"])
        title = (
            f"Q{result['question_index']} | {clean_question}\n"
            f"question-token mean={summary['mean']:.4f}, max={summary['max']:.4f}, "
            f"layer={payload['saved_steering_layer']}"
        )
        if code_trigger_similarity is not None:
            title += f", ###->Code={code_trigger_similarity:.4f}"
        ax.set_title(title, fontsize=11.8, loc="left", pad=10)

        add_extrema_guides(ax, token_records)
        annotate_top_points(ax, token_records, top_k_annotate)

        if code_trigger_similarity is not None:
            trigger_x = len(token_records) + 1
            ax.scatter([trigger_x], [code_trigger_similarity], color="#8E44AD", s=48, marker="D", zorder=5)
            ax.annotate(
                "### -> Code",
                xy=(trigger_x, code_trigger_similarity),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#8E44AD",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#8E44AD",
                    "alpha": 0.92,
                },
            )

        xtick_count = min(12, len(token_records))
        xticks = np.linspace(1, len(token_records), xtick_count, dtype=int)
        xticks = sorted(set(int(value) for value in xticks))
        if code_trigger_similarity is not None:
            xticks.append(len(token_records) + 1)
        ax.set_xticks(xticks)
        if code_trigger_similarity is not None:
            labels = [str(value) for value in xticks[:-1]] + ["###"]
            ax.set_xticklabels(labels)
        ax.set_xlabel("Question Token Position", fontsize=11.5)
        ax.tick_params(axis="both", labelsize=10.2)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    fig.suptitle(
        "Llama-3.1-8B Prompt Token Similarity Trends\n"
        "Question tokens vs. layer-16 code steering vector",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Saved PNG to {png_path}")
    print(f"Saved PDF to {pdf_path}")


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input)
    plot_results(
        payload=payload,
        output_prefix=args.output_prefix,
        top_k_annotate=args.top_k_annotate,
    )


if __name__ == "__main__":
    main()
