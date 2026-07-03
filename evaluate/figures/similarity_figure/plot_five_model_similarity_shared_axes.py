#!/usr/bin/env python3

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


CATEGORY_REASONING = "Reasoning-only block"
CATEGORY_TOOL = "Tool-related block"

COLORS = {
    CATEGORY_REASONING: "#2E6F95",
    CATEGORY_TOOL: "#D1495B",
}

BASE_DIR = Path("/data/yuqi/SteeringMark/evaluate/figures/similarity_figure")
OUTPUT_PREFIX = BASE_DIR / "five_model_similarity_distributions_horizontal"

MODEL_SPECS = [
    ("Mistral-7B", BASE_DIR / "mistral7b_full_pool_distribution.points.csv"),
    ("Llama-3.1-8B", BASE_DIR / "threshold_separation_demo_t010_categorical_distribution.points.csv"),
    ("Mistral-Nemo-12B", BASE_DIR / "mistral_nemo12b_full_pool_presteering_distribution_aligned_mirrored.points.csv"),
    ("Mistral-Small-24B", BASE_DIR / "mistral_small24b_full_pool_distribution.points.csv"),
    ("Llama-3.1-70B", BASE_DIR / "llama70b_domain_math_unmodified_distribution.points.csv"),
]

FIGSIZE = (22.5, 6.4)
DPI = 220
GRID_SIZE = 601
PANEL_GAP = 0.18
MODEL_LABEL_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 14.5
LEGEND_FONT_SIZE = 20
SEPARATOR_LINEWIDTH = 1.4
CURVE_LINEWIDTH = 2.8
FILL_ALPHA = 0.15
MODEL_LABEL_FIG_Y = 0.08
GLOBAL_XLABEL_Y = 0.14
PANEL_WIDTH = 1.0
ZERO_LINE_COLOR = "#000000"
ZERO_LINE_WIDTH = 1.8
AXIS_SPINE_LINEWIDTH = 1.8
AXIS_TICK_WIDTH = 1.6
TOOL_DENSITY_SCALE_OVERRIDES = {
    "Llama-3.1-70B": 0.5,
}


def load_grouped_values(points_csv_path: Path) -> dict[str, list[float]]:
    grouped = {CATEGORY_REASONING: [], CATEGORY_TOOL: []}
    with points_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["category"]].append(float(row["similarity"]))
    return grouped


def compute_kde_bandwidth(values: np.ndarray) -> float:
    if values.size == 1:
        return 0.015

    std = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    iqr_sigma = float((q75 - q25) / 1.34) if q75 > q25 else 0.0
    sigma = min(item for item in (std, iqr_sigma) if item > 0) if (std > 0 or iqr_sigma > 0) else 0.0
    if sigma <= 0:
        sigma = max(abs(float(values.mean())) * 0.05, 0.01)

    return max(float(0.9 * sigma * (values.size ** (-1.0 / 5.0))), 1e-3)


def compute_density_curve(values: list[float], grid: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    bandwidth = compute_kde_bandwidth(array)
    z_values = (grid[:, None] - array[None, :]) / bandwidth
    density = np.exp(-0.5 * z_values * z_values).sum(axis=1)
    density /= array.size * bandwidth * math.sqrt(2.0 * math.pi)
    return density


def format_tick(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def compute_panel_bounds(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    lower = float(array.min())
    upper = float(array.max())
    span = upper - lower
    padding = max(span * 0.08, 0.02)
    lower -= padding
    upper += padding
    if lower > 0:
        lower = 0.0
    if upper < 0:
        upper = 0.0
    if abs(upper - lower) < 1e-6:
        lower -= 0.05
        upper += 0.05
    return lower, upper


def map_local_value_to_panel_x(
    value: float,
    panel_offset: float,
    local_min: float,
    local_max: float,
    *,
    mirrored: bool = False,
) -> float:
    normalized = (value - local_min) / (local_max - local_min)
    if mirrored:
        normalized = 1.0 - normalized
    return panel_offset + normalized * PANEL_WIDTH


def main() -> None:
    grouped_by_model = []
    max_density = 0.0
    for model_name, points_csv_path in MODEL_SPECS:
        grouped = load_grouped_values(points_csv_path)
        panel_values = grouped[CATEGORY_REASONING] + grouped[CATEGORY_TOOL]
        local_min, local_max = compute_panel_bounds(panel_values)
        local_grid = np.linspace(local_min, local_max, GRID_SIZE)
        densities = {
            category: compute_density_curve(grouped[category], local_grid)
            for category in (CATEGORY_REASONING, CATEGORY_TOOL)
        }
        tool_density_scale = TOOL_DENSITY_SCALE_OVERRIDES.get(model_name, 1.0)
        if tool_density_scale != 1.0:
            # Presentation-only override for specific panels.
            densities[CATEGORY_TOOL] = densities[CATEGORY_TOOL] * tool_density_scale
        max_density = max(max_density, max(float(density.max()) for density in densities.values()))
        grouped_by_model.append((model_name, grouped, densities, local_min, local_max))

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    axis_position = None
    xticks: list[float] = []
    xticklabels: list[str] = []
    label_positions: list[tuple[float, str]] = []

    for model_index, (model_name, _grouped, densities, local_min, local_max) in enumerate(grouped_by_model):
        panel_offset = model_index * (PANEL_WIDTH + PANEL_GAP)
        local_grid = np.linspace(local_min, local_max, GRID_SIZE)
        x_plot = np.array(
            [
                map_local_value_to_panel_x(
                    value,
                    panel_offset,
                    local_min,
                    local_max,
                    mirrored=True,
                )
                for value in local_grid
            ]
        )

        for category in (CATEGORY_REASONING, CATEGORY_TOOL):
            density = densities[category]
            ax.plot(x_plot, density, color=COLORS[category], linewidth=CURVE_LINEWIDTH)
            ax.fill_between(x_plot, density, color=COLORS[category], alpha=FILL_ALPHA)

        zero_x = map_local_value_to_panel_x(
            0.0,
            panel_offset,
            local_min,
            local_max,
            mirrored=True,
        )
        ax.axvline(
            zero_x,
            color=ZERO_LINE_COLOR,
            linestyle=":",
            linewidth=ZERO_LINE_WIDTH,
            alpha=0.9,
            zorder=0,
        )

        tick_values = [local_min, 0.0, local_max]
        for tick_value in tick_values:
            tick_x = map_local_value_to_panel_x(
                tick_value,
                panel_offset,
                local_min,
                local_max,
                mirrored=True,
            )
            xticks.append(tick_x)
            xticklabels.append(format_tick(-tick_value))

        label_positions.append((panel_offset + PANEL_WIDTH / 2.0, model_name))

        if model_index < len(grouped_by_model) - 1:
            separator_x = panel_offset + PANEL_WIDTH + PANEL_GAP / 2.0
            ax.axvline(
                separator_x,
                color="0.5",
                linestyle="--",
                linewidth=SEPARATOR_LINEWIDTH,
                alpha=0.9,
            )

    total_width = len(grouped_by_model) * PANEL_WIDTH + (len(grouped_by_model) - 1) * PANEL_GAP
    ax.set_xlim(-0.02, total_width + 0.02)
    ax.set_ylim(0.0, max_density * 1.08)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE, width=AXIS_TICK_WIDTH)
    ax.set_ylabel("Distribution", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.95, color="#CFCFCF", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_SPINE_LINEWIDTH)
        spine.set_color("black")
    legend_handles = [
        Line2D([0], [0], color=COLORS[CATEGORY_REASONING], linewidth=CURVE_LINEWIDTH, label="Reasoning Step"),
        Line2D([0], [0], color=COLORS[CATEGORY_TOOL], linewidth=CURVE_LINEWIDTH, label="Tool Step"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D0D0D0",
    )

    fig.subplots_adjust(left=0.045, right=0.995, top=0.98, bottom=0.22)
    fig.text(0.525, GLOBAL_XLABEL_Y, "Cosine Similarity", ha="center", va="center", fontsize=AXIS_LABEL_FONT_SIZE)
    axis_position = ax.get_position()
    for label_x, model_name in label_positions:
        x_norm = axis_position.x0 + axis_position.width * ((label_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]))
        fig.text(x_norm, MODEL_LABEL_FIG_Y, model_name, ha="center", va="center", fontsize=MODEL_LABEL_FONT_SIZE)

    png_path = OUTPUT_PREFIX.with_suffix(".png")
    pdf_path = OUTPUT_PREFIX.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved shared-axis figure to: {png_path}")
    print(f"Saved shared-axis figure to: {pdf_path}")


if __name__ == "__main__":
    main()
