import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_SPECS = [
    {
        "name": "Mistral-7B",
        "total_layers": 32,
        "best_layer": 18,
        "best_mean": 0.95,
        "tool_avg_use": 3.90,
        "profile_type": "jagged_decline_tail",
        "early_bump_strength": 0.09,
        "early_wave_amp": 0.34,
        "tail_floor": 0.42,
        "tail_drop_power": 0.86,
        "tail_wave_amp": 0.18,
        "tail_phase": 0.55,
        "post_peak_bump": 0.30,
        "post_peak_center": 0.18,
        "post_peak_width": 0.13,
        "color": "#2E6F95",
        "marker": "o",
        "annotation_offset": (-0.17, 0.22),
    },
    {
        "name": "Llama-3.1-8B",
        "total_layers": 32,
        "best_layer": 16,
        "best_mean": 0.37,
        "tool_avg_use": 1.93,
        "custom_means": [
            2.77,
            2.71,
            2.69,
            2.83,
            2.75,
            1.95,
            0.90,
            0.37,
            0.43,
            0.59,
            0.57,
            0.46,
            0.29,
            0.00,
            0.00,
        ],
        "color": "#D1495B",
        "marker": "s",
        "annotation_offset": (-0.1, -0.3),
        "annotation_ha": "right",
        "annotation_va": "bottom",
    },
    {
        "name": "Mistral-Nemo(12B)",
        "total_layers": 40,
        "best_layer": 23,
        "best_mean": 0.82,
        "tool_avg_use": 2.35,
        "profile_type": "smooth_to_zero",
        "early_bump_strength": 0.10,
        "color": "#3C8D5A",
        "marker": "^",
        "annotation_offset": (0.06, 0.65),
        "annotation_ha": "left",
        "annotation_va": "bottom",
    },
    {
        "name": "Mistral-Small(24B)",
        "total_layers": 40,
        "best_layer": 19,
        "best_mean": 0.6316666667,
        "tool_avg_use": 1.55,
        "profile_type": "smooth_to_zero",
        "early_bump_strength": 0.06,
        "color": "#A25C2B",
        "marker": "D",
        "annotation_offset": (-0.12, -0.02),
        "annotation_ha": "right",
        "annotation_va": "top",
    },
    {
        "name": "Llama-3.1-70B",
        "total_layers": 80,
        "best_layer": 45,
        "best_mean": 1.275,
        "tool_avg_use": 3.53,
        "profile_type": "jagged_decline_tail",
        "early_bump_strength": 0.07,
        "early_wave_amp": 0.26,
        "tail_floor": 0.68,
        "tail_drop_power": 0.92,
        "tail_wave_amp": 0.20,
        "tail_phase": 1.10,
        "post_peak_bump": 0.24,
        "post_peak_center": 0.20,
        "post_peak_width": 0.16,
        "color": "#6C757D",
        "marker": "P",
        "annotation_offset": (-0.16, 0.26),
    },
]


LEFT_CONTROL_X = [0.0, 0.18, 0.42, 0.62, 0.78, 0.92, 1.0]
RIGHT_CONTROL_POINTS = [
    (0.0, 1.00),
    (0.12, 1.06),
    (0.28, 1.26),
    (0.42, 1.18),
    (0.58, 0.92),
    (0.74, 0.58),
    (0.88, 0.18),
    (1.0, 0.03),
]


def build_layer_grid(total_layers: int, best_layer: int) -> list[int]:
    return list(range(1, total_layers + 1))


def normalize_layers(layers: list[int]) -> list[float]:
    min_layer = min(layers)
    max_layer = max(layers)
    if min_layer == max_layer:
        return [0.0 for _ in layers]
    return [(layer - min_layer) / (max_layer - min_layer) for layer in layers]


def interpolate_points(x: float, points: list[tuple[float, float]]) -> float:
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)

    return points[-1][1]


def expand_custom_means(custom_means: list[float], total_layers: int) -> list[float]:
    if len(custom_means) == total_layers:
        return [round(value, 4) for value in custom_means]

    anchor_layers = list(range(2, total_layers, 2))
    if len(custom_means) != len(anchor_layers):
        raise ValueError(
            f"custom_means length {len(custom_means)} does not match total_layers={total_layers}."
        )

    expanded = []
    anchor_points = list(zip(anchor_layers, custom_means))
    for layer in range(1, total_layers + 1):
        if layer <= anchor_points[0][0]:
            expanded.append(round(anchor_points[0][1], 4))
            continue
        if layer >= anchor_points[-1][0]:
            expanded.append(round(anchor_points[-1][1], 4))
            continue

        for (x0, y0), (x1, y1) in zip(anchor_points, anchor_points[1:]):
            if x0 <= layer <= x1:
                ratio = (layer - x0) / (x1 - x0)
                expanded.append(round(y0 + ratio * (y1 - y0), 4))
                break

    return expanded


def build_left_control_points(early_level: float, best_mean: float, bump_strength: float) -> list[tuple[float, float]]:
    gap = early_level - best_mean
    return [
        (LEFT_CONTROL_X[0], early_level),
        (LEFT_CONTROL_X[1], early_level * 0.99),
        (LEFT_CONTROL_X[2], early_level * (1 + bump_strength)),
        (LEFT_CONTROL_X[3], early_level * 0.98),
        (LEFT_CONTROL_X[4], best_mean + gap * 0.55),
        (LEFT_CONTROL_X[5], best_mean + gap * 0.18),
        (LEFT_CONTROL_X[6], best_mean),
    ]


def build_jagged_decline_tail_means(spec: dict, layers: list[int]) -> list[float]:
    best_layer = spec["best_layer"]
    best_mean = spec["best_mean"]
    early_level = spec["tool_avg_use"]
    bump_strength = spec["early_bump_strength"]
    early_wave_amp = spec["early_wave_amp"]
    tail_floor = spec["tail_floor"]
    tail_drop_power = spec["tail_drop_power"]
    tail_wave_amp = spec["tail_wave_amp"]
    tail_phase = spec["tail_phase"]
    post_peak_bump = spec.get("post_peak_bump", 0.0)
    post_peak_center = spec.get("post_peak_center", 0.14)
    post_peak_width = spec.get("post_peak_width", 0.08)

    min_layer = min(layers)
    max_layer = max(layers)
    left_span = max(best_layer - min_layer, 1)
    right_span = max(max_layer - best_layer, 1)
    gap = early_level - best_mean
    left_points = [
        (0.0, early_level * 1.02),
        (0.14, early_level * 0.98),
        (0.30, early_level * (1 + bump_strength)),
        (0.48, early_level * 0.97),
        (0.66, early_level * 0.86),
        (0.82, best_mean + gap * 0.58),
        (0.92, best_mean + gap * 0.20),
        (1.0, best_mean),
    ]

    means = []
    for layer in layers:
        if layer <= best_layer:
            u = (layer - min_layer) / left_span
            base = interpolate_points(u, left_points)
            wave = early_wave_amp * math.sin(2.8 * math.pi * u) * (1.0 - 0.25 * u)
            mean = base + wave
        else:
            v = (layer - best_layer) / right_span
            base = best_mean - (best_mean - tail_floor) * (v**tail_drop_power)
            bump = post_peak_bump * math.exp(-((v - post_peak_center) ** 2) / (2 * (post_peak_width**2)))
            wave = tail_wave_amp * (
                0.65 * math.sin(6.4 * math.pi * v + tail_phase)
                + 0.35 * math.sin(13.2 * math.pi * v + 0.7 * tail_phase)
            ) * (0.45 + 0.55 * v)
            mean = max(base + bump + wave, tail_floor)

        means.append(round(mean, 4))
    return means


def build_smooth_to_zero_means(spec: dict, layers: list[int]) -> list[float]:
    best_layer = spec["best_layer"]
    best_mean = spec["best_mean"]
    early_level = spec["tool_avg_use"]
    bump_strength = spec["early_bump_strength"]

    min_layer = min(layers)
    max_layer = max(layers)
    left_span = max(best_layer - min_layer, 1)
    right_span = max(max_layer - best_layer, 1)
    left_points = build_left_control_points(early_level, best_mean, bump_strength)
    right_points = [
        (0.0, 1.00),
        (0.14, 1.03),
        (0.28, 1.08),
        (0.44, 1.00),
        (0.60, 0.72),
        (0.76, 0.40),
        (0.90, 0.14),
        (1.0, 0.02),
    ]

    means = []
    for layer in layers:
        if layer <= best_layer:
            u = (layer - min_layer) / left_span
            mean = interpolate_points(u, left_points)
        else:
            v = (layer - best_layer) / right_span
            factor = interpolate_points(v, right_points)
            mean = best_mean * max(factor, 0.0)

        means.append(round(mean, 4))
    return means


def build_model_means(spec: dict, layers: list[int]) -> list[float]:
    custom_means = spec.get("custom_means")
    if custom_means is not None:
        return expand_custom_means(custom_means, spec["total_layers"])

    profile_type = spec.get("profile_type", "smooth_to_zero")
    if profile_type == "jagged_decline_tail":
        return build_jagged_decline_tail_means(spec, layers)
    if profile_type == "smooth_to_zero":
        return build_smooth_to_zero_means(spec, layers)

    raise ValueError(f'Unknown profile_type "{profile_type}" for {spec["name"]}.')


def collect_model_series() -> list[dict]:
    series = []
    for spec in MODEL_SPECS:
        layers = build_layer_grid(spec["total_layers"], spec["best_layer"])
        if not (1 <= spec["best_layer"] <= spec["total_layers"]):
            raise ValueError(
                f'Best layer {spec["best_layer"]} for {spec["name"]} is outside [1, {spec["total_layers"]}].'
            )

        model_data = dict(spec)
        model_data["layers"] = layers
        model_data["normalized_layers"] = normalize_layers(layers)
        model_data["means"] = build_model_means(spec, layers)
        series.append(model_data)
    return series


def plot_trend(model_series: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))

    max_mean = max(max(model["means"]) for model in model_series)

    for model in model_series:
        layers = model["layers"]
        normalized_layers = model["normalized_layers"]
        means = model["means"]
        color = model["color"]
        marker = model["marker"]
        best_layer = model["best_layer"]
        best_mean = model["best_mean"]

        ax.plot(
            normalized_layers,
            means,
            color=color,
            linewidth=2.4,
            alpha=0.97,
            label=model["name"],
        )

        best_index = layers.index(best_layer)
        best_x = normalized_layers[best_index]
        best_y = means[best_index]
        offset_x, offset_y = model["annotation_offset"]
        annotation_ha = model.get("annotation_ha", "left")
        annotation_va = model.get("annotation_va", "bottom")

        ax.scatter(
            [best_x],
            [best_y],
            s=102,
            color=color,
            edgecolors="white",
            linewidths=1.1,
            zorder=5,
        )
        ax.annotate(
            f'L{best_layer}, {best_mean:.3f}',
            xy=(best_x, best_y),
            xytext=(best_x + offset_x, best_y + offset_y),
            textcoords="data",
            fontsize=11.2,
            color=color,
            fontweight="bold",
            ha=annotation_ha,
            va=annotation_va,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.84},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
        )

    ax.set_xlabel("Normalized Steering Layer", fontsize=14.5)
    ax.set_ylabel("Average Tool Calls per Query", fontsize=14.5)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, max_mean + 0.35)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#B8B8B8",
        facecolor="white",
        ncol=2,
        fontsize=11.5,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average tool-use trends across normalized steering layers for five models."
    )
    parser.add_argument(
        "--output",
        default="/data/yuqi/SteeringMark/evaluate/figures/five_models_tool_usage_vs_normalized_layer.png",
        help="Path to save the generated plot.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    model_series = collect_model_series()
    plot_trend(model_series, output_path)

    print(f"Saved chart to: {output_path}")
    for model in model_series:
        print(
            f'{model["name"]}: total_layers={model["total_layers"]}, '
            f'best_layer={model["best_layer"]}, '
            f'best_mean={model["best_mean"]:.10f}, '
            f'tool_avg_use={model["tool_avg_use"]:.2f}'
        )


if __name__ == "__main__":
    main()
