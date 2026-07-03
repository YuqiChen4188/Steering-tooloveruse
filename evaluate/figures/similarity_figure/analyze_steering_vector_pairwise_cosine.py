#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean, median

import torch
import torch.nn.functional as F


DEFAULT_INPUT_DIR = Path("/data/yuqi/SteeringMark/steering_vector")
DEFAULT_OUTPUT_PREFIX = Path(
    "/data/yuqi/SteeringMark/evaluate/figures/similarity_figure/steering_vector_pairwise_cosine"
)
VECTOR_NAME_TO_FILE_STEM = {
    "code": "code",
    "search": "search",
    "askuser": "askuser",
}
VECTOR_NAMES = tuple(VECTOR_NAME_TO_FILE_STEM.keys())
PAIR_NAMES = tuple(itertools.combinations(VECTOR_NAMES, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise cosine similarities between code/search/askuser steering "
            "vectors for each model."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per-model steering-vector payloads.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix for the generated CSV and summary JSON files.",
    )
    parser.add_argument(
        "--include-qr-subspace",
        action="store_true",
        help="Also include directories whose name ends with `_qr_subspace`.",
    )
    return parser.parse_args()


def discover_model_dirs(root: Path, include_qr_subspace: bool) -> list[Path]:
    model_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "__pycache__":
            continue
        if child.name.endswith("_qr_subspace") and not include_qr_subspace:
            continue
        payload_paths = [child / f"step_mark_{file_stem}.pt" for file_stem in VECTOR_NAME_TO_FILE_STEM.values()]
        if all(path.exists() for path in payload_paths):
            model_dirs.append(child)
    return model_dirs


def load_steering_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "steering_vectors" in payload:
        return payload["steering_vectors"].float().cpu()

    if isinstance(payload, dict) and payload.get("payload_type") == "qr_subspace_projection":
        source_vectors = payload["source_vectors"].float().cpu()
        if source_vectors.ndim != 3 or source_vectors.shape[0] != 1:
            raise ValueError(f"Unsupported qr_subspace tensor shape at {path}: {tuple(source_vectors.shape)}")
        return source_vectors[0]

    raise ValueError(f"Unsupported payload format at {path}.")


def summarize_scalar_list(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def build_pair_key(left_name: str, right_name: str) -> str:
    return f"{left_name}_{right_name}"


def main() -> None:
    args = parse_args()
    model_dirs = discover_model_dirs(args.input_dir, args.include_qr_subspace)
    if not model_dirs:
        raise ValueError(f"No eligible model directories found under {args.input_dir}.")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    summary_path = args.output_prefix.with_suffix(".summary.json")

    csv_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "included_models": [],
        "vector_names": list(VECTOR_NAMES),
        "aggregation_default": "mean_layerwise_cosine",
        "models": {},
    }

    for model_dir in model_dirs:
        model_name = model_dir.name
        vectors: dict[str, torch.Tensor] = {}
        hidden_dim: int | None = None
        layer_count: int | None = None

        for vector_name, file_stem in VECTOR_NAME_TO_FILE_STEM.items():
            tensor = load_steering_tensor(model_dir / f"step_mark_{file_stem}.pt")
            current_layer_count, current_hidden_dim = tensor.shape

            if hidden_dim is None:
                hidden_dim = current_hidden_dim
                layer_count = current_layer_count
            elif hidden_dim != current_hidden_dim or layer_count != current_layer_count:
                raise ValueError(
                    f"Shape mismatch inside {model_name}: expected {(layer_count, hidden_dim)}, "
                    f"got {(current_layer_count, current_hidden_dim)} for {vector_name}."
                )

            vectors[vector_name] = tensor

        assert hidden_dim is not None
        assert layer_count is not None

        pairwise_summary: dict[str, dict[str, object]] = {}
        for left_name, right_name in PAIR_NAMES:
            left = vectors[left_name]
            right = vectors[right_name]
            per_layer = F.cosine_similarity(left, right, dim=1).tolist()
            per_layer_values = [float(value) for value in per_layer]
            flattened_cosine = F.cosine_similarity(
                left.flatten().unsqueeze(0),
                right.flatten().unsqueeze(0),
                dim=1,
            ).item()

            pair_key = build_pair_key(left_name, right_name)
            pairwise_summary[pair_key] = {
                "left_name": left_name,
                "right_name": right_name,
                "mean_layerwise_cosine": mean(per_layer_values),
                "flattened_cosine": float(flattened_cosine),
                "per_layer": per_layer_values,
                "per_layer_stats": summarize_scalar_list(per_layer_values),
            }
            csv_rows.append(
                {
                    "model": model_name,
                    "layer_count": layer_count,
                    "hidden_dim": hidden_dim,
                    "pair_name": pair_key,
                    "left_name": left_name,
                    "right_name": right_name,
                    "mean_layerwise_cosine": pairwise_summary[pair_key]["mean_layerwise_cosine"],
                    "flattened_cosine": pairwise_summary[pair_key]["flattened_cosine"],
                }
            )

        summary["included_models"].append(model_name)
        summary["models"][model_name] = {
            "layer_count": layer_count,
            "hidden_dim": hidden_dim,
            "pairwise_cosine": pairwise_summary,
        }

    fieldnames = [
        "model",
        "layer_count",
        "hidden_dim",
        "pair_name",
        "left_name",
        "right_name",
        "mean_layerwise_cosine",
        "flattened_cosine",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Wrote pairwise cosine CSV to {csv_path}")
    print(f"Wrote pairwise cosine summary to {summary_path}")


if __name__ == "__main__":
    main()
