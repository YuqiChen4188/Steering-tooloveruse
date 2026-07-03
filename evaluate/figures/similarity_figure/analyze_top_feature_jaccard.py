#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import torch


DEFAULT_INPUT_DIR = Path("/data/yuqi/SteeringMark/steering_vector")
DEFAULT_OUTPUT_DIR = Path("/data/yuqi/SteeringMark/evaluate/figures/similarity_figure")
VECTOR_NAME_TO_FILE_STEM = {
    "math": "code",
    "search": "search",
    "askuser": "askuser",
}
VECTOR_NAMES = tuple(VECTOR_NAME_TO_FILE_STEM.keys())
PAIR_NAMES = (
    ("math", "search"),
    ("math", "askuser"),
    ("search", "askuser"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the top-k hidden-dimension indices for each steering vector and "
            "compute per-layer Jaccard similarities."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per-model steering-vector payloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.10,
        help="Fraction of hidden dimensions to keep from each layer.",
    )
    parser.add_argument(
        "--ranking",
        choices=("abs", "positive"),
        default="abs",
        help="How to rank dimensions inside each layer before selecting the top subset.",
    )
    parser.add_argument(
        "--include-qr-subspace",
        action="store_true",
        help="Also include qr_subspace payloads when they are present.",
    )
    return parser.parse_args()


def validate_top_ratio(value: float) -> float:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"--top-ratio must be in (0, 1], got {value}.")
    return value


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
    if "steering_vectors" in payload:
        return payload["steering_vectors"].float().cpu()

    if payload.get("payload_type") == "qr_subspace" and "source_vectors" in payload:
        source_vectors = payload["source_vectors"].float().cpu()
        if source_vectors.ndim != 3 or source_vectors.shape[0] != 1:
            raise ValueError(f"Unsupported qr_subspace tensor shape at {path}: {tuple(source_vectors.shape)}")
        return source_vectors[0]

    raise ValueError(f"Unsupported payload format at {path}.")


def select_top_indices_per_layer(
    steering_vectors: torch.Tensor,
    top_ratio: float,
    ranking: str,
) -> tuple[list[list[int]], int]:
    if steering_vectors.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(steering_vectors.shape)}")

    hidden_dim = steering_vectors.shape[1]
    top_k = max(1, math.ceil(hidden_dim * top_ratio))
    selected_indices: list[list[int]] = []

    for layer_tensor in steering_vectors:
        scores = layer_tensor.abs() if ranking == "abs" else layer_tensor
        topk_indices = torch.topk(scores, k=top_k, largest=True).indices.tolist()
        selected_indices.append(sorted(int(index) for index in topk_indices))

    return selected_indices, top_k


def jaccard_similarity(indices_a: list[int], indices_b: list[int]) -> float:
    set_a = set(indices_a)
    set_b = set(indices_b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def jaccard_similarity_three(indices_a: list[int], indices_b: list[int], indices_c: list[int]) -> float:
    set_a = set(indices_a)
    set_b = set(indices_b)
    set_c = set(indices_c)
    union = set_a | set_b | set_c
    if not union:
        return 1.0
    return len(set_a & set_b & set_c) / len(union)


def build_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def to_layer_key(layer_index: int) -> str:
    return f"layer_{layer_index:02d}"


def main() -> None:
    args = parse_args()
    top_ratio = validate_top_ratio(args.top_ratio)
    model_dirs = discover_model_dirs(args.input_dir, args.include_qr_subspace)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratio_tag = f"t{int(round(top_ratio * 1000)):03d}"
    output_prefix = args.output_dir / f"top_feature_jaccard_{args.ranking}_{ratio_tag}"
    csv_path = output_prefix.with_suffix(".csv")
    summary_path = output_prefix.with_suffix(".summary.json")
    indices_path = output_prefix.with_suffix(".indices.json")

    csv_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "ranking": args.ranking,
        "top_ratio": top_ratio,
        "math_source": "step_mark_code.pt",
        "included_models": [],
        "models": {},
    }
    indices_payload: dict[str, object] = {
        "ranking": args.ranking,
        "top_ratio": top_ratio,
        "math_source": "step_mark_code.pt",
        "models": {},
    }

    for model_dir in model_dirs:
        model_name = model_dir.name
        vector_indices: dict[str, list[list[int]]] = {}
        hidden_dim: int | None = None
        layer_count: int | None = None
        top_k: int | None = None

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

            selected_indices, current_top_k = select_top_indices_per_layer(
                tensor,
                top_ratio=top_ratio,
                ranking=args.ranking,
            )
            if top_k is None:
                top_k = current_top_k
            elif top_k != current_top_k:
                raise ValueError(f"Inconsistent top-k inside {model_name}.")
            vector_indices[vector_name] = selected_indices

        assert hidden_dim is not None
        assert layer_count is not None
        assert top_k is not None

        per_pair_values: dict[str, list[float]] = {f"{left}_{right}": [] for left, right in PAIR_NAMES}
        per_pair_best_layers: dict[str, dict[str, float | int]] = {}
        per_pair_worst_layers: dict[str, dict[str, float | int]] = {}
        three_way_values: list[float] = []
        layerwise_indices = {
            vector_name: {
                to_layer_key(layer_index + 1): indices
                for layer_index, indices in enumerate(indices_by_layer)
            }
            for vector_name, indices_by_layer in vector_indices.items()
        }

        for layer_index in range(layer_count):
            row: dict[str, object] = {
                "model": model_name,
                "layer_index": layer_index + 1,
                "hidden_dim": hidden_dim,
                "top_k": top_k,
            }

            for left, right in PAIR_NAMES:
                pair_key = f"{left}_{right}"
                score = jaccard_similarity(
                    vector_indices[left][layer_index],
                    vector_indices[right][layer_index],
                )
                per_pair_values[pair_key].append(score)
                row[pair_key] = score

            three_way_score = jaccard_similarity_three(
                vector_indices["math"][layer_index],
                vector_indices["search"][layer_index],
                vector_indices["askuser"][layer_index],
            )
            three_way_values.append(three_way_score)
            row["math_search_askuser"] = three_way_score
            csv_rows.append(row)

        pair_summaries: dict[str, dict[str, object]] = {}
        for pair_key, values in per_pair_values.items():
            best_layer_index = max(range(len(values)), key=values.__getitem__)
            worst_layer_index = min(range(len(values)), key=values.__getitem__)
            per_pair_best_layers[pair_key] = {
                "layer_index": best_layer_index + 1,
                "jaccard": values[best_layer_index],
            }
            per_pair_worst_layers[pair_key] = {
                "layer_index": worst_layer_index + 1,
                "jaccard": values[worst_layer_index],
            }
            pair_summaries[pair_key] = {
                **build_summary(values),
                "best_layer": per_pair_best_layers[pair_key],
                "worst_layer": per_pair_worst_layers[pair_key],
            }

        three_way_summary = build_summary(three_way_values)
        three_way_summary["best_layer"] = {
            "layer_index": max(range(len(three_way_values)), key=three_way_values.__getitem__) + 1,
            "jaccard": max(three_way_values),
        }
        three_way_summary["worst_layer"] = {
            "layer_index": min(range(len(three_way_values)), key=three_way_values.__getitem__) + 1,
            "jaccard": min(three_way_values),
        }

        summary["included_models"].append(model_name)
        summary["models"][model_name] = {
            "layer_count": layer_count,
            "hidden_dim": hidden_dim,
            "top_k": top_k,
            "pairwise_jaccard_summary": pair_summaries,
            "three_way_jaccard_summary": three_way_summary,
        }
        indices_payload["models"][model_name] = {
            "layer_count": layer_count,
            "hidden_dim": hidden_dim,
            "top_k": top_k,
            "top_feature_indices": layerwise_indices,
        }

    fieldnames = [
        "model",
        "layer_index",
        "hidden_dim",
        "top_k",
        "math_search",
        "math_askuser",
        "search_askuser",
        "math_search_askuser",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with indices_path.open("w", encoding="utf-8") as handle:
        json.dump(indices_payload, handle, indent=2, ensure_ascii=False)

    print(f"Wrote per-layer Jaccard CSV to {csv_path}")
    print(f"Wrote summary JSON to {summary_path}")
    print(f"Wrote top-index JSON to {indices_path}")


if __name__ == "__main__":
    main()
