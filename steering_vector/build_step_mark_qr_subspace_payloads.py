#!/usr/bin/env python3
"""Build QR-orthonormal subspace payloads from existing step-mark steering vectors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


VECTOR_GROUPS = ("search", "code", "askuser", "search_askuser", "all")
GROUP_TO_SOURCE_NAMES = {
    "search": ["search"],
    "code": ["code"],
    "askuser": ["askuser"],
    "search_askuser": ["search", "askuser"],
    "all": ["search", "code", "askuser"],
}
DEFAULT_INPUT_DIR = Path("/data/yuqi/SteeringMark/steering_vector/Llama_3_8_vector_heading")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load existing step_mark_{search,code,askuser}.pt payloads, build a per-layer "
            "QR orthonormal basis for the requested tool group, and save the resulting "
            "subspace-projection payloads into a separate directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the original step_mark_{search,code,askuser}.pt payloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where QR-subspace payloads will be written. Defaults to <input-dir>_qr_subspace.",
    )
    parser.add_argument(
        "--vector-groups",
        default="search,code,askuser,search_askuser,all",
        help="Comma-separated subset of: search,code,askuser,search_askuser,all",
    )
    return parser.parse_args()


def parse_csv_choices(raw_value: str, allowed: tuple[str, ...], arg_name: str) -> list[str]:
    values: list[str] = []
    seen = set()
    for piece in raw_value.split(","):
        value = piece.strip()
        if not value:
            continue
        if value not in allowed:
            raise ValueError(f"Unsupported {arg_name} value {value!r}. Allowed: {', '.join(allowed)}")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError(f"{arg_name} cannot be empty.")
    return values


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "steering_vectors" not in payload:
        raise ValueError(f"Payload at {path} is missing 'steering_vectors'.")
    if "layer_indices" not in payload:
        raise ValueError(f"Payload at {path} is missing 'layer_indices'.")
    return payload


def summarize_scalar_list(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def summarize_vector_norms_per_layer(vector: torch.Tensor) -> dict[str, Any]:
    norms = torch.linalg.vector_norm(vector.float(), dim=1).tolist()
    norm_list = [float(value) for value in norms]
    return {
        "per_layer": norm_list,
        "stats": summarize_scalar_list(norm_list),
    }


def collect_source_payloads(input_dir: Path) -> dict[str, dict[str, Any]]:
    source_payloads: dict[str, dict[str, Any]] = {}
    for source_name in ("search", "code", "askuser"):
        path = input_dir / f"step_mark_{source_name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Required source payload not found: {path}")
        payload = load_payload(path)
        payload["__payload_path__"] = str(path)
        source_payloads[source_name] = payload
    return source_payloads


def validate_source_payloads(source_payloads: dict[str, dict[str, Any]]) -> None:
    reference_name = "search"
    reference_payload = source_payloads[reference_name]
    reference_shape = tuple(reference_payload["steering_vectors"].shape)
    reference_layers = list(reference_payload["layer_indices"])
    reference_model = reference_payload.get("model_name_or_path")

    for source_name, payload in source_payloads.items():
        current_shape = tuple(payload["steering_vectors"].shape)
        if current_shape != reference_shape:
            raise ValueError(
                f"Shape mismatch between {reference_name} {reference_shape} and {source_name} {current_shape}."
            )
        current_layers = list(payload["layer_indices"])
        if current_layers != reference_layers:
            raise ValueError(
                f"Layer index mismatch between {reference_name} and {source_name}: "
                f"{reference_layers[:5]}... vs {current_layers[:5]}..."
            )
        current_model = payload.get("model_name_or_path")
        if current_model != reference_model:
            raise ValueError(
                f"model_name_or_path mismatch between {reference_name} and {source_name}: "
                f"{reference_model!r} vs {current_model!r}"
            )


def build_pairwise_cosine_summary(
    source_names: list[str],
    source_vectors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for left_name, right_name in itertools.combinations(source_names, 2):
        left = source_vectors[left_name].float()
        right = source_vectors[right_name].float()
        flattened_cosine = F.cosine_similarity(
            left.flatten().unsqueeze(0),
            right.flatten().unsqueeze(0),
            dim=1,
        ).item()
        layer_cosines = F.cosine_similarity(left, right, dim=1).tolist()
        layer_cosine_values = [float(value) for value in layer_cosines]
        summary[f"{left_name}__{right_name}"] = {
            "flattened_cosine": float(flattened_cosine),
            "per_layer": layer_cosine_values,
            "per_layer_stats": summarize_scalar_list(layer_cosine_values),
        }
    return summary


def build_group_payload(
    group_name: str,
    source_names: list[str],
    source_payloads: dict[str, dict[str, Any]],
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_payload = source_payloads[source_names[0]]
    layer_indices = list(reference_payload["layer_indices"])
    source_vectors = {
        source_name: source_payloads[source_name]["steering_vectors"].float().cpu()
        for source_name in source_names
    }
    source_tensor = torch.stack([source_vectors[source_name] for source_name in source_names], dim=0)
    num_layers = source_tensor.shape[1]

    basis_rows_per_layer = []
    qr_r_per_layer = []
    subspace_strengths: list[float] = []
    singular_values_per_layer = []
    effective_ranks: list[int] = []

    for layer_idx in range(num_layers):
        layer_matrix = source_tensor[:, layer_idx, :].transpose(0, 1).contiguous()
        column_norms = torch.linalg.vector_norm(layer_matrix, dim=0)
        if torch.any(column_norms <= 0):
            raise ValueError(
                f"Encountered a zero-norm source vector at saved layer {layer_indices[layer_idx]} "
                f"for group {group_name}."
            )

        q_matrix, r_matrix = torch.linalg.qr(layer_matrix, mode="reduced")
        singular_values = torch.linalg.svdvals(layer_matrix).float().cpu()
        basis_rows_per_layer.append(q_matrix.transpose(0, 1).contiguous().cpu())
        qr_r_per_layer.append(r_matrix.contiguous().cpu())
        singular_values_per_layer.append(singular_values)
        subspace_strengths.append(float(torch.linalg.matrix_norm(layer_matrix, ord="fro").item()))
        effective_ranks.append(int(torch.linalg.matrix_rank(layer_matrix).item()))

    subspace_basis = torch.stack(basis_rows_per_layer, dim=0)
    qr_r_factors = torch.stack(qr_r_per_layer, dim=0)
    singular_value_tensor = torch.stack(singular_values_per_layer, dim=0)

    payload = {
        "payload_type": "qr_subspace_projection",
        "subspace_basis": subspace_basis,
        "qr_r_factors": qr_r_factors,
        "source_vectors": source_tensor,
        "source_groups": source_names,
        "layer_indices": layer_indices,
        "basis_dim": int(len(source_names)),
        "effective_ranks": effective_ranks,
        "subspace_strengths": subspace_strengths,
        "singular_values": singular_value_tensor,
        "basis_method": "torch.linalg.qr(mode='reduced')",
        "projection_rule": "hidden = hidden - scale * strength * proj_subspace(hidden)",
        "vector_group": group_name,
        "model_name_or_path": reference_payload.get("model_name_or_path"),
        "method": reference_payload.get("method"),
        "tag_token_mode": reference_payload.get("tag_token_mode"),
        "tag_span_mode": reference_payload.get("tag_span_mode"),
        "data_paths": reference_payload.get("data_paths", {}),
        "source_payload_paths": {
            source_name: source_payloads[source_name]["__payload_path__"] for source_name in source_names
        },
        "source_sample_counts": {
            source_name: int(source_payloads[source_name].get("sample_count", 0)) for source_name in source_names
        },
        "built_from_input_dir": str(input_dir),
    }

    source_vector_norms = {
        source_name: summarize_vector_norms_per_layer(source_vectors[source_name]) for source_name in source_names
    }
    singular_value_lists = singular_value_tensor.tolist()
    pairwise_cosine_summary = build_pairwise_cosine_summary(source_names, source_vectors)
    summary = {
        "payload_type": payload["payload_type"],
        "vector_group": group_name,
        "source_groups": source_names,
        "basis_dim": int(len(source_names)),
        "layer_indices": layer_indices,
        "basis_shape": list(subspace_basis.shape),
        "model_name_or_path": payload["model_name_or_path"],
        "method": payload["method"],
        "tag_token_mode": payload["tag_token_mode"],
        "tag_span_mode": payload["tag_span_mode"],
        "basis_method": payload["basis_method"],
        "projection_rule": payload["projection_rule"],
        "source_payload_paths": payload["source_payload_paths"],
        "source_sample_counts": payload["source_sample_counts"],
        "source_vector_norms": source_vector_norms,
        "subspace_strengths": subspace_strengths,
        "subspace_strength_stats": summarize_scalar_list(subspace_strengths),
        "effective_ranks": effective_ranks,
        "effective_rank_stats": summarize_scalar_list([float(value) for value in effective_ranks]),
        "singular_values": [[float(value) for value in row] for row in singular_value_lists],
        "pairwise_cosine_summary": pairwise_cosine_summary,
        "built_from_input_dir": str(input_dir),
    }
    return payload, summary


def resolve_output_dir(input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return input_dir.parent / f"{input_dir.name}_qr_subspace"


def main() -> int:
    args = parse_args()
    vector_groups = parse_csv_choices(args.vector_groups, VECTOR_GROUPS, "--vector-groups")
    output_dir = resolve_output_dir(args.input_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_payloads = collect_source_payloads(args.input_dir)
    validate_source_payloads(source_payloads)

    for group_name in vector_groups:
        payload, summary = build_group_payload(
            group_name=group_name,
            source_names=GROUP_TO_SOURCE_NAMES[group_name],
            source_payloads=source_payloads,
            input_dir=args.input_dir,
        )
        payload_path = output_dir / f"step_mark_{group_name}.pt"
        summary_path = output_dir / f"step_mark_{group_name}_summary.json"
        torch.save(payload, payload_path)
        save_json(summary_path, summary)
        print(f"Saved QR-subspace payload to {payload_path}")
        print(f"Saved QR-subspace summary to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
