#!/usr/bin/env python3
"""Utilities for classifier-gated heading steering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_heading_classifier_payload(path: Path, num_model_layers: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    required_keys = ("weight", "bias", "feature_mean", "feature_std", "layer_id")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"Classifier payload at {path} is missing keys: {', '.join(missing)}")

    layer_id = int(payload["layer_id"])
    if not (1 <= layer_id <= num_model_layers):
        raise ValueError(
            f"Classifier payload layer_id={layer_id} is incompatible with a model that has {num_model_layers} layers."
        )

    return {
        "path": str(path),
        "layer_id": layer_id,
        "model_layer_idx": layer_id - 1,
        "weight": payload["weight"].detach().float().cpu(),
        "bias": float(payload["bias"]),
        "feature_mean": payload["feature_mean"].detach().float().cpu(),
        "feature_std": payload["feature_std"].detach().float().cpu(),
        "classifier_target": payload.get("classifier_target", "unknown"),
        "label_definition": payload.get("label_definition"),
    }


def run_heading_classifier(
    hidden_states: tuple[torch.Tensor, ...],
    classifier_payload: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    model_layer_idx = classifier_payload["model_layer_idx"]
    raw_state = hidden_states[model_layer_idx + 1][0, -1, :].detach().float().cpu()
    normalized_state = (raw_state - classifier_payload["feature_mean"]) / classifier_payload["feature_std"]
    logit = torch.dot(normalized_state, classifier_payload["weight"]) + classifier_payload["bias"]
    probability = torch.sigmoid(logit).item()
    return {
        "classifier_path": classifier_payload["path"],
        "classifier_target": classifier_payload["classifier_target"],
        "classifier_layer_id": int(classifier_payload["layer_id"]),
        "classifier_model_layer": int(model_layer_idx),
        "threshold": float(threshold),
        "logit": float(logit.item()),
        "probability": float(probability),
        "apply_steering": bool(probability >= threshold),
        "hidden_state_norm": float(torch.linalg.vector_norm(raw_state).item()),
    }
