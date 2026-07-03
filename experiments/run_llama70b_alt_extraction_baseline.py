#!/usr/bin/env python3
"""Run Llama-70B alternative extraction baselines across tool domains.

This script is intentionally self-contained: it builds alternative steering
directions from target-tool-vs-Reasoning heading-anchor states, exports them
in the existing inference payload format, runs inference, optionally judges
Math outputs, and writes compact rebuttal tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_PATH = Path("/data/yuqi/Steering-tooloveruse/model_weights/Llama-3.1-70B-Instruct")
DEFAULT_MEAN_DIFF_DIR = Path("/data/yuqi/SteeringMark/steering_vector/Llama_3_70B_vector_heading")
DEFAULT_OUTPUT_ROOT = Path("/data/yuqi/SteeringMark/results/ablations")

INFERENCE_SCRIPT = PROJECT_ROOT / "inference" / "inference_tool_prompt_tag_suppressed_kvcache.py"
MATH_JUDGE_SCRIPT = PROJECT_ROOT / "evaluate" / "inference_eval_math.py"

NEGATIVE_HEADING = "### Reasoning"
CAUSAL_DIRECTIONS = ("MeanDiff", "LinearProbe", "DiagWhitenedMeanDiff", "Random", "ShuffledProbe")
DOMAIN_CONFIGS = {
    "math": {
        "target_tag": "code",
        "positive_heading": "### Code",
        "target_tool": "Code",
        "payload_name": "step_mark_code.pt",
        "extraction_data_path": Path("/data/yuqi/SteeringMark/steering_/steering_data_code_20.json"),
        "eval_data_path": Path("/data/yuqi/SteeringMark/data_inference/domain_math_tool_prompt.json"),
        "output_dir": DEFAULT_OUTPUT_ROOT / "llama70b_alt_extraction",
        "supports_judge": True,
    },
    "time": {
        "target_tag": "search",
        "positive_heading": "### Search",
        "target_tool": "Search",
        "payload_name": "step_mark_search.pt",
        "extraction_data_path": Path("/data/yuqi/SteeringMark/steering_/steering_data_search_20.json"),
        "eval_data_path": Path("/data/yuqi/SteeringMark/data_inference/domain_time_tool_prompt.json"),
        "output_dir": DEFAULT_OUTPUT_ROOT / "llama70b_alt_extraction_time",
        "supports_judge": False,
    },
    "intention": {
        "target_tag": "askuser",
        "positive_heading": "### AskUser",
        "target_tool": "AskUser",
        "payload_name": "step_mark_askuser.pt",
        "extraction_data_path": Path("/data/yuqi/SteeringMark/steering_/steering_data_askuser_20.json"),
        "eval_data_path": Path("/data/yuqi/SteeringMark/data_inference/domain_intention_tool_prompt.json"),
        "output_dir": DEFAULT_OUTPUT_ROOT / "llama70b_alt_extraction_intention",
        "supports_judge": False,
    },
}


@dataclass
class HeadingExample:
    record_index: int
    heading_tag: str
    token_position: int
    feature: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "infer", "judge", "summarize", "all"), default="all")
    parser.add_argument("--domain", choices=tuple(DOMAIN_CONFIGS), default="math")
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--extraction-data-path", type=Path, default=None)
    parser.add_argument("--eval-data-path", type=Path, default=None)
    parser.add_argument("--mean-diff-vector-dir", type=Path, default=DEFAULT_MEAN_DIFF_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated integer seeds.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-records", type=int, default=None, help="Limit extraction records for smoke tests.")
    parser.add_argument("--max-eval-examples", type=int, default=200)
    parser.add_argument("--test-start-id", type=int, default=0)
    parser.add_argument("--device", default="auto", help='Use "auto", "cuda", or "cpu".')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--judge-workers", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()




def apply_domain_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = DOMAIN_CONFIGS[args.domain]
    if args.extraction_data_path is None:
        args.extraction_data_path = config["extraction_data_path"]
    if args.eval_data_path is None:
        args.eval_data_path = config["eval_data_path"]
    if args.output_dir is None:
        args.output_dir = config["output_dir"]
    return args


def domain_config(args: argparse.Namespace) -> dict[str, Any]:
    return DOMAIN_CONFIGS[args.domain]


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty.")
    return seeds


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_tokenizer(model_name_or_path: Path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_input_device(model: AutoModelForCausalLM, device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if hasattr(model, "hf_device_map"):
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        if isinstance(first_device, str) and first_device not in {"cpu", "disk"}:
            return torch.device(first_device)
    return next(model.parameters()).device


def load_model_and_tokenizer(model_name_or_path: Path, device_arg: str):
    tokenizer = load_tokenizer(model_name_or_path)
    model_kwargs: dict[str, Any] = {"dtype": "auto"}
    if device_arg == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_name_or_path), **model_kwargs)
    if device_arg in {"cpu", "cuda"}:
        model = model.to(torch.device(device_arg))
    model.eval()
    return tokenizer, model, resolve_input_device(model, device_arg)


def build_messages(instruction: str, input_text: str, output_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text.strip()},
    ]


def render_messages_without_template(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(message["content"].strip() for message in messages if message["content"].strip())


def build_full_sequence_ids(
    tokenizer: AutoTokenizer,
    instruction: str,
    input_text: str,
    output_text: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    messages = build_messages(instruction, input_text, output_text)
    if getattr(tokenizer, "chat_template", None):
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
    else:
        full_ids = tokenizer(render_messages_without_template(messages), add_special_tokens=True, return_tensors="pt").input_ids
    full_ids = full_ids.to(device)
    return full_ids, full_ids[0].detach().cpu().tolist()


def find_subsequence_spans(sequence: list[int], subsequence: list[int]) -> list[tuple[int, int]]:
    spans = []
    width = len(subsequence)
    if width == 0:
        return spans
    for start in range(0, len(sequence) - width + 1):
        if sequence[start : start + width] == subsequence:
            spans.append((start, start + width))
    return spans


def find_last_subsequence_span(sequence: list[int], subsequence: list[int]) -> tuple[int, int]:
    spans = find_subsequence_spans(sequence, subsequence)
    if not spans:
        raise ValueError("Could not align assistant output token sequence inside the full prompt sequence.")
    return spans[-1]


def find_text_spans(text: str, pattern: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(pattern)))
        start = idx + len(pattern)
    return spans


def map_char_position_to_token_index(offsets: list[tuple[int, int]], char_position: int) -> int | None:
    for idx, (token_char_start, token_char_end) in enumerate(offsets):
        if token_char_end <= char_position:
            continue
        if token_char_start > char_position:
            return idx
        return idx
    return None


def extract_record_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict[str, str],
    record_index: int,
    saved_layer: int,
    device: torch.device,
    positive_heading: str,
    positive_tag: str,
) -> list[HeadingExample]:
    assistant_text = record["output"].strip()
    full_ids, sequence_ids = build_full_sequence_ids(
        tokenizer=tokenizer,
        instruction=record["instruction"],
        input_text=record["input"],
        output_text=assistant_text,
        device=device,
    )
    assistant_encoding = tokenizer(
        assistant_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    assistant_ids = assistant_encoding["input_ids"]
    if not assistant_ids:
        return []
    assistant_offsets = [tuple(item) for item in assistant_encoding["offset_mapping"]]
    assistant_start, _assistant_end = find_last_subsequence_span(sequence_ids, assistant_ids)

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True, use_cache=False, return_dict=True)
    if outputs.hidden_states is None or saved_layer >= len(outputs.hidden_states):
        raise ValueError(f"Saved layer {saved_layer} is unavailable in model outputs.")

    examples: list[HeadingExample] = []
    for heading_tag, heading_text in [("reasoning", NEGATIVE_HEADING), (positive_tag, positive_heading)]:
        for char_start, _char_end in find_text_spans(assistant_text, heading_text):
            local_token_index = map_char_position_to_token_index(assistant_offsets, char_start)
            if local_token_index is None:
                continue
            token_position = assistant_start + local_token_index
            feature = outputs.hidden_states[saved_layer][0, token_position, :].detach().float().cpu()
            examples.append(
                HeadingExample(
                    record_index=record_index,
                    heading_tag=heading_tag,
                    token_position=int(token_position),
                    feature=feature,
                )
            )
    return examples


def extract_features(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = args.output_dir / f"heading_features_{args.domain}_layer{args.layer}.pt"
    if cache_path.exists() and not args.overwrite and args.max_records is None:
        return torch.load(cache_path, map_location="cpu")

    records = load_json(args.extraction_data_path)
    if args.max_records is not None:
        records = records[: args.max_records]
    tokenizer, model, device = load_model_and_tokenizer(args.model_name_or_path, args.device)
    config = domain_config(args)

    examples: list[HeadingExample] = []
    for record_index, record in enumerate(records):
        examples.extend(
            extract_record_examples(
                model=model,
                tokenizer=tokenizer,
                record=record,
                record_index=record_index,
                saved_layer=args.layer,
                device=device,
                positive_heading=config["positive_heading"],
                positive_tag=config["target_tag"],
            )
        )
    if not examples:
        raise ValueError("No heading examples were extracted.")

    features = torch.stack([example.feature for example in examples], dim=0)
    labels = torch.tensor([1 if example.heading_tag == config["target_tag"] else 0 for example in examples], dtype=torch.long)
    record_indices = torch.tensor([example.record_index for example in examples], dtype=torch.long)
    metadata = [
        {
            "record_index": example.record_index,
            "heading_tag": example.heading_tag,
            "token_position": example.token_position,
        }
        for example in examples
    ]
    payload = {
        "features": features,
        "labels": labels,
        "record_indices": record_indices,
        "metadata": metadata,
        "layer": args.layer,
        "model_name_or_path": str(args.model_name_or_path),
        "extraction_data_path": str(args.extraction_data_path),
        "domain": args.domain,
        "target_tag": config["target_tag"],
    }
    if args.max_records is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    return payload


def split_by_record(record_indices: torch.Tensor, labels: torch.Tensor, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    records = sorted({int(value) for value in record_indices.tolist()})
    rng.shuffle(records)
    train_record_count = max(1, min(len(records) - 1, int(round(len(records) * train_ratio)))) if len(records) > 1 else len(records)
    train_records = set(records[:train_record_count])
    train_idx = [idx for idx, record_index in enumerate(record_indices.tolist()) if int(record_index) in train_records]
    val_idx = [idx for idx, record_index in enumerate(record_indices.tolist()) if int(record_index) not in train_records]
    if val_idx and len(set(labels[val_idx].tolist())) < 2:
        # Fall back to a stratified occurrence split for tiny smoke tests where a
        # record-level validation split does not contain both classes.
        return split_by_label(labels, train_ratio, seed)
    return train_idx, val_idx


def split_by_label(labels: torch.Tensor, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    positives = [idx for idx, value in enumerate(labels.tolist()) if int(value) == 1]
    negatives = [idx for idx, value in enumerate(labels.tolist()) if int(value) == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_bucket(bucket: list[int]) -> tuple[list[int], list[int]]:
        if len(bucket) <= 1:
            return bucket[:], []
        train_size = max(1, min(len(bucket) - 1, int(round(len(bucket) * train_ratio))))
        return bucket[:train_size], bucket[train_size:]

    pos_train, pos_val = split_bucket(positives)
    neg_train, neg_val = split_bucket(negatives)
    train_idx = pos_train + neg_train
    val_idx = pos_val + neg_val
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def standardize_train_full(train_x: torch.Tensor, full_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (full_x - mean) / std, mean, std


def rank_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().cpu().long()
    scores = scores.detach().cpu().float()
    positives = labels == 1
    negatives = labels == 0
    num_pos = int(positives.sum().item())
    num_neg = int(negatives.sum().item())
    if num_pos == 0 or num_neg == 0:
        return math.nan
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    start = 0
    # Average ranks for ties. Ranks are 1-indexed for the Mann-Whitney formula.
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and sorted_scores[end].item() == sorted_scores[start].item():
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    pos_rank_sum = float(ranks[positives].sum().item())
    return (pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)


def binary_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    labels = labels.detach().cpu().long()
    scores = scores.detach().cpu().float()
    if labels.numel() == 0:
        return {"auroc": math.nan, "accuracy": math.nan, "f1": math.nan}
    preds = (scores >= 0.5).long()
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "auroc": float(rank_auc(labels, scores)),
        "accuracy": float((tp + tn) / labels.numel()),
        "f1": float(f1),
    }


def normalize_like_reference(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    vector = vector.detach().float().cpu()
    reference = reference.detach().float().cpu()
    if torch.dot(vector, reference).item() < 0:
        vector = -vector
    vector_norm = torch.linalg.vector_norm(vector)
    reference_norm = torch.linalg.vector_norm(reference)
    if vector_norm.item() == 0:
        return vector
    return vector * (reference_norm / vector_norm)


def cosine_with_reference(vector: torch.Tensor, reference: torch.Tensor) -> float:
    if torch.linalg.vector_norm(vector).item() == 0 or torch.linalg.vector_norm(reference).item() == 0:
        return float("nan")
    return float(F.cosine_similarity(vector.unsqueeze(0), reference.unsqueeze(0), dim=1).item())


def layer_vector_from_payload(payload_path: Path, saved_layer: int) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    payload = torch.load(payload_path, map_location="cpu")
    vectors = payload["steering_vectors"].detach().float().cpu()
    layer_indices = [int(value) for value in payload["layer_indices"]]
    if saved_layer not in layer_indices:
        raise ValueError(f"Layer {saved_layer} is not present in {payload_path}.")
    vector_idx = layer_indices.index(saved_layer)
    return vectors[vector_idx], vectors, layer_indices, vector_idx


def make_full_payload(
    direction: torch.Tensor,
    template_vectors: torch.Tensor,
    layer_indices: list[int],
    vector_idx: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    steering_vectors = torch.zeros_like(template_vectors)
    steering_vectors[vector_idx] = direction.detach().float().cpu()
    return {
        "steering_vectors": steering_vectors,
        "layer_indices": layer_indices,
        **metadata,
    }


def train_linear_probe_direction(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    shuffled: bool,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    torch.manual_seed(seed)
    train_x = features[train_idx]
    full_x_z, _mean, std = standardize_train_full(train_x, features)
    train_y = labels[train_idx].float().clone()
    if shuffled:
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(train_y.numel(), generator=generator)
        train_y = train_y[permutation]

    model = nn.Linear(full_x_z.shape[1], 1)
    positives = float((train_y == 1).sum().item())
    negatives = float((train_y == 0).sum().item())
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32) if positives else torch.tensor([1.0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    train_z = full_x_z[train_idx]
    for _epoch in range(1000):
        optimizer.zero_grad()
        logits = model(train_z).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(full_x_z).squeeze(-1)).detach().cpu()
    metrics = binary_metrics(labels[val_idx], scores[val_idx]) if val_idx else binary_metrics(labels[train_idx], scores[train_idx])
    weight_z = model.weight.detach().cpu().squeeze(0)
    weight_raw = weight_z / std
    return weight_raw, metrics


def mean_diff_direction(features: torch.Tensor, labels: torch.Tensor, train_idx: list[int]) -> torch.Tensor:
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    positive = train_features[train_labels == 1]
    negative = train_features[train_labels == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("Need both positive and negative training examples.")
    return positive.mean(dim=0) - negative.mean(dim=0)


def diag_whitened_direction(features: torch.Tensor, labels: torch.Tensor, train_idx: list[int]) -> torch.Tensor:
    train_features = features[train_idx]
    train_labels = labels[train_idx]
    positive = train_features[train_labels == 1]
    negative = train_features[train_labels == 0]
    if positive.shape[0] < 2 or negative.shape[0] < 2:
        return mean_diff_direction(features, labels, train_idx)
    pooled_var = 0.5 * (positive.var(dim=0, unbiased=False) + negative.var(dim=0, unbiased=False))
    return (positive.mean(dim=0) - negative.mean(dim=0)) / (pooled_var + 1e-4)


def direction_score_metrics(
    direction: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    val_idx: list[int],
) -> dict[str, float]:
    projection = features @ direction
    selected_idx = val_idx if val_idx else list(range(features.shape[0]))
    selected_scores = projection[selected_idx]
    if selected_scores.numel() == 0:
        return {"auroc": math.nan, "accuracy": math.nan, "f1": math.nan}
    threshold = float(selected_scores.median().item())
    probs = torch.sigmoid(selected_scores - threshold)
    return binary_metrics(labels[selected_idx], probs)


def pairwise_stability(vectors: list[torch.Tensor]) -> float:
    values = []
    for left_idx in range(len(vectors)):
        for right_idx in range(left_idx + 1, len(vectors)):
            values.append(cosine_with_reference(vectors[left_idx], vectors[right_idx]))
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def stage_build(args: argparse.Namespace) -> None:
    seeds = parse_seeds(args.seeds)
    config = domain_config(args)
    mean_payload_path = args.mean_diff_vector_dir / config["payload_name"]
    reference_vector, template_vectors, layer_indices, vector_idx = layer_vector_from_payload(mean_payload_path, args.layer)

    if args.dry_run:
        print(f"Would load extraction data: {args.extraction_data_path}")
        print(f"Would load model: {args.model_name_or_path}")
        print(f"Would use mean-diff payload: {mean_payload_path}")
        print(f"Would write outputs under: {args.output_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = args.output_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    feature_payload = extract_features(args)
    features = feature_payload["features"].float()
    labels = feature_payload["labels"].long()
    record_indices = feature_payload["record_indices"].long()

    geometry_rows: list[dict[str, Any]] = []
    direction_by_name_seed: dict[tuple[str, int], torch.Tensor] = {}
    first_seed_payloads: dict[str, torch.Tensor] = {}

    for seed in seeds:
        train_idx, val_idx = split_by_record(record_indices, labels, args.train_ratio, seed)
        train_mean_diff = normalize_like_reference(mean_diff_direction(features, labels, train_idx), reference_vector)
        train_mean_diff_metrics = direction_score_metrics(train_mean_diff, features, labels, val_idx)
        direction_by_name_seed[("MeanDiffTrainSplit", seed)] = train_mean_diff
        geometry_rows.append(
            {
                "direction": "MeanDiffTrainSplit",
                "seed": seed,
                "layer": args.layer,
                "val_auroc": train_mean_diff_metrics["auroc"],
                "val_accuracy": train_mean_diff_metrics["accuracy"],
                "val_f1": train_mean_diff_metrics["f1"],
                "cosine_with_existing_mean_diff": cosine_with_reference(train_mean_diff, reference_vector),
                "num_train": len(train_idx),
                "num_val": len(val_idx),
            }
        )

        probe_direction, probe_metrics = train_linear_probe_direction(features, labels, train_idx, val_idx, False, seed)
        probe_direction = normalize_like_reference(probe_direction, reference_vector)
        direction_by_name_seed[("LinearProbe", seed)] = probe_direction
        geometry_rows.append(
            {
                "direction": "LinearProbe",
                "seed": seed,
                "layer": args.layer,
                "val_auroc": probe_metrics["auroc"],
                "val_accuracy": probe_metrics["accuracy"],
                "val_f1": probe_metrics["f1"],
                "cosine_with_existing_mean_diff": cosine_with_reference(probe_direction, reference_vector),
                "num_train": len(train_idx),
                "num_val": len(val_idx),
            }
        )

        whitened_direction = normalize_like_reference(diag_whitened_direction(features, labels, train_idx), reference_vector)
        whitened_metrics = direction_score_metrics(whitened_direction, features, labels, val_idx)
        direction_by_name_seed[("DiagWhitenedMeanDiff", seed)] = whitened_direction
        geometry_rows.append(
            {
                "direction": "DiagWhitenedMeanDiff",
                "seed": seed,
                "layer": args.layer,
                "val_auroc": whitened_metrics["auroc"],
                "val_accuracy": whitened_metrics["accuracy"],
                "val_f1": whitened_metrics["f1"],
                "cosine_with_existing_mean_diff": cosine_with_reference(whitened_direction, reference_vector),
                "num_train": len(train_idx),
                "num_val": len(val_idx),
            }
        )

        shuffled_direction, shuffled_metrics = train_linear_probe_direction(features, labels, train_idx, val_idx, True, seed)
        shuffled_direction = normalize_like_reference(shuffled_direction, reference_vector)
        direction_by_name_seed[("ShuffledProbe", seed)] = shuffled_direction
        geometry_rows.append(
            {
                "direction": "ShuffledProbe",
                "seed": seed,
                "layer": args.layer,
                "val_auroc": shuffled_metrics["auroc"],
                "val_accuracy": shuffled_metrics["accuracy"],
                "val_f1": shuffled_metrics["f1"],
                "cosine_with_existing_mean_diff": cosine_with_reference(shuffled_direction, reference_vector),
                "num_train": len(train_idx),
                "num_val": len(val_idx),
            }
        )

        if seed == seeds[0]:
            first_seed_payloads["LinearProbe"] = probe_direction
            first_seed_payloads["DiagWhitenedMeanDiff"] = whitened_direction
            first_seed_payloads["ShuffledProbe"] = shuffled_direction

    random_generator = torch.Generator().manual_seed(seeds[0])
    random_direction = normalize_like_reference(torch.randn(reference_vector.shape, generator=random_generator), reference_vector)
    first_seed_payloads["Random"] = random_direction
    random_metrics = direction_score_metrics(random_direction, features, labels, list(range(features.shape[0])))
    geometry_rows.append(
        {
            "direction": "Random",
            "seed": seeds[0],
            "layer": args.layer,
            "val_auroc": random_metrics["auroc"],
            "val_accuracy": random_metrics["accuracy"],
            "val_f1": random_metrics["f1"],
            "cosine_with_existing_mean_diff": cosine_with_reference(random_direction, reference_vector),
            "num_train": "",
            "num_val": features.shape[0],
        }
    )

    for direction_name in ["MeanDiffTrainSplit", "LinearProbe", "DiagWhitenedMeanDiff", "ShuffledProbe"]:
        seed_vectors = [direction_by_name_seed[(direction_name, seed)] for seed in seeds if (direction_name, seed) in direction_by_name_seed]
        stability = pairwise_stability(seed_vectors)
        for row in geometry_rows:
            if row["direction"] == direction_name:
                row["seed_pairwise_cosine_stability"] = stability
    for row in geometry_rows:
        row.setdefault("seed_pairwise_cosine_stability", "")

    for direction_name, direction in first_seed_payloads.items():
        filename = {
            "LinearProbe": "linear_probe_seed42.pt",
            "DiagWhitenedMeanDiff": "diag_whitened_seed42.pt",
            "Random": "random_seed42.pt",
            "ShuffledProbe": "shuffled_probe_seed42.pt",
        }[direction_name]
        payload = make_full_payload(
            direction=direction,
            template_vectors=template_vectors,
            layer_indices=layer_indices,
            vector_idx=vector_idx,
            metadata={
                "direction_name": direction_name,
                "reference_payload": str(mean_payload_path),
                "reference_norm_matched": True,
                "orientation_reference": f"existing_{config['payload_name']}_layer{args.layer}",
                "layer": args.layer,
                "seed": seeds[0],
            },
        )
        torch.save(payload, payload_dir / filename)

    write_csv(
        args.output_dir / "geometry_metrics.csv",
        geometry_rows,
        [
            "direction",
            "seed",
            "layer",
            "val_auroc",
            "val_accuracy",
            "val_f1",
            "cosine_with_existing_mean_diff",
            "seed_pairwise_cosine_stability",
            "num_train",
            "num_val",
        ],
    )
    save_json(
        args.output_dir / "build_summary.json",
        {
            "domain": args.domain,
            "target_tag": config["target_tag"],
            "positive_heading": config["positive_heading"],
            "model_name_or_path": str(args.model_name_or_path),
            "extraction_data_path": str(args.extraction_data_path),
        "domain": args.domain,
        "target_tag": config["target_tag"],
            "mean_diff_payload": str(mean_payload_path),
            "layer": args.layer,
            "seeds": seeds,
            "num_heading_examples": int(features.shape[0]),
            "num_positive": int((labels == 1).sum().item()),
            "num_negative": int((labels == 0).sum().item()),
            "payload_dir": str(payload_dir),
        },
    )


def result_path(args: argparse.Namespace, direction_name: str) -> Path:
    return args.output_dir / "inference_outputs" / (
        f"llama70b_{args.domain}_alt_extraction_{direction_name.lower()}_layer{args.layer}_n{args.max_eval_examples}.json"
    )


def inference_command(args: argparse.Namespace, direction_name: str, save_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--model_name_or_path",
        str(args.model_name_or_path),
        "--data_path",
        str(args.eval_data_path),
        "--save_path",
        str(save_path),
        "--domain",
        args.domain,
        "--steering_layers",
        str(args.layer),
        "--suppress_scale",
        "1.0",
        "--method",
        "llama",
        "--prompt-policy",
        "base_tool",
        "--extract-schema",
        "markdown",
        "--eval-schema",
        "markdown",
        "--schema",
        "markdown",
        "--code-heading",
        "Code",
        "--ablation",
        "none",
        "--max_test_num",
        str(args.max_eval_examples),
        "--test_start_id",
        str(args.test_start_id),
        "--device",
        args.device,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--max_steps",
        str(args.max_steps),
        "--quiet",
    ]
    config = domain_config(args)
    if direction_name == "MeanDiff":
        command.extend(["--steering_vector_dir", str(args.mean_diff_vector_dir), "--steering_payload_name", config["payload_name"]])
    else:
        payload_name = {
            "LinearProbe": "linear_probe_seed42.pt",
            "DiagWhitenedMeanDiff": "diag_whitened_seed42.pt",
            "Random": "random_seed42.pt",
            "ShuffledProbe": "shuffled_probe_seed42.pt",
        }[direction_name]
        command.extend(["--steering_vector_dir", str(args.output_dir / "payloads"), "--steering_payload_name", payload_name])
    if args.overwrite:
        command.append("--overwrite")
    return command


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True, cwd=str(cwd))


def stage_infer(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for direction_name in CAUSAL_DIRECTIONS:
        save_path = result_path(args, direction_name)
        if save_path.exists() and not args.overwrite:
            print(f"Skipping existing inference result: {save_path}")
            continue
        command = inference_command(args, direction_name, save_path)
        run_command(command, PROJECT_ROOT, args.dry_run)


def stage_judge(args: argparse.Namespace) -> None:
    config = domain_config(args)
    if not config["supports_judge"]:
        print(f"Judge stage is only wired for math outputs; skipping judge for domain={args.domain}.")
        return
    data_paths = [result_path(args, direction_name) for direction_name in CAUSAL_DIRECTIONS if result_path(args, direction_name).exists()]
    if not data_paths:
        print("No inference outputs found to judge.")
        return
    command = [
        sys.executable,
        str(MATH_JUDGE_SCRIPT),
        "--model",
        args.judge_model,
        "--max-workers",
        str(args.judge_workers),
    ]
    for data_path in data_paths:
        command.extend(["--data-path", str(data_path)])
    if args.overwrite:
        command.append("--overwrite")
    try:
        run_command(command, PROJECT_ROOT, args.dry_run)
    except subprocess.CalledProcessError as error:
        print(f"Judge stage failed with exit code {error.returncode}; continuing without accuracy metrics.")


def count_target_tools(records: list[dict[str, Any]], target_tool: str) -> int:
    total = 0
    for record in records:
        for step in record.get("predict", []):
            if step.get("type") == "tool" and step.get("tool_name") == target_tool:
                total += 1
    return total


def summarize_inference_result(path: Path, target_tool: str) -> dict[str, Any]:
    notes: list[str] = []
    records = load_json(path) if path.exists() else []
    if not records:
        notes.append("missing_result")
    judge_path = path.with_name(path.stem + "_judge.json")
    judged = load_json(judge_path) if judge_path.exists() else []
    metric_records = judged or records
    if not judged:
        notes.append("missing_judge")
    num_examples = len(metric_records)
    tool_avg_use = count_target_tools(metric_records, target_tool) / num_examples if num_examples else math.nan
    accuracy = math.nan
    if judged:
        judged_records = [record for record in judged if record.get("judge") in {"correct", "wrong"}]
        if judged_records:
            accuracy = sum(1 for record in judged_records if record.get("judge") == "correct") / len(judged_records)
    return {
        "tool_avg_use": tool_avg_use,
        "accuracy": accuracy,
        "num_examples": num_examples,
        "notes": ";".join(notes),
    }


def read_geometry_summary(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    summary: dict[str, dict[str, float]] = {}
    for direction in ["MeanDiffTrainSplit", "LinearProbe", "DiagWhitenedMeanDiff", "Random", "ShuffledProbe"]:
        selected = [row for row in rows if row["direction"] == direction]
        if not selected:
            continue
        def mean_float(key: str) -> float:
            values = [float(row[key]) for row in selected if row.get(key) not in {"", "nan", "NaN"}]
            return sum(values) / len(values) if values else math.nan

        summary[direction] = {
            "val_auroc": mean_float("val_auroc"),
            "val_accuracy": mean_float("val_accuracy"),
            "val_f1": mean_float("val_f1"),
            "cosine_with_existing_mean_diff": mean_float("cosine_with_existing_mean_diff"),
            "seed_pairwise_cosine_stability": mean_float("seed_pairwise_cosine_stability"),
        }
    return summary


def format_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(numeric):
        return ""
    return f"{numeric:.4f}"


def write_rebuttal_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Direction",
        "Val AUROC",
        "Val F1",
        "Cosine w/ MeanDiff",
        "Seed Stability",
        "ToolAvgUse",
        "Accuracy",
        "Notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["direction"],
                    format_float(row["val_auroc"]),
                    format_float(row["val_f1"]),
                    format_float(row["cosine_with_existing_mean_diff"]),
                    format_float(row["seed_pairwise_cosine_stability"]),
                    format_float(row["tool_avg_use"]),
                    format_float(row["accuracy"]),
                    row["notes"],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_summarize(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(f"Would summarize outputs under {args.output_dir}")
        return
    config = domain_config(args)
    geometry = read_geometry_summary(args.output_dir / "geometry_metrics.csv")
    direction_to_geometry_key = {
        "MeanDiff": "MeanDiffTrainSplit",
        "LinearProbe": "LinearProbe",
        "DiagWhitenedMeanDiff": "DiagWhitenedMeanDiff",
        "Random": "Random",
        "ShuffledProbe": "ShuffledProbe",
    }
    rows: list[dict[str, Any]] = []
    for direction_name in CAUSAL_DIRECTIONS:
        causal = summarize_inference_result(result_path(args, direction_name), config["target_tool"])
        geom = geometry.get(direction_to_geometry_key[direction_name], {})
        rows.append(
            {
                "direction": direction_name,
                "layer": args.layer,
                "val_auroc": geom.get("val_auroc", math.nan),
                "val_accuracy": geom.get("val_accuracy", math.nan),
                "val_f1": geom.get("val_f1", math.nan),
                "cosine_with_existing_mean_diff": geom.get("cosine_with_existing_mean_diff", math.nan),
                "seed_pairwise_cosine_stability": geom.get("seed_pairwise_cosine_stability", math.nan),
                "tool_avg_use": causal["tool_avg_use"],
                "accuracy": causal["accuracy"],
                "num_examples": causal["num_examples"],
                "notes": causal["notes"],
            }
        )
    write_csv(
        args.output_dir / "causal_metrics.csv",
        rows,
        [
            "direction",
            "layer",
            "val_auroc",
            "val_accuracy",
            "val_f1",
            "cosine_with_existing_mean_diff",
            "seed_pairwise_cosine_stability",
            "tool_avg_use",
            "accuracy",
            "num_examples",
            "notes",
        ],
    )
    write_rebuttal_table(args.output_dir / "rebuttal_table.md", rows)
    print(f"Wrote summary outputs under {args.output_dir}")


def main() -> int:
    args = apply_domain_defaults(parse_args())
    if args.stage in {"build", "all"}:
        stage_build(args)
    if args.stage in {"infer", "all"}:
        stage_infer(args)
    if args.stage in {"judge", "all"}:
        stage_judge(args)
    if args.stage in {"summarize", "all"}:
        stage_summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
