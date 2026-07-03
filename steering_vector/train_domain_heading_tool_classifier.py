#!/usr/bin/env python3
"""Train domain-specific any-tool classifiers on ### heading positions."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = Path("/data/yuqi/Steering-tooloveruse/model_weights/Llama-3.1-8B-Instruct")
DEFAULT_DATA_DIR = Path("/data/yuqi/SteeringMark/steering_")
DEFAULT_OUTPUT_DIR = Path("/data/yuqi/SteeringMark/steering_vector/domain_tool_classifiers")

DOMAIN_TO_TARGET = {
    "math": ("code", DEFAULT_DATA_DIR / "steering_data_code_20.json"),
    "time": ("search", DEFAULT_DATA_DIR / "steering_data_search_20.json"),
    "intention": ("askuser", DEFAULT_DATA_DIR / "steering_data_askuser_20.json"),
}
TOOL_HEADING_TAGS = {"search", "code", "askuser"}
FULL_HEADING_TEXT = {
    "reasoning": "### Reasoning",
    "search": "### Search",
    "code": "### Code",
    "askuser": "### AskUser",
    "finalresponse": "### Final Response",
}


@dataclass
class HeadingExample:
    record_index: int
    record_label: str
    heading_tag: str
    heading_text: str
    token_position: int
    label: int
    layer_states: torch.Tensor  # [num_layers, hidden_dim]


class LinearProbe(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a domain-specific binary classifier on ### heading hidden states to predict "
            "whether the next heading is any tool class."
        )
    )
    parser.add_argument("--domain", choices=("math", "time", "intention"), required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional explicit dataset path. Defaults to the matching file under SteeringMark/steering_.",
    )
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method", choices=("llama", "mistral"), default="llama")
    parser.add_argument("--device", default="auto", help='Use "auto", "cuda", or "cpu".')
    parser.add_argument(
        "--layers",
        required=True,
        help="Saved layer ids to train on, e.g. '21' or '16-24' or '18,20,22'.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument(
        "--include-final-response-as-negative",
        action="store_true",
        help="Include `### Final Response` heading positions as negative examples.",
    )
    parser.add_argument(
        "--save-examples-limit",
        type=int,
        default=20,
        help="How many example metadata rows to keep in the summary.",
    )
    return parser.parse_args()


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer_with_compat(model_name_or_path: str) -> AutoTokenizer:
    tokenizer_kwargs: dict[str, Any] = {}
    model_name_lower = str(model_name_or_path).lower()
    if "mistral-small" in model_name_lower or "mistral_small" in model_name_lower:
        tokenizer_kwargs["fix_mistral_regex"] = True
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    except TypeError:
        tokenizer_kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_layer_spec(spec: str) -> list[int]:
    selected: list[int] = []
    seen = set()
    for piece in spec.split(","):
        value = piece.strip()
        if not value:
            continue
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError(f"Invalid layer range {value!r}.")
            candidates = range(start, end + 1)
        else:
            candidates = [int(value)]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)
    if not selected:
        raise ValueError("--layers cannot be empty.")
    return selected


def resolve_device(device_arg: str) -> tuple[str | None, torch.device]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return None, torch.device("cuda")
        return None, torch.device("cpu")
    if device_arg == "cuda":
        return None, torch.device("cuda")
    if device_arg == "cpu":
        return None, torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_arg}")


def load_model_and_tokenizer(model_name_or_path: str, device_arg: str):
    _device_map, device = resolve_device(device_arg)
    tokenizer = load_tokenizer_with_compat(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="auto")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def get_transformer_layers(model: AutoModelForCausalLM) -> list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported model architecture: could not locate transformer layers.")


def resolve_saved_to_model_layers(num_model_layers: int) -> tuple[list[int], dict[int, int]]:
    saved_layers = list(range(1, num_model_layers + 1))
    return saved_layers, {saved_layer: saved_layer - 1 for saved_layer in saved_layers}


def build_messages(instruction: str, input_text: str, output_text: str, method: str) -> list[dict[str, str]]:
    assistant_text = output_text.strip()
    if method == "mistral":
        return [
            {"role": "user", "content": instruction.strip() + "\n\n" + input_text.strip()},
            {"role": "assistant", "content": assistant_text},
        ]
    if method == "llama":
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": assistant_text},
        ]
    raise ValueError(f"Unsupported method: {method}")


def render_messages_without_template(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(message["content"].strip() for message in messages if message["content"].strip())


def build_full_sequence_ids(
    tokenizer: AutoTokenizer,
    instruction: str,
    input_text: str,
    output_text: str,
    method: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    messages = build_messages(instruction=instruction, input_text=input_text, output_text=output_text, method=method)
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


def normalize_heading_key(raw_heading: str) -> str:
    compact = raw_heading.lower().replace(" ", "")
    if compact == "finalresponse":
        return "finalresponse"
    return compact


def extract_heading_examples_for_record(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict[str, str],
    record_index: int,
    target_tag: str,
    method: str,
    device: torch.device,
    include_final_response_as_negative: bool,
) -> list[HeadingExample]:
    assistant_text = record["output"].strip()
    full_ids, sequence_ids = build_full_sequence_ids(
        tokenizer=tokenizer,
        instruction=record["instruction"],
        input_text=record["input"],
        output_text=assistant_text,
        method=method,
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

    heading_examples: list[HeadingExample] = []
    label_text = record["input"][len("### Task\n") :].strip() if record["input"].startswith("### Task\n") else record["input"][:120]

    for heading_key, heading_text in FULL_HEADING_TEXT.items():
        if heading_key == "finalresponse" and not include_final_response_as_negative:
            continue
        for char_start, _char_end in find_text_spans(assistant_text, heading_text):
            local_token_index = map_char_position_to_token_index(assistant_offsets, char_start)
            if local_token_index is None:
                continue
            token_position = assistant_start + local_token_index
            label = 1 if heading_key in TOOL_HEADING_TAGS else 0
            layer_states = torch.stack(
                [outputs.hidden_states[layer_idx][0, token_position, :].detach().float().cpu() for layer_idx in range(1, len(outputs.hidden_states))],
                dim=0,
            )
            heading_examples.append(
                HeadingExample(
                    record_index=record_index,
                    record_label=label_text,
                    heading_tag=heading_key,
                    heading_text=heading_text,
                    token_position=int(token_position),
                    label=label,
                    layer_states=layer_states,
                )
            )
    return heading_examples


def compute_binary_metrics(labels: torch.Tensor, probs: torch.Tensor) -> dict[str, float]:
    labels = labels.float()
    preds = (probs >= 0.5).float()
    accuracy = float((preds == labels).float().mean().item())
    tp = float(((preds == 1.0) & (labels == 1.0)).sum().item())
    tn = float(((preds == 0.0) & (labels == 0.0)).sum().item())
    fp = float(((preds == 1.0) & (labels == 0.0)).sum().item())
    fn = float(((preds == 0.0) & (labels == 1.0)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def stratified_split_indices(labels: list[int], train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    positives = [idx for idx, value in enumerate(labels) if value == 1]
    negatives = [idx for idx, value in enumerate(labels) if value == 0]
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


def standardize_features(train_x: torch.Tensor, full_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    normalized = (full_x - mean) / std
    return normalized, mean, std


def train_probe_for_layer(
    layer_id: int,
    layer_features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    positive_class_weight: float,
) -> dict[str, Any]:
    train_x_raw = layer_features[train_idx]
    train_y = labels[train_idx]
    full_x, mean, std = standardize_features(train_x_raw, layer_features)
    train_x = full_x[train_idx]
    val_x = full_x[val_idx] if val_idx else None
    val_y = labels[val_idx] if val_idx else None

    model = LinearProbe(hidden_dim=train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor([positive_class_weight], dtype=train_x.dtype)
    best_state = None
    best_score = -math.inf
    best_metrics = None

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if val_x is not None and len(val_idx) > 0:
                val_probs = torch.sigmoid(model(val_x))
                val_metrics = compute_binary_metrics(val_y, val_probs)
                score = val_metrics["f1"]
            else:
                train_probs = torch.sigmoid(model(train_x))
                val_metrics = compute_binary_metrics(train_y, train_probs)
                score = val_metrics["f1"]
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_metrics = val_metrics

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_probs = torch.sigmoid(model(train_x))
        train_metrics = compute_binary_metrics(train_y, train_probs)
        all_probs = torch.sigmoid(model(full_x))
        all_metrics = compute_binary_metrics(labels, all_probs)
        if val_x is not None and len(val_idx) > 0:
            val_probs = torch.sigmoid(model(val_x))
            val_metrics = compute_binary_metrics(val_y, val_probs)
        else:
            val_metrics = None

    weight = model.linear.weight.detach().cpu().squeeze(0)
    bias = float(model.linear.bias.detach().cpu().item())
    return {
        "layer_id": layer_id,
        "weight": weight,
        "bias": bias,
        "feature_mean": mean,
        "feature_std": std,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "all_metrics": all_metrics,
        "best_selection_metric": best_score,
        "all_probabilities": all_probs.detach().cpu(),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    target_tag, default_data_path = DOMAIN_TO_TARGET[args.domain]
    data_path = args.data_path or default_data_path
    output_dir = args.output_dir / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_json(data_path)
    tokenizer, model, device = load_model_and_tokenizer(str(args.model_name_or_path), args.device)
    num_model_layers = len(get_transformer_layers(model))
    saved_layers, saved_to_model = resolve_saved_to_model_layers(num_model_layers)
    requested_saved_layers = parse_layer_spec(args.layers)
    missing_layers = [layer for layer in requested_saved_layers if layer not in saved_to_model]
    if missing_layers:
        raise ValueError(
            f"Requested layers {missing_layers} are not available for a model with {num_model_layers} transformer layers."
        )

    examples: list[HeadingExample] = []
    for record_index, record in enumerate(records):
        examples.extend(
            extract_heading_examples_for_record(
                model=model,
                tokenizer=tokenizer,
                record=record,
                record_index=record_index,
                target_tag=target_tag,
                method=args.method,
                device=device,
                include_final_response_as_negative=args.include_final_response_as_negative,
            )
        )

    if not examples:
        raise ValueError(f"No heading examples were extracted from {data_path}.")

    labels = [example.label for example in examples]
    if len(set(labels)) < 2:
        raise ValueError("Need both positive and negative heading examples to train the classifier.")

    layer_to_features = {}
    for layer_id in requested_saved_layers:
        model_layer_idx = saved_to_model[layer_id]
        layer_to_features[layer_id] = torch.stack([example.layer_states[model_layer_idx] for example in examples], dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    train_idx, val_idx = stratified_split_indices(labels, args.train_ratio, args.seed)
    results_by_layer = []
    for layer_id in requested_saved_layers:
        result = train_probe_for_layer(
            layer_id=layer_id,
            layer_features=layer_to_features[layer_id],
            labels=labels_tensor,
            train_idx=train_idx,
            val_idx=val_idx,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            positive_class_weight=args.positive_class_weight,
        )
        payload = {
            "domain": args.domain,
            "target_tag": target_tag,
            "classifier_target": "any_tool_heading",
            "layer_id": layer_id,
            "weight": result["weight"],
            "bias": result["bias"],
            "feature_mean": result["feature_mean"],
            "feature_std": result["feature_std"],
            "train_metrics": result["train_metrics"],
            "val_metrics": result["val_metrics"],
            "all_metrics": result["all_metrics"],
            "best_selection_metric": result["best_selection_metric"],
            "train_indices": train_idx,
            "val_indices": val_idx,
            "data_path": str(data_path),
            "model_name_or_path": str(args.model_name_or_path),
            "method": args.method,
            "anchor_definition": "### heading hash token in assistant output",
            "label_definition": "1 if heading tag is one of `search`, `code`, `askuser`; else 0",
        }
        torch.save(payload, output_dir / f"{args.domain}_any_tool_layer{layer_id}_classifier.pt")
        results_by_layer.append(
            {
                "layer_id": layer_id,
                "train_metrics": result["train_metrics"],
                "val_metrics": result["val_metrics"],
                "all_metrics": result["all_metrics"],
                "best_selection_metric": result["best_selection_metric"],
            }
        )

    heading_counts = {}
    for heading_tag in sorted({example.heading_tag for example in examples}):
        heading_counts[heading_tag] = sum(1 for example in examples if example.heading_tag == heading_tag)

    summary = {
        "domain": args.domain,
        "target_tag": target_tag,
        "classifier_target": "any_tool_heading",
        "data_path": str(data_path),
        "model_name_or_path": str(args.model_name_or_path),
        "requested_layers": requested_saved_layers,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "positive_class_weight": args.positive_class_weight,
        "num_examples": len(examples),
        "num_positive_examples": int(sum(labels)),
        "num_negative_examples": int(len(labels) - sum(labels)),
        "heading_counts": heading_counts,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "results_by_layer": results_by_layer,
        "example_preview": [
            {
                "record_index": example.record_index,
                "record_label": example.record_label,
                "heading_tag": example.heading_tag,
                "heading_text": example.heading_text,
                "token_position": example.token_position,
                "label": example.label,
            }
            for example in examples[: args.save_examples_limit]
        ],
    }
    save_json(output_dir / f"{args.domain}_any_tool_classifier_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
