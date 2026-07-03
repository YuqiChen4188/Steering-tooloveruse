#!/usr/bin/env python3
"""Build steering vectors from ### heading token positions in full trajectories."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_schema_utils import (  # noqa: E402
    CODE_HEADINGS,
    SCHEMAS,
    convert_record_for_schema,
    find_json_action_value_spans,
    resolve_code_heading,
    resolve_schema,
)


VECTOR_GROUPS = ("all", "search", "code", "askuser", "search_askuser")
DEFAULT_MODEL_PATH = Path("/data/yuqi/Steering-tooloveruse/model_weights/Llama-3.1-8B-Instruct")
DEFAULT_SEARCH_DATA_PATH = Path("/data/yuqi/SteeringMark/steering_/steering_data_search_20.json")
DEFAULT_CODE_DATA_PATH = Path("/data/yuqi/SteeringMark/steering_/steering_data_code_20.json")
DEFAULT_ASKUSER_DATA_PATH = Path("/data/yuqi/SteeringMark/steering_/steering_data_askuser_20.json")
DEFAULT_OUTPUT_DIR = Path("/data/yuqi/SteeringMark/steering_vector")

DEFAULT_FULL_HEADING_TEXT = {
    "reasoning": "### Reasoning",
    "search": "### Search",
    "code": "### Code",
    "askuser": "### AskUser",
}
TAG_TO_JSON_ACTION = {
    "reasoning": "reasoning",
    "search": "search",
    "code": "code",
    "askuser": "askuser",
}
GROUP_TO_TAG = {
    "search": "search",
    "code": "code",
    "askuser": "askuser",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build steering vectors from the hidden states anchored to ### heading tokens "
            "in full tool trajectories."
        )
    )
    parser.add_argument(
        "--model-name-or-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local model path used to extract hidden states.",
    )
    parser.add_argument(
        "--search-data-path",
        type=Path,
        default=DEFAULT_SEARCH_DATA_PATH,
        help="Search trajectory JSON generated under SteeringMark/steering_.",
    )
    parser.add_argument(
        "--code-data-path",
        type=Path,
        default=DEFAULT_CODE_DATA_PATH,
        help="Code trajectory JSON generated under SteeringMark/steering_.",
    )
    parser.add_argument(
        "--askuser-data-path",
        type=Path,
        default=DEFAULT_ASKUSER_DATA_PATH,
        help="AskUser trajectory JSON generated under SteeringMark/steering_.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where vector payloads and summaries will be written.",
    )
    parser.add_argument(
        "--method",
        choices=("llama", "mistral"),
        default="llama",
        help="Prompt packing style for the target model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device selection. Use "auto", "cuda", or "cpu".',
    )
    parser.add_argument(
        "--tag-token-mode",
        choices=("heading_hash", "heading_text_last", "heading_text_mean"),
        default="heading_hash",
        help=(
            "How to anchor the heading representation. Default uses the token position "
            "corresponding to the leading ### marker of each heading."
        ),
    )
    parser.add_argument(
        "--vector-groups",
        default="all,search,code,askuser,search_askuser",
        help="Comma-separated subset of: all,search,code,askuser,search_askuser",
    )
    parser.add_argument(
        "--schema",
        choices=SCHEMAS,
        default=None,
        help="Alias for --extract-schema. Defaults to markdown.",
    )
    parser.add_argument(
        "--extract-schema",
        choices=SCHEMAS,
        default=None,
        help="Trajectory format used to extract heading/action states. Defaults to --schema or markdown.",
    )
    parser.add_argument(
        "--code-heading",
        choices=CODE_HEADINGS,
        default="Code",
        help="Markdown heading name used for the code tool when --extract-schema markdown.",
    )
    parser.add_argument(
        "--ablation",
        choices=("none", "cross_format", "heading_rename"),
        default="none",
        help="Optional ablation metadata saved into payloads and summaries.",
    )
    return parser.parse_args()


def parse_csv_choices(raw_value: str, allowed: tuple[str, ...], arg_name: str) -> list[str]:
    values = []
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_device(device_arg: str) -> tuple[str | None, torch.device]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            if importlib.util.find_spec("accelerate") is not None:
                return "auto", torch.device("cuda")
            return None, torch.device("cuda")
        return None, torch.device("cpu")
    if device_arg == "cuda":
        return None, torch.device("cuda")
    if device_arg == "cpu":
        return None, torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_arg}")


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


def load_model_and_tokenizer(model_name_or_path: str, device_arg: str):
    device_map, device = resolve_device(device_arg)
    tokenizer = load_tokenizer_with_compat(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"dtype": "auto"}
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if device_map is None:
        model = model.to(device)
    model.eval()
    return tokenizer, model, device


def build_messages(instruction: str, input_text: str, output_text: str, method: str) -> list[dict[str, str]]:
    assistant_text = output_text.strip()
    if method == "mistral":
        return [
            {
                "role": "user",
                "content": instruction.strip() + "\n\n" + input_text.strip(),
            },
            {"role": "assistant", "content": assistant_text},
        ]
    if method == "llama":
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": assistant_text},
        ]
    raise ValueError(f"Unsupported method: {method}")


def coerce_token_ids_to_tensor(encoded: Any) -> torch.Tensor:
    if torch.is_tensor(encoded):
        return encoded
    if hasattr(encoded, "input_ids"):
        return coerce_token_ids_to_tensor(encoded.input_ids)
    if hasattr(encoded, "ids"):
        return torch.tensor([encoded.ids], dtype=torch.long)
    if isinstance(encoded, list):
        if encoded and isinstance(encoded[0], int):
            return torch.tensor([encoded], dtype=torch.long)
        return torch.tensor(encoded, dtype=torch.long)
    raise TypeError(f"Unsupported token id container: {type(encoded).__name__}")


def render_messages_without_template(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        content = message["content"].strip()
        if not content:
            continue
        parts.append(content)
    return "\n\n".join(parts)


def build_full_sequence_ids(
    tokenizer: AutoTokenizer,
    instruction: str,
    input_text: str,
    output_text: str,
    method: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    messages = build_messages(
        instruction=instruction,
        input_text=input_text,
        output_text=output_text,
        method=method,
    )
    if getattr(tokenizer, "chat_template", None):
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        full_ids = coerce_token_ids_to_tensor(full_ids)
    else:
        rendered_text = render_messages_without_template(messages)
        full_ids = coerce_token_ids_to_tensor(tokenizer(rendered_text, add_special_tokens=True, return_tensors="pt"))
    full_ids = full_ids.to(device)
    return full_ids, full_ids[0].detach().cpu().tolist()


def find_subsequence_spans(sequence: list[int], subsequence: list[int]) -> list[tuple[int, int]]:
    spans = []
    if not subsequence:
        return spans
    width = len(subsequence)
    for start in range(0, len(sequence) - width + 1):
        if sequence[start : start + width] == subsequence:
            spans.append((start, start + width))
    return spans


def find_last_subsequence_span(sequence: list[int], subsequence: list[int]) -> tuple[int, int]:
    spans = find_subsequence_spans(sequence, subsequence)
    if not spans:
        raise ValueError("Could not align the assistant output token sequence inside the full prompt sequence.")
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


def map_char_span_to_token_span(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    token_start = None
    token_end = None
    for idx, (token_char_start, token_char_end) in enumerate(offsets):
        if token_char_end <= char_start:
            continue
        if token_char_start >= char_end:
            break
        if token_start is None:
            token_start = idx
        token_end = idx + 1
    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def map_char_position_to_token_index(
    offsets: list[tuple[int, int]],
    char_position: int,
) -> int | None:
    for idx, (token_char_start, token_char_end) in enumerate(offsets):
        if token_char_end <= char_position:
            continue
        if token_char_start > char_position:
            return idx
        return idx
    return None


def reduce_tag_spans_to_states(
    hidden_states: tuple[torch.Tensor, ...],
    spans: list[tuple[int, int]],
    reduction: str,
) -> torch.Tensor:
    per_occurrence = []
    for start, end in spans:
        if reduction == "last":
            token_position = end - 1
            state = torch.stack(
                [hidden_states[layer_idx][0, token_position, :].detach().float().cpu() for layer_idx in range(1, len(hidden_states))],
                dim=0,
            )
        elif reduction == "mean":
            token_positions = list(range(start, end))
            state = torch.stack(
                [
                    hidden_states[layer_idx][0, token_positions, :].detach().float().cpu().mean(dim=0)
                    for layer_idx in range(1, len(hidden_states))
                ],
                dim=0,
            )
        else:
            raise ValueError(f"Unsupported tag-token reduction: {reduction}")
        per_occurrence.append(state)
    return torch.stack(per_occurrence, dim=0).mean(dim=0)


def reduce_anchor_positions_to_states(
    hidden_states: tuple[torch.Tensor, ...],
    positions: list[int],
) -> torch.Tensor:
    per_occurrence = []
    for token_position in positions:
        state = torch.stack(
            [hidden_states[layer_idx][0, token_position, :].detach().float().cpu() for layer_idx in range(1, len(hidden_states))],
            dim=0,
        )
        per_occurrence.append(state)
    return torch.stack(per_occurrence, dim=0).mean(dim=0)


def compute_anchor_positions(
    assistant_offsets: list[tuple[int, int]],
    assistant_start: int,
    full_heading_spans: list[tuple[int, int]],
    anchor_mode: str,
) -> list[int]:
    anchor_positions = []
    for char_start, _char_end in full_heading_spans:
        local_token_index = map_char_position_to_token_index(assistant_offsets, char_start)
        if local_token_index is None:
            continue

        if anchor_mode == "heading_hash":
            anchor_index = assistant_start + local_token_index
        else:
            raise ValueError(f"Unsupported anchor mode: {anchor_mode}")

        anchor_positions.append(anchor_index)
    return anchor_positions


def build_full_heading_text(code_heading: str) -> dict[str, str]:
    heading_text = dict(DEFAULT_FULL_HEADING_TEXT)
    heading_text["code"] = f"### {code_heading}"
    return heading_text


def extract_tag_layer_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict[str, str],
    method: str,
    device: torch.device,
    tag_token_mode: str,
    extract_schema: str,
    code_heading: str,
) -> dict[str, Any]:
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
    if "offset_mapping" not in assistant_encoding:
        raise ValueError("Tokenizer does not provide offset mappings; cannot align heading/action token positions.")
    assistant_ids = assistant_encoding["input_ids"]
    if not assistant_ids:
        raise ValueError("Assistant output tokenization is empty.")
    assistant_offsets = [tuple(item) for item in assistant_encoding["offset_mapping"]]
    assistant_start, assistant_end = find_last_subsequence_span(sequence_ids, assistant_ids)

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True, use_cache=False, return_dict=True)

    tag_states: dict[str, torch.Tensor] = {}
    tag_positions: dict[str, list[list[int]]] = {}
    full_heading_text_by_tag = build_full_heading_text(code_heading)
    for tag_name, full_heading_text in full_heading_text_by_tag.items():
        if extract_schema == "json":
            action_spans = find_json_action_value_spans(assistant_text, TAG_TO_JSON_ACTION[tag_name])
            anchor_positions = compute_anchor_positions(
                assistant_offsets=assistant_offsets,
                assistant_start=assistant_start,
                full_heading_spans=action_spans,
                anchor_mode="heading_hash",
            )
            if not anchor_positions:
                continue
            tag_states[tag_name] = reduce_anchor_positions_to_states(
                hidden_states=outputs.hidden_states,
                positions=anchor_positions,
            )
            tag_positions[tag_name] = [[position, position + 1] for position in anchor_positions]
            tag_positions[f"{tag_name}_anchor_positions"] = anchor_positions
            tag_positions[f"{tag_name}_anchor_mode"] = ["json_action_value_start"]
            tag_positions[f"{tag_name}_full_char_spans"] = [list(span) for span in action_spans]
            continue

        full_heading_spans = find_text_spans(assistant_text, full_heading_text)
        if tag_token_mode == "heading_hash":
            anchor_positions = compute_anchor_positions(
                assistant_offsets=assistant_offsets,
                assistant_start=assistant_start,
                full_heading_spans=full_heading_spans,
                anchor_mode=tag_token_mode,
            )
            if not anchor_positions:
                continue
            tag_states[tag_name] = reduce_anchor_positions_to_states(
                hidden_states=outputs.hidden_states,
                positions=anchor_positions,
            )
            tag_positions[tag_name] = [[position, position + 1] for position in anchor_positions]
            tag_positions[f"{tag_name}_anchor_positions"] = anchor_positions
            tag_positions[f"{tag_name}_anchor_mode"] = [tag_token_mode]
            tag_positions[f"{tag_name}_full_char_spans"] = [list(span) for span in full_heading_spans]
            continue

        tag_label = full_heading_text[len("### ") :]
        spans = []
        for char_start, _char_end in full_heading_spans:
            inner_start = char_start + len("### ")
            inner_end = inner_start + len(tag_label)
            local_span = map_char_span_to_token_span(assistant_offsets, inner_start, inner_end)
            if local_span is None:
                continue
            spans.append((assistant_start + local_span[0], assistant_start + local_span[1]))
        if not spans:
            continue
        tag_states[tag_name] = reduce_tag_spans_to_states(
            hidden_states=outputs.hidden_states,
            spans=spans,
            reduction="last" if tag_token_mode == "heading_text_last" else "mean",
        )
        tag_positions[tag_name] = [list(span) for span in spans]
        tag_positions[f"{tag_name}_full_char_spans"] = [list(span) for span in full_heading_spans]

    return {
        "tag_states": tag_states,
        "tag_positions": tag_positions,
        "sequence_length": int(full_ids.shape[1]),
        "assistant_token_span": [assistant_start, assistant_end],
    }

def summarize_scalar_list(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def summarize_vector_norms(vector: torch.Tensor) -> dict[str, Any]:
    norms = torch.linalg.vector_norm(vector, dim=1).tolist()
    return {
        "layer_norms": [float(value) for value in norms],
        "stats": summarize_scalar_list([float(value) for value in norms]),
    }


def build_record_label(record: dict[str, str]) -> str:
    input_text = record["input"]
    prefix = "### Task\n"
    if input_text.startswith(prefix):
        return input_text[len(prefix) :].strip()
    return input_text[:120].replace("\n", " ")


def compute_sample_delta(
    record: dict[str, str],
    positive_tag: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    method: str,
    tag_token_mode: str,
    extract_schema: str,
    code_heading: str,
    state_cache: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    cache_key = (
        record["instruction"],
        record["input"],
        record["output"],
    )
    if cache_key not in state_cache:
        state_cache[cache_key] = extract_tag_layer_states(
            model=model,
            tokenizer=tokenizer,
            record=record,
            method=method,
            device=device,
            tag_token_mode=tag_token_mode,
            extract_schema=extract_schema,
            code_heading=code_heading,
        )

    extracted = state_cache[cache_key]
    tag_states = extracted["tag_states"]
    if "reasoning" not in tag_states:
        raise ValueError("Record is missing reasoning anchor states.")
    if positive_tag not in tag_states:
        raise ValueError(f"Record is missing {positive_tag!r} anchor states.")

    raw_delta = tag_states[positive_tag] - tag_states["reasoning"]
    sample_info = {
        "label": build_record_label(record),
        "positive_tag": positive_tag,
        "sequence_length": extracted["sequence_length"],
        "assistant_token_span": extracted["assistant_token_span"],
        "tag_positions": extracted["tag_positions"],
        "per_layer_raw_delta_norms": [
            float(value) for value in torch.linalg.vector_norm(raw_delta, dim=1).tolist()
        ],
    }
    return {"raw_delta": raw_delta, "sample_info": sample_info}


def build_group_payload(
    group_name: str,
    positive_tag: str,
    records: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    method: str,
    tag_token_mode: str,
    extract_schema: str,
    code_heading: str,
    ablation: str,
    model_name_or_path: str,
    data_paths: dict[str, str],
    state_cache: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    computed = [
        compute_sample_delta(
            record=record,
            positive_tag=positive_tag,
            tokenizer=tokenizer,
            model=model,
            device=device,
            method=method,
            tag_token_mode=tag_token_mode,
            extract_schema=extract_schema,
            code_heading=code_heading,
            state_cache=state_cache,
        )
        for record in records
    ]
    deltas = [item["raw_delta"] for item in computed]
    delta_tensor = torch.stack(deltas, dim=0)
    steering_vector = delta_tensor.mean(dim=0)
    layer_indices = list(range(1, steering_vector.shape[0] + 1))
    sample_infos = [item["sample_info"] for item in computed]

    payload = {
        "steering_vectors": steering_vector,
        "layer_indices": layer_indices,
        "mean_direction": f"{group_name}_tag_minus_reasoning_tag_raw_mean",
        "sample_count": len(records),
        "per_sample_deltas": delta_tensor,
        "sample_infos": sample_infos,
        "model_name_or_path": model_name_or_path,
        "method": method,
        "tag_token_mode": tag_token_mode,
        "tag_span_mode": {
            "heading_hash": "heading_hash_prefix",
            "heading_text_last": "heading_text_last_token",
            "heading_text_mean": "heading_text_mean_pool",
        }[tag_token_mode] if extract_schema == "markdown" else "json_action_value_start",
        "vector_group": group_name,
        "extract_schema": extract_schema,
        "code_heading": code_heading,
        "ablation": ablation,
        "data_paths": data_paths,
    }
    summary = {
        "model_name_or_path": model_name_or_path,
        "method": method,
        "tag_token_mode": tag_token_mode,
        "vector_group": group_name,
        "sample_count": len(records),
        "layer_indices": layer_indices,
        "mean_direction": payload["mean_direction"],
        "tag_span_mode": payload["tag_span_mode"],
        "extract_schema": extract_schema,
        "code_heading": code_heading,
        "ablation": ablation,
        "steering_vector_norms": summarize_vector_norms(steering_vector),
        "data_paths": data_paths,
        "tag_count_stats": dict(Counter(info["positive_tag"] for info in sample_infos)),
    }
    return payload, summary


def build_merged_group_payload(
    group_name: str,
    source_names: list[str],
    source_payloads: dict[str, dict[str, Any]],
    data_paths: dict[str, str],
    model_name_or_path: str,
    method: str,
    tag_token_mode: str,
    extract_schema: str,
    code_heading: str,
    ablation: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    delta_tensor = torch.cat([source_payloads[source_name]["per_sample_deltas"] for source_name in source_names], dim=0)
    steering_vector = delta_tensor.mean(dim=0)
    layer_indices = list(range(1, steering_vector.shape[0] + 1))
    sample_count = delta_tensor.shape[0]
    sample_infos = []
    for source_name in source_names:
        sample_infos.extend(source_payloads[source_name]["sample_infos"])

    payload = {
        "steering_vectors": steering_vector,
        "layer_indices": layer_indices,
        "mean_direction": f"{group_name}_tag_minus_reasoning_tag_raw_mean",
        "sample_count": int(sample_count),
        "per_sample_deltas": delta_tensor,
        "sample_infos": sample_infos,
        "model_name_or_path": model_name_or_path,
        "method": method,
        "tag_token_mode": tag_token_mode,
        "vector_group": group_name,
        "extract_schema": extract_schema,
        "code_heading": code_heading,
        "ablation": ablation,
        "data_paths": data_paths,
        "source_groups": source_names,
    }
    summary = {
        "model_name_or_path": model_name_or_path,
        "method": method,
        "tag_token_mode": tag_token_mode,
        "vector_group": group_name,
        "sample_count": int(sample_count),
        "layer_indices": layer_indices,
        "mean_direction": payload["mean_direction"],
        "extract_schema": extract_schema,
        "code_heading": code_heading,
        "ablation": ablation,
        "steering_vector_norms": summarize_vector_norms(steering_vector),
        "data_paths": data_paths,
        "source_groups": source_names,
        "source_counts": {name: len(source_payloads[name]["sample_infos"]) for name in source_names},
    }
    return payload, summary


def main() -> int:
    args = parse_args()
    vector_groups = parse_csv_choices(args.vector_groups, VECTOR_GROUPS, "--vector-groups")
    extract_schema = resolve_schema(args.extract_schema, fallback=args.schema or "markdown")
    code_heading = resolve_code_heading(args.code_heading)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_by_group = {
        "search": load_json(args.search_data_path),
        "code": load_json(args.code_data_path),
        "askuser": load_json(args.askuser_data_path),
    }
    for group_name, data in data_by_group.items():
        if not isinstance(data, list) or not data:
            raise ValueError(f"{group_name} data at the provided path must be a non-empty list.")

    source_groups_to_prepare = {
        name
        for name in ("search", "code", "askuser")
        if name in vector_groups or "all" in vector_groups or "search_askuser" in vector_groups
    }
    data_by_group = {
        group_name: [
            convert_record_for_schema(record, schema=extract_schema, code_heading=code_heading)
            for record in records
        ]
        for group_name, records in data_by_group.items()
        if group_name in source_groups_to_prepare
    }

    tokenizer, model, device = load_model_and_tokenizer(str(args.model_name_or_path), args.device)

    state_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    data_paths = {
        "search": str(args.search_data_path),
        "code": str(args.code_data_path),
        "askuser": str(args.askuser_data_path),
    }

    source_payloads: dict[str, dict[str, Any]] = {}
    source_summaries: dict[str, dict[str, Any]] = {}
    for source_name in sorted(source_groups_to_prepare):
        payload, summary = build_group_payload(
            group_name=source_name,
            positive_tag=GROUP_TO_TAG[source_name],
            records=data_by_group[source_name],
            tokenizer=tokenizer,
            model=model,
            device=device,
            method=args.method,
            tag_token_mode=args.tag_token_mode,
            extract_schema=extract_schema,
            code_heading=code_heading,
            ablation=args.ablation,
            model_name_or_path=str(args.model_name_or_path),
            data_paths=data_paths,
            state_cache=state_cache,
        )
        source_payloads[source_name] = payload
        source_summaries[source_name] = summary

    for group_name in vector_groups:
        if group_name == "all":
            payload, summary = build_merged_group_payload(
                group_name="all",
                source_names=["search", "code", "askuser"],
                source_payloads=source_payloads,
                data_paths=data_paths,
                model_name_or_path=str(args.model_name_or_path),
                method=args.method,
                tag_token_mode=args.tag_token_mode,
                extract_schema=extract_schema,
                code_heading=code_heading,
                ablation=args.ablation,
            )
        elif group_name == "search_askuser":
            payload, summary = build_merged_group_payload(
                group_name="search_askuser",
                source_names=["search", "askuser"],
                source_payloads=source_payloads,
                data_paths=data_paths,
                model_name_or_path=str(args.model_name_or_path),
                method=args.method,
                tag_token_mode=args.tag_token_mode,
                extract_schema=extract_schema,
                code_heading=code_heading,
                ablation=args.ablation,
            )
        else:
            payload = source_payloads[group_name]
            summary = source_summaries[group_name]

        payload_path = args.output_dir / f"step_mark_{group_name}.pt"
        summary_path = args.output_dir / f"step_mark_{group_name}_summary.json"
        torch.save(payload, payload_path)
        save_json(summary_path, summary)
        print(f"Saved payload to {payload_path}")
        print(f"Saved summary to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
