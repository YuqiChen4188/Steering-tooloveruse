#!/usr/bin/env python3
"""Build first-tool-decision steering vectors from paired tool/no-tool trajectories."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


TOOL_TYPES = ("askuser", "search", "code")
ALIGNMENT_MODES = ("same_step", "semantic")
VECTOR_GROUPS = ("all", "askuser", "search", "code")

DEFAULT_TOOL_DATA_PATH = Path(
    "/data/yuqi/Steering-tooloveruse/data_train/base_reasoning_raw/all_domain_tool_used_selected_raw.json"
)
DEFAULT_BASE_DATA_PATH = Path(
    "/data/yuqi/Steering-tooloveruse/data_train/base_reasoning_raw/all_domain_base_reasoning_raw.json"
)
DEFAULT_OUTPUT_DIR = Path("/data/yuqi/Steering-tooloveruse/steering_vector/decision_point")

INSTRUCTION_SOURCES = {
    "askuser": Path("/data/yuqi/Steering-tooloveruse/data_inference/domain_intention_tool_prompt.json"),
    "search": Path("/data/yuqi/Steering-tooloveruse/data_inference/domain_time_tool_prompt.json"),
    "code": Path("/data/yuqi/Steering-tooloveruse/data_inference/domain_math_tool_prompt.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build first-tool-decision steering vectors from paired tool-use and no-tool "
            "reasoning trajectories using last-prompt-token hidden states."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Local model path or Hugging Face model id.")
    parser.add_argument(
        "--tool-data-path",
        type=Path,
        default=DEFAULT_TOOL_DATA_PATH,
        help="Merged tool-used-selected raw JSON.",
    )
    parser.add_argument(
        "--base-data-path",
        type=Path,
        default=DEFAULT_BASE_DATA_PATH,
        help="Merged base-reasoning raw JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved steering vectors and summaries.",
    )
    parser.add_argument(
        "--method",
        choices=("llama", "mistral"),
        default="llama",
        help="Prompt packing style matching inference_tool_prompt_steered.py.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device selection. Use "auto", "cuda", or "cpu".',
    )
    parser.add_argument(
        "--alignment-modes",
        default="same_step,semantic",
        help="Comma-separated subset of: same_step,semantic",
    )
    parser.add_argument(
        "--vector-groups",
        default="all,askuser,search,code",
        help="Comma-separated subset of: all,askuser,search,code",
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


def load_model_and_tokenizer(model_name_or_path: str, device_arg: str):
    device_map, device = resolve_device(device_arg)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
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


def build_messages(instruction: str, input_text: str, method: str) -> list[dict[str, str]]:
    if method == "mistral":
        return [{"role": "user", "content": instruction.strip() + "\n\n" + input_text.strip()}]
    if method == "llama":
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ]
    raise ValueError(f"Unsupported method: {method}")


def render_messages_without_template(messages: list[dict[str, str]]) -> str:
    if len(messages) == 1:
        return messages[0]["content"].strip()
    return "\n\n".join(message["content"].strip() for message in messages if message["content"].strip())


def build_prompt_ids(tokenizer: AutoTokenizer, messages: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt_text = render_messages_without_template(messages)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt").input_ids
    return prompt_ids.to(device)


def extract_last_prompt_layer_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    prompt_ids = build_prompt_ids(tokenizer, messages, device)
    with torch.no_grad():
        outputs = model(prompt_ids, output_hidden_states=True, use_cache=False, return_dict=True)

    hidden_states = outputs.hidden_states
    last_prompt_position = int(prompt_ids.shape[1] - 1)
    layer_states = torch.stack(
        [hidden_states[layer_idx][0, last_prompt_position, :].detach().float().cpu() for layer_idx in range(1, len(hidden_states))],
        dim=0,
    )
    return layer_states, {
        "prompt_length": int(prompt_ids.shape[1]),
        "last_prompt_position": last_prompt_position,
    }


def extract_problem_text(record: dict[str, Any]) -> str:
    data = record.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Record data field is not a dict.")
    problem = data.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Record is missing a non-empty data.problem field.")
    return problem.strip()


def is_final_response_step(step: dict[str, Any]) -> bool:
    return str(step.get("name", "")).strip().lower() == "final response"


def find_first_tool_step(steps: list[dict[str, Any]]) -> tuple[str, int]:
    for idx, step in enumerate(steps):
        step_type = str(step.get("type", "")).strip().lower()
        if step_type in TOOL_TYPES:
            return step_type, idx
    raise ValueError("Tool trajectory does not contain any askuser/search/code step.")


def collect_positive_prefix_steps(steps: list[dict[str, Any]], first_tool_index: int) -> list[str]:
    prefix_steps = []
    for step in steps[:first_tool_index]:
        if str(step.get("type", "")).strip().lower() != "normal":
            continue
        if is_final_response_step(step):
            continue
        reasoning = step.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            prefix_steps.append(reasoning.strip())
    return prefix_steps


def collect_negative_reasoning_steps(steps: list[dict[str, Any]]) -> list[str]:
    prefix_steps = []
    for step in steps:
        if str(step.get("type", "")).strip().lower() != "normal":
            continue
        if is_final_response_step(step):
            continue
        reasoning = step.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            prefix_steps.append(reasoning.strip())
    return prefix_steps


def build_reasoning_input(problem: str, prefix_steps: list[str]) -> str:
    prompt = f"### Task\n{problem.strip()}\n\n### Reasoning Steps\n"
    if prefix_steps:
        prompt += "\n".join(prefix_steps).strip()
    return prompt


def load_instruction_templates() -> dict[str, str]:
    templates = {}
    for tool_family, path in INSTRUCTION_SOURCES.items():
        data = load_json(path)
        if not isinstance(data, list) or not data:
            raise ValueError(f"Instruction source at {path} is not a non-empty list.")
        instruction = data[0].get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Instruction source at {path} is missing a non-empty instruction.")
        templates[tool_family] = instruction
    return templates


def build_validated_samples(
    tool_data: dict[str, Any],
    base_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tool_keys = set(tool_data.keys())
    base_keys = set(base_data.keys())
    if tool_keys != base_keys:
        only_in_tool = sorted(tool_keys - base_keys)
        only_in_base = sorted(base_keys - tool_keys)
        raise ValueError(
            "Input datasets do not share the same task ids. "
            f"only_in_tool={only_in_tool[:5]}, only_in_base={only_in_base[:5]}"
        )

    samples = []
    family_counter = Counter()
    skipped_missing_negative_reasoning_ids = []
    for task_id in sorted(tool_keys):
        tool_record = tool_data[task_id]
        base_record = base_data[task_id]

        tool_problem = extract_problem_text(tool_record)
        base_problem = extract_problem_text(base_record)
        if tool_problem != base_problem:
            raise ValueError(f"Problem mismatch for task_id={task_id}")

        tool_steps = tool_record.get("reasoning_complete")
        base_steps = base_record.get("reasoning_complete")
        if not isinstance(tool_steps, list) or not isinstance(base_steps, list):
            raise ValueError(f"reasoning_complete must be a list for task_id={task_id}")

        first_tool_type, first_tool_index = find_first_tool_step(tool_steps)
        positive_prefix_steps = collect_positive_prefix_steps(tool_steps, first_tool_index)
        negative_reasoning_steps = collect_negative_reasoning_steps(base_steps)
        if not negative_reasoning_steps:
            skipped_missing_negative_reasoning_ids.append(task_id)
            continue

        family_counter[first_tool_type] += 1
        samples.append(
            {
                "task_id": task_id,
                "problem": tool_problem,
                "tool_family": first_tool_type,
                "first_tool_type": first_tool_type,
                "first_tool_step_index": first_tool_index,
                "positive_prefix_steps": positive_prefix_steps,
                "negative_reasoning_steps": negative_reasoning_steps,
            }
        )

    validation_summary = {
        "sample_count": len(samples),
        "tool_family_counts": {family: family_counter.get(family, 0) for family in TOOL_TYPES},
        "all_count": len(samples),
        "skipped_missing_negative_reasoning_count": len(skipped_missing_negative_reasoning_ids),
        "skipped_missing_negative_reasoning_ids": skipped_missing_negative_reasoning_ids,
    }
    return samples, validation_summary


def build_sample_prompt_messages(
    instruction: str,
    problem: str,
    prefix_steps: list[str],
    method: str,
) -> list[dict[str, str]]:
    input_text = build_reasoning_input(problem=problem, prefix_steps=prefix_steps)
    return build_messages(instruction=instruction, input_text=input_text, method=method)


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


def get_prefix_states(
    prefix_steps: list[str],
    prefix_count: int,
    instruction: str,
    problem: str,
    method: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if prefix_count not in cache:
        selected_steps = prefix_steps[:prefix_count]
        messages = build_sample_prompt_messages(
            instruction=instruction,
            problem=problem,
            prefix_steps=selected_steps,
            method=method,
        )
        states, info = extract_last_prompt_layer_states(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
        )
        cache[prefix_count] = {
            "states": states,
            "prompt_length": info["prompt_length"],
            "last_prompt_position": info["last_prompt_position"],
        }
    return cache[prefix_count]


def choose_semantic_prefix(
    positive_last_layer_state: torch.Tensor,
    negative_prefix_steps: list[str],
    instruction: str,
    problem: str,
    method: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    cache: dict[int, dict[str, Any]],
) -> tuple[int, float]:
    best_prefix_count = -1
    best_cosine = float("-inf")
    # Include the empty-prefix candidate so samples with no explicit non-final
    # base reasoning can still align against the task-only prompt.
    for prefix_count in range(0, len(negative_prefix_steps) + 1):
        candidate = get_prefix_states(
            prefix_steps=negative_prefix_steps,
            prefix_count=prefix_count,
            instruction=instruction,
            problem=problem,
            method=method,
            tokenizer=tokenizer,
            model=model,
            device=device,
            cache=cache,
        )
        cosine = float(
            F.cosine_similarity(
                positive_last_layer_state.unsqueeze(0),
                candidate["states"][-1].unsqueeze(0),
                dim=1,
            ).item()
        )
        if cosine > best_cosine:
            best_cosine = cosine
            best_prefix_count = prefix_count

    if best_prefix_count < 0:
        raise RuntimeError("Failed to select a semantic alignment prefix.")
    return best_prefix_count, best_cosine


def build_alignment_result(
    alignment_mode: str,
    sample: dict[str, Any],
    instruction: str,
    method: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    positive_cache: dict[int, dict[str, Any]],
    negative_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    positive_prefix_steps = sample["positive_prefix_steps"]
    negative_prefix_steps = sample["negative_reasoning_steps"]
    positive_prefix_count = len(positive_prefix_steps)

    positive = get_prefix_states(
        prefix_steps=positive_prefix_steps,
        prefix_count=positive_prefix_count,
        instruction=instruction,
        problem=sample["problem"],
        method=method,
        tokenizer=tokenizer,
        model=model,
        device=device,
        cache=positive_cache,
    )

    semantic_alignment_cosine = None
    truncated_same_step = False
    if alignment_mode == "same_step":
        negative_prefix_count = min(positive_prefix_count, len(negative_prefix_steps))
        truncated_same_step = negative_prefix_count < positive_prefix_count
    elif alignment_mode == "semantic":
        negative_prefix_count, semantic_alignment_cosine = choose_semantic_prefix(
            positive_last_layer_state=positive["states"][-1],
            negative_prefix_steps=negative_prefix_steps,
            instruction=instruction,
            problem=sample["problem"],
            method=method,
            tokenizer=tokenizer,
            model=model,
            device=device,
            cache=negative_cache,
        )
    else:
        raise ValueError(f"Unsupported alignment mode: {alignment_mode}")

    negative = get_prefix_states(
        prefix_steps=negative_prefix_steps,
        prefix_count=negative_prefix_count,
        instruction=instruction,
        problem=sample["problem"],
        method=method,
        tokenizer=tokenizer,
        model=model,
        device=device,
        cache=negative_cache,
    )

    raw_delta = positive["states"] - negative["states"]
    layer_delta_norms = torch.linalg.vector_norm(raw_delta, dim=1)
    sample_info = {
        "task_id": sample["task_id"],
        "problem": sample["problem"],
        "tool_family": sample["tool_family"],
        "alignment_mode": alignment_mode,
        "first_tool_type": sample["first_tool_type"],
        "first_tool_step_index": sample["first_tool_step_index"],
        "positive_prefix_normal_steps": positive_prefix_count,
        "negative_prefix_normal_steps": negative_prefix_count,
        "truncated_same_step": truncated_same_step,
        "semantic_alignment_cosine": semantic_alignment_cosine,
        "positive_prompt_length": positive["prompt_length"],
        "negative_prompt_length": negative["prompt_length"],
        "positive_last_prompt_position": positive["last_prompt_position"],
        "negative_last_prompt_position": negative["last_prompt_position"],
        "per_layer_raw_delta_norms": [float(value) for value in layer_delta_norms.tolist()],
    }
    return {
        "raw_delta": raw_delta,
        "sample_info": sample_info,
    }


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    alignment_modes = parse_csv_choices(args.alignment_modes, ALIGNMENT_MODES, "--alignment-modes")
    vector_groups = parse_csv_choices(args.vector_groups, VECTOR_GROUPS, "--vector-groups")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading instruction templates...")
    instruction_templates = load_instruction_templates()

    print("Loading paired datasets...")
    tool_data = load_json(args.tool_data_path)
    base_data = load_json(args.base_data_path)
    if not isinstance(tool_data, dict) or not isinstance(base_data, dict):
        raise ValueError("Input datasets must be JSON objects keyed by task id.")

    samples, validation_summary = build_validated_samples(tool_data=tool_data, base_data=base_data)
    print("Validated paired samples:", validation_summary)

    print("Loading model and tokenizer...")
    tokenizer, model, device = load_model_and_tokenizer(args.model_name_or_path, args.device)

    print("Extracting decision-point deltas...")
    computed_samples: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples, start=1):
        tool_family = sample["tool_family"]
        instruction = instruction_templates[tool_family]
        positive_cache: dict[int, dict[str, Any]] = {}
        negative_cache: dict[int, dict[str, Any]] = {}
        sample_result = {
            "task_id": sample["task_id"],
            "tool_family": tool_family,
            "alignments": {},
        }
        for alignment_mode in alignment_modes:
            sample_result["alignments"][alignment_mode] = build_alignment_result(
                alignment_mode=alignment_mode,
                sample=sample,
                instruction=instruction,
                method=args.method,
                tokenizer=tokenizer,
                model=model,
                device=device,
                positive_cache=positive_cache,
                negative_cache=negative_cache,
            )
        computed_samples.append(sample_result)
        if sample_idx % 10 == 0 or sample_idx == len(samples):
            print(f"Processed {sample_idx}/{len(samples)} samples")

    for alignment_mode in alignment_modes:
        for vector_group in vector_groups:
            group_samples = [
                sample for sample in computed_samples if vector_group == "all" or sample["tool_family"] == vector_group
            ]
            if not group_samples:
                raise ValueError(f"No samples found for vector group {vector_group!r}.")

            deltas = [sample["alignments"][alignment_mode]["raw_delta"] for sample in group_samples]
            delta_tensor = torch.stack(deltas, dim=0)
            steering_vector = delta_tensor.mean(dim=0)
            layer_indices = list(range(1, steering_vector.shape[0] + 1))
            sample_infos = [sample["alignments"][alignment_mode]["sample_info"] for sample in group_samples]

            payload_path = args.output_dir / f"decision_point_{alignment_mode}_{vector_group}.pt"
            summary_path = args.output_dir / f"decision_point_{alignment_mode}_{vector_group}_summary.json"

            payload = {
                "steering_vectors": steering_vector,
                "layer_indices": layer_indices,
                "mean_direction": f"first_tool_decision_minus_no_tool_{alignment_mode}_raw_mean",
                "sample_count": len(group_samples),
                "per_sample_deltas": delta_tensor,
                "sample_infos": sample_infos,
                "model_name_or_path": args.model_name_or_path,
                "method": args.method,
                "alignment_mode": alignment_mode,
                "vector_group": vector_group,
                "tool_data_path": str(args.tool_data_path),
                "base_data_path": str(args.base_data_path),
            }
            torch.save(payload, payload_path)

            tool_family_counts = Counter(info["tool_family"] for info in sample_infos)
            semantic_cosines = [
                float(info["semantic_alignment_cosine"])
                for info in sample_infos
                if info["semantic_alignment_cosine"] is not None
            ]
            summary = {
                "model_name_or_path": args.model_name_or_path,
                "method": args.method,
                "alignment_mode": alignment_mode,
                "vector_group": vector_group,
                "sample_count": len(group_samples),
                "tool_family_counts": {family: tool_family_counts.get(family, 0) for family in TOOL_TYPES},
                "validation_summary": validation_summary,
                "layer_indices": layer_indices,
                "mean_direction": payload["mean_direction"],
                "truncated_same_step_count": sum(1 for info in sample_infos if info["truncated_same_step"]),
                "semantic_alignment_cosine_stats": summarize_scalar_list(semantic_cosines),
                "steering_vector_norms": summarize_vector_norms(steering_vector),
                "payload_path": str(payload_path),
                "tool_data_path": str(args.tool_data_path),
                "base_data_path": str(args.base_data_path),
                "instruction_sources": {family: str(path) for family, path in INSTRUCTION_SOURCES.items()},
            }
            save_json(summary_path, summary)
            print(f"Saved {alignment_mode}/{vector_group} payload to: {payload_path}")
            print(f"Saved {alignment_mode}/{vector_group} summary to: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
