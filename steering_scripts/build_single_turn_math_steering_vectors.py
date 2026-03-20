#!/usr/bin/env python3
"""Build prompt-level steering vectors from paired single-turn math datasets."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CODE_PAIR_PATH = Path(
    "/data/yuqi/Steering-tooloveruse/steering_data/math_single_turn_gpt/domain_math_code_tool_pairs.json"
)
DEFAULT_NO_TOOL_PAIR_PATH = Path(
    "/data/yuqi/Steering-tooloveruse/steering_data/math_single_turn_gpt/domain_math_no_tool_pairs.json"
)
DEFAULT_CODE_RAW_PATH = Path(
    "/data/yuqi/Steering-tooloveruse/steering_data/math_single_turn_gpt/domain_math_code_tool_raw.json"
)
DEFAULT_NO_TOOL_RAW_PATH = Path(
    "/data/yuqi/Steering-tooloveruse/steering_data/math_single_turn_gpt/domain_math_no_tool_raw.json"
)
DEFAULT_OUTPUT_DIR = Path("/data/yuqi/Steering-tooloveruse/steering_vector/math_single_turn_prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build steering vectors from paired single-turn math prompt variants "
            "using last-prompt-token hidden states."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Local model path or Hugging Face model id.")
    parser.add_argument(
        "--code-pair-path",
        type=Path,
        default=DEFAULT_CODE_PAIR_PATH,
        help="Single-turn code-tool pair JSON.",
    )
    parser.add_argument(
        "--no-tool-pair-path",
        type=Path,
        default=DEFAULT_NO_TOOL_PAIR_PATH,
        help="Single-turn no-tool pair JSON.",
    )
    parser.add_argument(
        "--code-raw-path",
        type=Path,
        default=DEFAULT_CODE_RAW_PATH,
        help="Single-turn code-tool raw JSON keyed by task id.",
    )
    parser.add_argument(
        "--no-tool-raw-path",
        type=Path,
        default=DEFAULT_NO_TOOL_RAW_PATH,
        help="Single-turn no-tool raw JSON keyed by task id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved steering vectors and summary metadata.",
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
        "--normalize-deltas",
        action="store_true",
        help="If set, L2-normalize each sample delta independently before averaging.",
    )
    return parser.parse_args()


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


def extract_unique_instruction(pair_data: list[dict[str, Any]], label: str) -> str:
    if not isinstance(pair_data, list) or not pair_data:
        raise ValueError(f"{label} pair data must be a non-empty list.")
    instructions = {item.get("instruction", "").strip() for item in pair_data}
    instructions.discard("")
    if len(instructions) != 1:
        raise ValueError(f"{label} pair data must contain exactly one shared instruction.")
    return next(iter(instructions))


def build_input_text(problem: str) -> str:
    return f"### Task\n{problem.strip()}\n\n### Reasoning Steps\n"


def build_validated_samples(
    code_raw: dict[str, Any],
    no_tool_raw: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(code_raw, dict) or not isinstance(no_tool_raw, dict):
        raise ValueError("Raw datasets must be JSON objects keyed by task id.")

    code_keys = set(code_raw.keys())
    no_tool_keys = set(no_tool_raw.keys())
    shared_ids = [task_id for task_id in no_tool_raw.keys() if task_id in code_keys]

    samples = []
    skipped_problem_mismatch_ids = []
    for task_id in shared_ids:
        code_record = code_raw[task_id]
        no_tool_record = no_tool_raw[task_id]
        code_problem = extract_problem_text(code_record)
        no_tool_problem = extract_problem_text(no_tool_record)
        if code_problem != no_tool_problem:
            skipped_problem_mismatch_ids.append(task_id)
            continue

        samples.append(
            {
                "task_id": task_id,
                "problem": code_problem,
                "code_tool_involved": bool(code_record.get("tool_involved")),
                "no_tool_involved": bool(no_tool_record.get("tool_involved")),
            }
        )

    validation_summary = {
        "sample_count": len(samples),
        "shared_task_count": len(shared_ids),
        "code_only_count": len(code_keys - no_tool_keys),
        "no_tool_only_count": len(no_tool_keys - code_keys),
        "code_only_ids": sorted(code_keys - no_tool_keys),
        "no_tool_only_ids": sorted(no_tool_keys - code_keys),
        "skipped_problem_mismatch_count": len(skipped_problem_mismatch_ids),
        "skipped_problem_mismatch_ids": skipped_problem_mismatch_ids,
    }
    return samples, validation_summary


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


def maybe_normalize_delta(delta: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return delta
    norms = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1e-12)
    return delta / norms


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading single-turn datasets...")
    code_pair_data = load_json(args.code_pair_path)
    no_tool_pair_data = load_json(args.no_tool_pair_path)
    code_raw = load_json(args.code_raw_path)
    no_tool_raw = load_json(args.no_tool_raw_path)

    code_instruction = extract_unique_instruction(code_pair_data, "code-tool")
    no_tool_instruction = extract_unique_instruction(no_tool_pair_data, "no-tool")
    samples, validation_summary = build_validated_samples(code_raw=code_raw, no_tool_raw=no_tool_raw)
    print("Validated paired samples:", validation_summary)

    print("Loading model and tokenizer...")
    tokenizer, model, device = load_model_and_tokenizer(args.model_name_or_path, args.device)

    print("Extracting prompt-state deltas...")
    per_sample_deltas = []
    sample_infos = []
    for sample_idx, sample in enumerate(samples, start=1):
        input_text = build_input_text(sample["problem"])
        code_messages = build_messages(code_instruction, input_text, args.method)
        no_tool_messages = build_messages(no_tool_instruction, input_text, args.method)

        code_states, code_info = extract_last_prompt_layer_states(
            model=model,
            tokenizer=tokenizer,
            messages=code_messages,
            device=device,
        )
        no_tool_states, no_tool_info = extract_last_prompt_layer_states(
            model=model,
            tokenizer=tokenizer,
            messages=no_tool_messages,
            device=device,
        )

        raw_delta = code_states - no_tool_states
        steering_delta = maybe_normalize_delta(raw_delta, normalize=args.normalize_deltas)
        per_sample_deltas.append(steering_delta)

        last_layer_cosine = float(
            F.cosine_similarity(
                code_states[-1].unsqueeze(0),
                no_tool_states[-1].unsqueeze(0),
                dim=1,
            ).item()
        )
        sample_infos.append(
            {
                "task_id": sample["task_id"],
                "problem": sample["problem"],
                "code_prompt_length": code_info["prompt_length"],
                "no_tool_prompt_length": no_tool_info["prompt_length"],
                "code_last_prompt_position": code_info["last_prompt_position"],
                "no_tool_last_prompt_position": no_tool_info["last_prompt_position"],
                "last_layer_prompt_cosine": last_layer_cosine,
                "per_layer_raw_delta_norms": [
                    float(value)
                    for value in torch.linalg.vector_norm(raw_delta, dim=1).tolist()
                ],
                "code_tool_involved": sample["code_tool_involved"],
                "no_tool_involved": sample["no_tool_involved"],
            }
        )

        if sample_idx % 10 == 0 or sample_idx == len(samples):
            print(f"Processed {sample_idx}/{len(samples)} samples")

    if not per_sample_deltas:
        raise ValueError("No shared paired samples available to build steering vectors.")

    delta_tensor = torch.stack(per_sample_deltas, dim=0)
    steering_vector = delta_tensor.mean(dim=0)
    layer_indices = list(range(1, steering_vector.shape[0] + 1))

    suffix = "normalized" if args.normalize_deltas else "raw"
    payload_path = args.output_dir / f"single_turn_math_code_minus_no_tool_{suffix}.pt"
    summary_path = args.output_dir / f"single_turn_math_code_minus_no_tool_{suffix}_summary.json"

    payload = {
        "steering_vectors": steering_vector,
        "layer_indices": layer_indices,
        "mean_direction": f"single_turn_code_prompt_minus_no_tool_prompt_{suffix}_mean",
        "sample_count": len(samples),
        "per_sample_deltas": delta_tensor,
        "sample_infos": sample_infos,
        "model_name_or_path": args.model_name_or_path,
        "method": args.method,
        "code_pair_path": str(args.code_pair_path),
        "no_tool_pair_path": str(args.no_tool_pair_path),
        "code_raw_path": str(args.code_raw_path),
        "no_tool_raw_path": str(args.no_tool_raw_path),
        "code_instruction": code_instruction,
        "no_tool_instruction": no_tool_instruction,
        "normalize_deltas": args.normalize_deltas,
    }
    torch.save(payload, payload_path)

    prompt_cosines = [float(info["last_layer_prompt_cosine"]) for info in sample_infos]
    code_prompt_lengths = [float(info["code_prompt_length"]) for info in sample_infos]
    no_tool_prompt_lengths = [float(info["no_tool_prompt_length"]) for info in sample_infos]
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "method": args.method,
        "sample_count": len(samples),
        "mean_direction": payload["mean_direction"],
        "normalize_deltas": args.normalize_deltas,
        "layer_indices": layer_indices,
        "validation_summary": validation_summary,
        "last_layer_prompt_cosine_stats": summarize_scalar_list(prompt_cosines),
        "code_prompt_length_stats": summarize_scalar_list(code_prompt_lengths),
        "no_tool_prompt_length_stats": summarize_scalar_list(no_tool_prompt_lengths),
        "steering_vector_norms": summarize_vector_norms(steering_vector),
        "payload_path": str(payload_path),
        "code_pair_path": str(args.code_pair_path),
        "no_tool_pair_path": str(args.no_tool_pair_path),
        "code_raw_path": str(args.code_raw_path),
        "no_tool_raw_path": str(args.no_tool_raw_path),
        "code_instruction_preview": code_instruction[:500],
        "no_tool_instruction_preview": no_tool_instruction[:500],
    }
    save_json(summary_path, summary)

    print(f"Saved payload to: {payload_path}")
    print(f"Saved summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
