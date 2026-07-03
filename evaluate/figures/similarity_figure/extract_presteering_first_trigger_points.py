#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


INFERENCE_DIR = Path("/data/yuqi/SteeringMark/inference")
if str(INFERENCE_DIR) not in sys.path:
    sys.path.append(str(INFERENCE_DIR))

import inference_tool_prompt_tag_suppressed_kvcache as base  # noqa: E402


TOOL_HEADINGS = ("### Code", "### Search", "### AskUser")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay existing steering result files without applying steering and extract "
            "the pre-steering cosine at the first heading-trigger point of each raw block."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("DATASET", "RESULT_JSON"),
        required=True,
        help="Dataset label and an existing inference result JSON file. Repeat for multiple domains.",
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Local model path used to replay the saved generations.",
    )
    parser.add_argument(
        "--steering-vector-dir",
        type=Path,
        required=True,
        help="Directory containing step_mark_search/code/askuser.pt for the target model.",
    )
    parser.add_argument(
        "--steering-layer",
        required=True,
        help="Saved steering layer id to replay, e.g. '20'.",
    )
    parser.add_argument(
        "--method",
        choices=("llama", "mistral"),
        required=True,
        help="Prompt packing style used by the original results.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device selection. Use "auto", "cuda", or "cpu".',
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Prompt truncation length used when replaying prompts.",
    )
    parser.add_argument(
        "--max-records-per-input",
        type=int,
        default=None,
        help="Optional cap per input file for debugging.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix without extension.",
    )
    return parser.parse_args()


def classify_raw_block(raw_text: str) -> str | None:
    stripped = raw_text.lstrip()
    if stripped.startswith("### Final Response"):
        return None
    return "tool" if any(heading in raw_text for heading in TOOL_HEADINGS) else "reasoning"


def build_pre_trigger_info(
    generated_text: str,
    hidden_states: tuple[torch.Tensor, ...],
    selected_pairs: list[tuple[int, int]],
    steering_vectors: torch.Tensor,
    layer_indices: list[int],
) -> dict[str, Any]:
    per_layer = []
    for model_layer_idx, vector_idx in selected_pairs:
        hidden_state = hidden_states[model_layer_idx + 1][0, -1, :].detach().float().cpu()
        steering_vector = steering_vectors[vector_idx].detach().float().cpu()
        cosine = F.cosine_similarity(hidden_state.unsqueeze(0), steering_vector.unsqueeze(0), dim=1).item()
        per_layer.append(
            {
                "model_layer": int(model_layer_idx),
                "saved_layer": int(layer_indices[vector_idx]),
                "cosine_similarity": float(cosine),
            }
        )
    mean_cosine = sum(item["cosine_similarity"] for item in per_layer) / len(per_layer) if per_layer else None
    return {
        "context_tail_before_trigger": generated_text[-160:],
        "mean_selected_layer_cosine": float(mean_cosine) if mean_cosine is not None else None,
        "per_layer_cosine": per_layer,
    }


def replay_first_pre_trigger(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    raw_text: str,
    selected_pairs: list[tuple[int, int]],
    steering_vectors: torch.Tensor,
    layer_indices: list[int],
) -> dict[str, Any] | None:
    output_ids = tokenizer(raw_text, add_special_tokens=False, return_tensors="pt").input_ids.to(prompt_ids.device)
    generated_text = ""
    current_input_ids = prompt_ids
    attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
    past_key_values = None

    for token_idx in range(output_ids.shape[1]):
        should_consider_heading = base.has_open_heading_prefix(generated_text)
        with torch.inference_mode():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=should_consider_heading,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values

        if should_consider_heading and outputs.hidden_states is not None:
            return build_pre_trigger_info(
                generated_text=generated_text,
                hidden_states=outputs.hidden_states,
                selected_pairs=selected_pairs,
                steering_vectors=steering_vectors,
                layer_indices=layer_indices,
            )

        next_token = output_ids[:, token_idx : token_idx + 1]
        generated_text += tokenizer.decode(next_token[0], skip_special_tokens=False)
        current_input_ids = next_token
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )

    return None


def trim_round_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_tool_idx = next((idx for idx, step in enumerate(steps) if step.get("type") == "tool"), None)
    if first_tool_idx is not None and first_tool_idx + 1 < len(steps):
        return steps[: first_tool_idx + 1]
    return steps


def build_domain_start_messages(task: str, domain: str, method: str) -> list[dict[str, str]]:
    instruction = base.build_domain_instruction(domain)
    return base.build_messages(
        instruction=instruction,
        input_text=base.build_steering_aligned_input(task),
        method=method,
    )


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_points_csv(path: Path, points: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "input_path",
        "example_index",
        "raw_index",
        "kind",
        "similarity",
        "tool_tags",
        "task",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(point)


def main() -> None:
    args = parse_args()

    print("Loading model and tokenizer for pre-steering replay...")
    tokenizer, model, device = base.load_model_and_tokenizer(args.model_name_or_path, args.device)
    transformer_layer_count = len(base.get_transformer_layers(model))

    domain_cache: dict[str, tuple[torch.Tensor, list[int], list[tuple[int, int]], list[int]]] = {}
    merged_records: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for dataset_label, result_path_str in args.input:
        result_path = Path(result_path_str)
        domain = base.infer_domain(result_path, None)
        tool_group = base.DOMAIN_TO_TOOL_GROUP[domain]

        if domain not in domain_cache:
            steering_vectors, layer_indices = base.load_steering_payload(args.steering_vector_dir / f"step_mark_{tool_group}.pt")
            layer_map = base.resolve_layer_map(layer_indices, transformer_layer_count)
            selection_summary = base.select_pairs_for_explicit_saved_layers(
                layer_indices=layer_indices,
                layer_map=layer_map,
                steering_layer_spec=args.steering_layer,
            )
            domain_cache[domain] = (
                steering_vectors,
                layer_indices,
                selection_summary["selected_pairs"],
                selection_summary["selected_layers"],
            )

        steering_vectors, layer_indices, selected_pairs, selected_layers = domain_cache[domain]

        print(f"Replaying {result_path} with domain={domain}, tool_group={tool_group}, selected_saved_layer={args.steering_layer}")
        records = json.loads(result_path.read_text())
        if args.max_records_per_input is not None:
            records = records[: args.max_records_per_input]

        for example_index, record in enumerate(records):
            input_messages = build_domain_start_messages(record["task"], domain, args.method)
            predict_steps = record.get("predict", [])
            pre_diagnostics: list[dict[str, Any]] = []
            predict_ptr = 0

            for raw_index, raw_text in enumerate(record.get("raw", []), start=1):
                prompt_ids = base.build_prompt_ids(tokenizer, input_messages, device, args.max_seq_length)
                pre_trigger_info = replay_first_pre_trigger(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    raw_text=raw_text,
                    selected_pairs=selected_pairs,
                    steering_vectors=steering_vectors,
                    layer_indices=layer_indices,
                )
                pre_diagnostics.append(
                    {
                        "step": raw_index,
                        "gating_mode": "presteering_replay",
                        "domain": domain,
                        "tool_group": tool_group,
                        "selected_layers": selected_layers,
                        "selected_saved_layers": [int(args.steering_layer)],
                        "token_diagnostics": None,
                        "token_diagnostics_summary": {
                            "first_trigger_info": pre_trigger_info,
                            "max_mean_selected_layer_cosine": (
                                pre_trigger_info["mean_selected_layer_cosine"] if pre_trigger_info is not None else None
                            ),
                        },
                    }
                )

                kind = classify_raw_block(raw_text)
                if kind is not None and pre_trigger_info is not None:
                    point_rows.append(
                        {
                            "dataset": dataset_label,
                            "input_path": str(result_path),
                            "example_index": example_index,
                            "raw_index": raw_index,
                            "kind": kind,
                            "similarity": pre_trigger_info["mean_selected_layer_cosine"],
                            "tool_tags": "|".join(
                                heading.replace("### ", "") for heading in TOOL_HEADINGS if heading in raw_text
                            ),
                            "task": record["task"].replace("\n", " "),
                        }
                    )

                parsed_steps = base.canonicalize_steps(
                    base.normalize_round_steps(raw_text),
                    start_step_number=predict_ptr + 1,
                )
                parsed_steps = trim_round_steps(parsed_steps)
                stored_steps = deepcopy(predict_steps[predict_ptr : predict_ptr + len(parsed_steps)])
                predict_ptr += len(parsed_steps)
                if stored_steps and stored_steps[-1]["name"] != "Final Response":
                    input_messages[-1]["content"] = (
                        input_messages[-1]["content"].strip()
                        + "\n"
                        + base.format_steps(stored_steps).strip()
                        + base.build_continue_prompt(predict_ptr + 1)
                    )

            merged_records.append(
                {
                    "task": record["task"],
                    "predict": predict_steps,
                    "ground_truth": record.get("ground_truth", ""),
                    "raw": record.get("raw", []),
                    "steering_diagnostics": pre_diagnostics,
                    "selected_steering_layers": selected_layers,
                }
            )

    output_json = args.output_prefix.with_suffix(".json")
    output_csv = args.output_prefix.with_suffix(".points.csv")
    output_summary = args.output_prefix.with_suffix(".summary.json")

    save_json(output_json, merged_records)
    save_points_csv(output_csv, point_rows)
    save_json(
        output_summary,
        {
            "num_examples": len(merged_records),
            "num_points": len(point_rows),
            "inputs": {dataset: path for dataset, path in args.input},
            "model_name_or_path": args.model_name_or_path,
            "steering_vector_dir": str(args.steering_vector_dir),
            "steering_layer": args.steering_layer,
            "method": args.method,
            "note": (
                "Each point uses the cosine between the saved steering vector and the hidden state "
                "at the first heading-trigger point, replayed without applying steering."
            ),
        },
    )

    print(f"Saved replayed pre-steering JSON to: {output_json}")
    print(f"Saved point CSV to: {output_csv}")
    print(f"Saved summary to: {output_summary}")


if __name__ == "__main__":
    main()
