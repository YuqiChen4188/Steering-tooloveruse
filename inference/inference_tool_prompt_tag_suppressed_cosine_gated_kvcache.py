#!/usr/bin/env python3
"""Run SteeringMark tool-prompt inference with KV-cache cosine-gated heading-time suppression."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import inference_tool_prompt_tag_suppressed_kvcache as base


def run_cosine_gate(
    hidden_states: tuple[torch.Tensor, ...],
    selected_pairs: list[tuple[int, int]],
    steering_vectors: torch.Tensor,
    layer_indices: list[int],
    threshold: float,
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
    apply_steering = mean_cosine is not None and mean_cosine >= threshold
    return {
        "apply_steering": bool(apply_steering),
        "threshold": float(threshold),
        "mean_selected_layer_cosine": float(mean_cosine) if mean_cosine is not None else None,
        "per_layer_cosine": per_layer,
    }


def summarize_token_diagnostics(token_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    suppressed_steps = [
        item["generation_step"]
        for item in token_diagnostics
        if item.get("suppression_active")
    ]
    trigger_infos = [item["trigger_info"] for item in token_diagnostics if item.get("trigger_info") is not None]
    cosine_positive_steps = [
        item["generation_step"]
        for item in token_diagnostics
        if (item.get("cosine_gate_info") or {}).get("apply_steering")
    ]
    gate_infos = [item["cosine_gate_info"] for item in token_diagnostics if item.get("cosine_gate_info") is not None]
    return {
        "num_generation_steps": len(token_diagnostics),
        "num_suppressed_steps": len(suppressed_steps),
        "first_suppressed_step": suppressed_steps[0] if suppressed_steps else None,
        "last_suppressed_step": suppressed_steps[-1] if suppressed_steps else None,
        "num_cosine_positive_steps": len(cosine_positive_steps),
        "first_cosine_positive_step": cosine_positive_steps[0] if cosine_positive_steps else None,
        "final_fragment_tail": token_diagnostics[-1]["generated_fragment_tail"] if token_diagnostics else "",
        "first_trigger_info": trigger_infos[0] if trigger_infos else None,
        "max_gate_mean_cosine": (
            max(
                info["mean_selected_layer_cosine"]
                for info in gate_infos
                if info.get("mean_selected_layer_cosine") is not None
            )
            if gate_infos
            else None
        ),
        "max_mean_selected_layer_cosine": (
            max(info["mean_selected_layer_cosine"] for info in trigger_infos) if trigger_infos else None
        ),
    }


def greedy_generate_with_cosine_gated_suppression_kv_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    hook_manager: base.TagTriggeredSuppressionHookManager,
    suppress_scale: float,
    max_new_tokens: int,
    selected_pairs: list[tuple[int, int]],
    steering_vectors: torch.Tensor,
    layer_indices: list[int],
    cosine_threshold: float,
) -> tuple[str, list[dict[str, Any]]]:
    generated_text = ""
    diagnostics: list[dict[str, Any]] = []
    generated_ids: list[torch.Tensor] = []
    attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
    current_input_ids = prompt_ids
    past_key_values = None

    for generation_step in range(max_new_tokens):
        should_consider_heading = base.has_open_heading_prefix(generated_text)
        cosine_gate_info = None
        steering_applied = False
        current_past_key_values = past_key_values
        hook_manager.set_scale(0.0)

        if should_consider_heading:
            with torch.inference_mode():
                base_outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=current_past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            cosine_gate_info = run_cosine_gate(
                hidden_states=base_outputs.hidden_states,
                selected_pairs=selected_pairs,
                steering_vectors=steering_vectors,
                layer_indices=layer_indices,
                threshold=cosine_threshold,
            )
            steering_applied = cosine_gate_info["apply_steering"]
            if steering_applied:
                hook_manager.set_scale(suppress_scale)
                with torch.inference_mode():
                    outputs = model(
                        input_ids=current_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=current_past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
            else:
                outputs = base_outputs
        else:
            with torch.inference_mode():
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=current_past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_token)

        next_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
        generated_text += next_text
        trigger_info = None
        if steering_applied and outputs.hidden_states is not None:
            trigger_info = base.build_trigger_info(
                generated_text=generated_text,
                hidden_states=outputs.hidden_states,
                selected_pairs=selected_pairs,
                steering_vectors=steering_vectors,
                layer_indices=layer_indices,
            )

        diagnostics.append(
            {
                "generation_step": generation_step + 1,
                "suppression_active": steering_applied,
                "cosine_gate_info": cosine_gate_info,
                "generated_fragment_tail": generated_text[-40:],
                "trigger_info": trigger_info,
            }
        )

        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            break

        current_input_ids = next_token
        next_attention = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, next_attention], dim=1)

    hook_manager.set_scale(0.0)
    if generated_ids:
        output_ids = torch.cat(generated_ids, dim=1)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    else:
        output_text = ""
    return output_text, diagnostics


def inference(args: argparse.Namespace) -> None:
    print("Loading model and tokenizer...")
    tokenizer, model, device = base.load_model_and_tokenizer(args.model_name_or_path, args.device)

    domain = base.infer_domain(args.data_path, args.domain)

    print("Loading and preprocessing dataset...")
    dataset = base.preprocess_dataset(
        data_path=args.data_path,
        max_num=args.max_test_num,
        start_id=args.test_start_id,
        method=args.method,
        domain=domain,
    )

    tool_group = base.DOMAIN_TO_TOOL_GROUP[domain]
    steering_vector_path = args.steering_vector_dir / f"step_mark_{tool_group}.pt"
    print(f"Using domain={domain} -> tool_group={tool_group}")
    print(f"Loading steering payload from {steering_vector_path}")
    steering_vectors, layer_indices = base.load_steering_payload(steering_vector_path)
    layer_map = base.resolve_layer_map(layer_indices, len(base.get_transformer_layers(model)))
    if args.steering_layers is not None:
        selection_summary = base.select_pairs_for_explicit_saved_layers(
            layer_indices=layer_indices,
            layer_map=layer_map,
            steering_layer_spec=args.steering_layers,
        )
    else:
        selection_summary = base.select_top_layers_by_vector_norm(
            layer_map=layer_map,
            steering_vectors=steering_vectors,
            max_steering_layers=args.max_steering_layers,
        )
    selected_pairs = selection_summary["selected_pairs"]
    selected_layers = selection_summary["selected_layers"]
    selected_saved_layers = selection_summary.get(
        "selected_saved_layers",
        [layer_indices[vector_idx] for _model_layer_idx, vector_idx in selected_pairs],
    )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    args.code_exec_dir.mkdir(parents=True, exist_ok=True)
    error_save_path = args.error_save_path or args.save_path.with_name(args.save_path.stem + "_errors.json")
    error_save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_path.exists():
        with args.save_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        existing_tasks = {result["task"] for result in results}
    else:
        results = []
        existing_tasks = set()

    if error_save_path.exists():
        with error_save_path.open("r", encoding="utf-8") as f:
            error_results = json.load(f)
        existing_error_tasks = {result["task"] for result in error_results}
    else:
        error_results = []
        existing_error_tasks = set()

    hook_manager = base.TagTriggeredSuppressionHookManager(
        model=model,
        steering_vectors=steering_vectors,
        selected_pairs=selected_pairs,
        strength=args.steering_strength,
    )

    log = {"fail": 0, "success": 0}
    example_count = 0
    skipped_count = 0
    tool_usage_counts: dict[str, int] = defaultdict(int)

    progress = base.tqdm(dataset, desc="Inference", dynamic_ncols=True) if base.tqdm is not None else dataset

    try:
        for example in progress:
            input_messages = deepcopy(example["input"])
            ground_truth = example["ground_truth"]
            task = example["task"]

            example_count += 1
            code_file = args.code_exec_dir / f"{example_count}_{task[:3]}.py"

            if task in existing_tasks or task in existing_error_tasks:
                skipped_count += 1
                if base.tqdm is not None:
                    progress.set_postfix(base.build_progress_postfix(log, skipped_count, tool_usage_counts), refresh=False)
                continue

            steps: list[dict[str, Any]] = []
            raw: list[str] = []
            steering_diagnostics: list[dict[str, Any]] = []
            step_time = 0
            all_previous_code = ""
            task_tool_usage_counts: dict[str, int] = defaultdict(int)

            while True:
                try:
                    step_time += 1
                    if step_time > args.max_steps:
                        steps.append(
                            {
                                "name": "Final Response",
                                "type": "normal",
                                "tool_name": None,
                                "reasoning": base.TIMEOUT_FALLBACK,
                                "block_text": base.TIMEOUT_FALLBACK,
                            }
                        )
                        results.append(
                            base.build_result_record(
                                task=task,
                                ground_truth=ground_truth,
                                steps=steps,
                                raw=raw,
                                steering_diagnostics=steering_diagnostics,
                                selected_layers=selected_layers,
                            )
                        )
                        base.save_json(args.save_path, results)
                        base.log_example_tool_summary("timeout", step_time, task_tool_usage_counts, tool_usage_counts, args.quiet)
                        if base.tqdm is not None:
                            progress.set_postfix(
                                base.build_progress_postfix(
                                    log,
                                    skipped_count,
                                    tool_usage_counts,
                                    step_time=step_time,
                                    task_tool_usage_counts=task_tool_usage_counts,
                                ),
                                refresh=False,
                            )
                        break

                    prompt_ids = base.build_prompt_ids(tokenizer, input_messages, device, args.max_seq_length)
                    assistant_output, token_diagnostics = greedy_generate_with_cosine_gated_suppression_kv_cache(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        hook_manager=hook_manager,
                        suppress_scale=args.suppress_scale,
                        max_new_tokens=args.max_new_tokens,
                        selected_pairs=selected_pairs,
                        steering_vectors=steering_vectors,
                        layer_indices=layer_indices,
                        cosine_threshold=args.cosine_threshold,
                    )
                    raw.append(assistant_output)
                    steering_diagnostics.append(
                        {
                            "step": step_time,
                            "gating_mode": "cosine_gated_heading_suppression",
                            "domain": domain,
                            "tool_group": tool_group,
                            "selected_layers": selected_layers,
                            "selected_saved_layers": selected_saved_layers,
                            "suppress_scale": args.suppress_scale,
                            "cosine_threshold": args.cosine_threshold,
                            "token_diagnostics": token_diagnostics if args.token_diagnostics_mode == "full" else None,
                            "token_diagnostics_summary": summarize_token_diagnostics(token_diagnostics),
                        }
                    )

                    base.log_message("\n" + "+" * 10 + " Round Response " + "+" * 10, args.quiet)
                    base.log_message(assistant_output, args.quiet)

                    new_steps = base.canonicalize_steps(
                        base.normalize_round_steps(assistant_output),
                        start_step_number=len(steps) + 1,
                    )

                    first_tool_idx = next(
                        (idx for idx, step in enumerate(new_steps) if step.get("type") == "tool"),
                        None,
                    )
                    if first_tool_idx is not None and first_tool_idx + 1 < len(new_steps):
                        new_steps = new_steps[: first_tool_idx + 1]

                    if new_steps and new_steps[-1]["type"] == "tool":
                        tool_name = new_steps[-1]["tool_name"]
                        tool_usage_counts[tool_name] += 1
                        task_tool_usage_counts[tool_name] += 1
                        if tool_name == "AskUser":
                            response = base.simulate_user_response(task, new_steps[-1]["reasoning"])
                            new_steps[-1]["output"] = response
                        elif tool_name == "Search":
                            link = "intention" in str(args.data_path)
                            response = base.search_serper(new_steps[-1]["reasoning"], link=link, num=args.search_top_k)
                            new_steps[-1]["output"] = response
                        elif tool_name == "Code":
                            code_body = base.extract_python_code_block(new_steps[-1]["reasoning"])
                            if code_body is None:
                                response = "Error: No valid Python code block found."
                            else:
                                code_content = f"```python\n{all_previous_code}\n{code_body}\n```"
                                response = base.execute_code(code_content, code_file)
                            if not response.startswith("Error"):
                                new_code = base.parse_code_content(new_steps[-1]["reasoning"])
                                if new_code:
                                    all_previous_code += new_code + "\n"
                            new_steps[-1]["output"] = response
                        else:
                            raise AssertionError(f"Unknown tool name: {tool_name}")
                        base.log_tool_usage_update(tool_name, step_time, task_tool_usage_counts, tool_usage_counts, args.quiet)
                        base.log_message("\n" + "+" * 10 + f" Tool {tool_name} Response " + "+" * 10, args.quiet)
                        base.log_message(response, args.quiet)
                        if base.tqdm is not None:
                            progress.set_postfix(
                                base.build_progress_postfix(
                                    log,
                                    skipped_count,
                                    tool_usage_counts,
                                    step_time=step_time,
                                    task_tool_usage_counts=task_tool_usage_counts,
                                    last_tool=tool_name,
                                ),
                                refresh=False,
                            )

                    if new_steps and new_steps[-1]["name"] == "Final Response":
                        log["success"] += 1
                        steps.extend(new_steps)
                        results.append(
                            base.build_result_record(
                                task=task,
                                ground_truth=ground_truth,
                                steps=steps,
                                raw=raw,
                                steering_diagnostics=steering_diagnostics,
                                selected_layers=selected_layers,
                            )
                        )
                        base.save_json(args.save_path, results)
                        base.log_example_tool_summary("success", step_time, task_tool_usage_counts, tool_usage_counts, args.quiet)
                        if base.tqdm is not None:
                            progress.set_postfix(
                                {
                                    "ok": log["success"],
                                    "fail": log["fail"],
                                    "step": step_time,
                                    "tools_total": base.format_tool_usage_counts(tool_usage_counts),
                                    "tools_task": base.format_tool_usage_counts(task_tool_usage_counts),
                                },
                                refresh=False,
                            )
                        break

                    input_messages[-1]["content"] = (
                        input_messages[-1]["content"].strip()
                        + "\n"
                        + base.format_steps(new_steps).strip()
                        + base.build_continue_prompt(len(steps) + len(new_steps) + 1)
                    )
                    steps.extend(new_steps)

                except Exception as error:  # pragma: no cover - runtime path
                    log["fail"] += 1
                    error_results.append(
                        base.build_error_record(
                            task=task,
                            ground_truth=ground_truth,
                            steps=steps,
                            raw=raw,
                            step_time=step_time,
                            error=error,
                            steering_diagnostics=steering_diagnostics,
                            selected_layers=selected_layers,
                        )
                    )
                    base.save_json(error_save_path, error_results)
                    base.log_example_tool_summary("error", step_time, task_tool_usage_counts, tool_usage_counts, args.quiet)
                    base.log_message(str(error), args.quiet)
                    if base.tqdm is not None:
                        progress.set_postfix(
                            base.build_progress_postfix(
                                log,
                                skipped_count,
                                tool_usage_counts,
                                step_time=step_time,
                                task_tool_usage_counts=task_tool_usage_counts,
                            ),
                            refresh=False,
                        )
                    break
    finally:
        hook_manager.close()


def initialize() -> argparse.Namespace:
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the inference dataset")
    parser.add_argument(
        "--steering_vector_dir",
        type=Path,
        default=Path("/data/yuqi/SteeringMark/steering_vector/Llama_3_8_vector"),
        help="Directory containing step_mark_search/code/askuser.pt",
    )
    parser.add_argument("--save_path", type=Path, required=True, help="Path to save successful inference results")
    parser.add_argument("--error_save_path", type=Path, default=None, help="Optional path to save failed examples")
    parser.add_argument("--code_exec_dir", type=Path, default=Path("./env"), help="Directory for temporary code files")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens per reasoning round")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum reasoning rounds")
    parser.add_argument("--test_start_id", type=int, default=0, help="Start id for evaluation")
    parser.add_argument("--max_test_num", type=int, default=-1, help="Maximum number of evaluation examples")
    parser.add_argument("--method", type=str, default="llama", help="Prompt packing style")
    parser.add_argument("--device", type=str, default="auto", help='Device selection: "auto", "cuda", or "cpu"')
    parser.add_argument("--domain", choices=("math", "time", "intention"), default=None, help="Optional domain override")
    parser.add_argument("--max_steering_layers", type=int, default=4, help="Top-k steering layers by vector norm")
    parser.add_argument(
        "--steering_layers",
        type=str,
        default=None,
        help="Optional payload layer ids to intervene on, e.g. '16-20' or '18,20,22'. Overrides --max_steering_layers.",
    )
    parser.add_argument("--steering_strength", type=float, default=1.0, help="Multiplier applied to the loaded vector")
    parser.add_argument("--suppress_scale", type=float, default=1.0, help="Additional suppress-only scale once a `###` heading prefix is active")
    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.25,
        help="Apply negative steering only when the mean cosine similarity to the selected steering vector(s) at a `###` heading prefix is >= this threshold.",
    )
    parser.add_argument("--search_top_k", type=int, default=3, help="Top-k web results for Search")
    parser.add_argument(
        "--token_diagnostics_mode",
        choices=("summary", "full"),
        default="summary",
        help="How much per-token suppression trace to save. Default keeps only a compact summary.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce runtime logging")
    return parser.parse_args()


if __name__ == "__main__":
    inference(initialize())
