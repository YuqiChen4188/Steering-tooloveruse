#!/usr/bin/env python3
"""Run SteeringMark inference with heading-time QR-subspace projection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

import inference_tool_prompt_tag_suppressed_kvcache as base


DOMAIN_TO_SUBSPACE_GROUP = {
    "math": "code",
    "time": "search_askuser",
    "intention": "search_askuser",
}


def load_subspace_payload(
    path: Path,
) -> tuple[torch.Tensor, list[int], list[float], list[int], list[str], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if "subspace_basis" not in payload:
        raise ValueError(f"Subspace payload at {path} is missing 'subspace_basis'.")
    subspace_basis = payload["subspace_basis"]
    layer_indices = payload.get("layer_indices")
    if layer_indices is None:
        raise ValueError(f"Subspace payload at {path} is missing 'layer_indices'.")
    subspace_strengths = payload.get("subspace_strengths")
    if subspace_strengths is None:
        raise ValueError(f"Subspace payload at {path} is missing 'subspace_strengths'.")
    effective_ranks = payload.get("effective_ranks")
    if effective_ranks is None:
        raise ValueError(f"Subspace payload at {path} is missing 'effective_ranks'.")
    source_groups = payload.get("source_groups", [])
    return subspace_basis, list(layer_indices), list(subspace_strengths), list(effective_ranks), list(source_groups), payload


def select_top_layers_by_subspace_strength(
    layer_map: dict[int, int],
    subspace_strengths: list[float],
    max_steering_layers: int,
) -> dict[str, Any]:
    scored_pairs = []
    for model_layer_idx, basis_idx in layer_map.items():
        score = float(subspace_strengths[basis_idx])
        scored_pairs.append((model_layer_idx, basis_idx, score))
    scored_pairs.sort(key=lambda item: item[2], reverse=True)
    selected = sorted(scored_pairs[:max_steering_layers], key=lambda item: item[0])
    selected_pairs = [(model_layer_idx, basis_idx) for model_layer_idx, basis_idx, _ in selected]
    return {
        "selected_pairs": selected_pairs,
        "selected_layers": [layer_idx for layer_idx, _ in selected_pairs],
        "selected_strengths": [float(score) for _layer_idx, _basis_idx, score in selected],
    }


class TagTriggeredSubspaceProjectionHookManager:
    """Project out the tool subspace on the last position once generation enters a `###` heading prefix."""

    def __init__(
        self,
        model: base.AutoModelForCausalLM,
        subspace_basis: torch.Tensor,
        effective_ranks: list[int],
        selected_pairs: list[tuple[int, int]],
        strength: float,
    ) -> None:
        self.scale = 0.0
        self.handles: list[Any] = []
        self.strength = strength
        self.layers = base.get_transformer_layers(model)
        self.bases = subspace_basis.cpu()
        self.effective_ranks = effective_ranks

        for model_layer_idx, basis_idx in selected_pairs:
            basis = self.bases[basis_idx]
            effective_rank = int(self.effective_ranks[basis_idx])
            handle = self.layers[model_layer_idx].register_forward_hook(self._make_hook(basis, effective_rank))
            self.handles.append(handle)

    def set_scale(self, scale: float) -> None:
        self.scale = scale

    def _make_hook(self, basis: torch.Tensor, effective_rank: int):
        def hook(_module, _inputs, output):
            if self.scale == 0.0:
                return output

            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return output
            if hidden.shape[1] <= 0:
                return output

            basis_device = basis[:effective_rank].to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden.clone()
            last_hidden = hidden[:, -1, :]
            coefficients = torch.matmul(last_hidden, basis_device.transpose(0, 1))
            projected = torch.matmul(coefficients, basis_device)
            hidden[:, -1, :] = last_hidden - (projected * self.scale * self.strength)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def build_projection_trigger_info(
    generated_text: str,
    hidden_states: tuple[torch.Tensor, ...],
    selected_pairs: list[tuple[int, int]],
    subspace_basis: torch.Tensor,
    effective_ranks: list[int],
    layer_indices: list[int],
) -> dict[str, Any]:
    per_layer = []
    for model_layer_idx, basis_idx in selected_pairs:
        hidden_state = hidden_states[model_layer_idx + 1][0, -1, :].detach().float().cpu()
        effective_rank = int(effective_ranks[basis_idx])
        basis = subspace_basis[basis_idx][:effective_rank].detach().float().cpu()
        coefficients = torch.matmul(hidden_state, basis.transpose(0, 1))
        projected = torch.matmul(coefficients, basis)
        hidden_norm = torch.linalg.vector_norm(hidden_state).item()
        projection_norm = torch.linalg.vector_norm(projected).item()
        residual_norm = torch.linalg.vector_norm(hidden_state - projected).item()
        projection_ratio = projection_norm / hidden_norm if hidden_norm > 0 else None
        per_layer.append(
            {
                "model_layer": int(model_layer_idx),
                "saved_layer": int(layer_indices[basis_idx]),
                "basis_rank": int(basis.shape[0]),
                "hidden_norm": float(hidden_norm),
                "projection_norm": float(projection_norm),
                "residual_norm": float(residual_norm),
                "projection_norm_ratio": float(projection_ratio) if projection_ratio is not None else None,
            }
        )

    valid_ratios = [item["projection_norm_ratio"] for item in per_layer if item["projection_norm_ratio"] is not None]
    mean_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else None
    max_ratio = max(valid_ratios) if valid_ratios else None
    return {
        "context_tail_before_trigger": generated_text[-160:],
        "mean_selected_layer_projection_ratio": float(mean_ratio) if mean_ratio is not None else None,
        "max_selected_layer_projection_ratio": float(max_ratio) if max_ratio is not None else None,
        "per_layer_projection": per_layer,
    }


def summarize_token_diagnostics(token_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    projected_steps = [
        item["generation_step"]
        for item in token_diagnostics
        if item.get("projection_active")
    ]
    trigger_infos = [item["trigger_info"] for item in token_diagnostics if item.get("trigger_info") is not None]
    mean_ratios = [
        info["mean_selected_layer_projection_ratio"]
        for info in trigger_infos
        if info.get("mean_selected_layer_projection_ratio") is not None
    ]
    return {
        "num_generation_steps": len(token_diagnostics),
        "num_projected_steps": len(projected_steps),
        "first_projected_step": projected_steps[0] if projected_steps else None,
        "last_projected_step": projected_steps[-1] if projected_steps else None,
        "final_fragment_tail": token_diagnostics[-1]["generated_fragment_tail"] if token_diagnostics else "",
        "first_trigger_info": trigger_infos[0] if trigger_infos else None,
        "max_mean_selected_layer_projection_ratio": max(mean_ratios) if mean_ratios else None,
    }


def greedy_generate_with_tag_triggered_subspace_projection_kv_cache(
    model: base.AutoModelForCausalLM,
    tokenizer: base.AutoTokenizer,
    prompt_ids: torch.Tensor,
    hook_manager: TagTriggeredSubspaceProjectionHookManager,
    suppress_scale: float,
    max_new_tokens: int,
    selected_pairs: list[tuple[int, int]],
    subspace_basis: torch.Tensor,
    effective_ranks: list[int],
    layer_indices: list[int],
) -> tuple[str, list[dict[str, Any]]]:
    generated_text = ""
    diagnostics: list[dict[str, Any]] = []
    generated_ids: list[torch.Tensor] = []
    attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
    current_input_ids = prompt_ids
    past_key_values = None

    for generation_step in range(max_new_tokens):
        should_project = base.has_open_heading_prefix(generated_text)
        hook_manager.set_scale(suppress_scale if should_project else 0.0)
        with torch.inference_mode():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=should_project,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_token)

        next_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
        generated_text += next_text
        trigger_info = None
        if should_project and outputs.hidden_states is not None:
            trigger_info = build_projection_trigger_info(
                generated_text=generated_text,
                hidden_states=outputs.hidden_states,
                selected_pairs=selected_pairs,
                subspace_basis=subspace_basis,
                effective_ranks=effective_ranks,
                layer_indices=layer_indices,
            )

        diagnostics.append(
            {
                "generation_step": generation_step + 1,
                "projection_active": should_project,
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

    default_tool_group = DOMAIN_TO_SUBSPACE_GROUP[domain]
    steering_payload_name = args.steering_payload_name or f"step_mark_{default_tool_group}.pt"
    steering_vector_path = args.steering_vector_dir / steering_payload_name
    print(f"Using domain={domain} -> default_qr_subspace_group={default_tool_group}")
    print(f"Loading QR-subspace payload from {steering_vector_path}")
    (
        subspace_basis,
        layer_indices,
        subspace_strengths,
        effective_ranks,
        source_groups,
        payload,
    ) = load_subspace_payload(steering_vector_path)
    loaded_tool_group = str(payload.get("vector_group", default_tool_group))
    print(f"Loaded payload vector_group={loaded_tool_group} with source_groups={source_groups}")
    layer_map = base.resolve_layer_map(layer_indices, len(base.get_transformer_layers(model)))
    if args.steering_layers is not None:
        selection_summary = base.select_pairs_for_explicit_saved_layers(
            layer_indices=layer_indices,
            layer_map=layer_map,
            steering_layer_spec=args.steering_layers,
        )
        selection_summary["selected_strengths"] = [
            float(subspace_strengths[basis_idx]) for _model_layer_idx, basis_idx in selection_summary["selected_pairs"]
        ]
    else:
        selection_summary = select_top_layers_by_subspace_strength(
            layer_map=layer_map,
            subspace_strengths=subspace_strengths,
            max_steering_layers=args.max_steering_layers,
        )
    selected_pairs = selection_summary["selected_pairs"]
    selected_layers = selection_summary["selected_layers"]
    selected_saved_layers = selection_summary.get(
        "selected_saved_layers",
        [layer_indices[basis_idx] for _model_layer_idx, basis_idx in selected_pairs],
    )
    selected_strengths = selection_summary.get("selected_strengths", [])

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    args.code_exec_dir.mkdir(parents=True, exist_ok=True)
    error_save_path = args.error_save_path or args.save_path.with_name(args.save_path.stem + "_errors.json")
    error_save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_path.exists() and not args.overwrite:
        with args.save_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        existing_tasks = {result["task"] for result in results}
    else:
        results = []
        existing_tasks = set()

    if error_save_path.exists() and not args.overwrite:
        with error_save_path.open("r", encoding="utf-8") as f:
            error_results = json.load(f)
        existing_error_tasks = {result["task"] for result in error_results}
    else:
        error_results = []
        existing_error_tasks = set()

    hook_manager = TagTriggeredSubspaceProjectionHookManager(
        model=model,
        subspace_basis=subspace_basis,
        effective_ranks=effective_ranks,
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
                    assistant_output, token_diagnostics = greedy_generate_with_tag_triggered_subspace_projection_kv_cache(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        hook_manager=hook_manager,
                        suppress_scale=args.suppress_scale,
                        max_new_tokens=args.max_new_tokens,
                        selected_pairs=selected_pairs,
                        subspace_basis=subspace_basis,
                        effective_ranks=effective_ranks,
                        layer_indices=layer_indices,
                    )
                    raw.append(assistant_output)
                    steering_diagnostics.append(
                        {
                            "step": step_time,
                            "steering_mode": "qr_subspace_heading_projection",
                            "domain": domain,
                            "tool_group": loaded_tool_group,
                            "source_groups": source_groups,
                            "payload_name": steering_payload_name,
                            "payload_type": payload.get("payload_type"),
                            "selected_layers": selected_layers,
                            "selected_saved_layers": selected_saved_layers,
                            "selected_subspace_strengths": selected_strengths,
                            "steering_strength": args.steering_strength,
                            "suppress_scale": args.suppress_scale,
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
                            search_query = base.sanitize_search_query(new_steps[-1]["reasoning"])
                            new_steps[-1]["reasoning"] = search_query
                            new_steps[-1]["block_text"] = base.render_canonical_step(
                                new_steps[-1],
                                new_steps[-1]["step_number"],
                            )
                            response = base.search_serper(search_query, link=link, num=args.search_top_k)
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
        default=Path("/data/yuqi/SteeringMark/steering_vector/Llama_3_8_vector_heading_qr_subspace"),
        help="Directory containing QR-subspace payloads such as step_mark_search_askuser.pt.",
    )
    parser.add_argument(
        "--steering_payload_name",
        type=str,
        default=None,
        help="Optional payload filename inside --steering_vector_dir. Overrides the domain-specific default group.",
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
    parser.add_argument("--max_steering_layers", type=int, default=4, help="Top-k steering layers by subspace strength")
    parser.add_argument(
        "--steering_layers",
        type=str,
        default=None,
        help="Optional payload layer ids to intervene on, e.g. '16-20' or '18,20,22'. Overrides --max_steering_layers.",
    )
    parser.add_argument("--steering_strength", type=float, default=1.0, help="Multiplier applied to the projected subspace")
    parser.add_argument("--suppress_scale", type=float, default=1.0, help="Additional scale once a `###` heading prefix is active")
    parser.add_argument("--search_top_k", type=int, default=3, help="Top-k web results for Search")
    parser.add_argument(
        "--token_diagnostics_mode",
        choices=("summary", "full"),
        default="summary",
        help="How much per-token projection trace to save. Default keeps only a compact summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing save/error files and overwrite results from scratch.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce runtime logging")
    return parser.parse_args()


if __name__ == "__main__":
    inference(initialize())
