#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
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


CODE_HEADING_PATTERN = re.compile(r"###\s*Code\b\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute question-token similarities and the similarity at the ### -> Code "
            "next-token decoding position using the same prompt/reasoning logic as "
            "inference_tool_prompt_tag_suppressed_kvcache.py."
        )
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        required=True,
        help="JSON file containing either a list of question strings or objects with a 'problem' field.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Local model path used for prompt-only forward passes and generation replay.",
    )
    parser.add_argument(
        "--steering-vector-dir",
        type=Path,
        required=True,
        help="Directory containing step_mark_search/code/askuser.pt for the target model.",
    )
    parser.add_argument(
        "--steering-payload-name",
        type=str,
        default="step_mark_code.pt",
        help="Steering payload filename inside --steering-vector-dir.",
    )
    parser.add_argument(
        "--steering-layer",
        type=str,
        default="16",
        help="Saved steering layer id to analyze, for example '16'.",
    )
    parser.add_argument(
        "--domain",
        choices=("math", "time", "intention"),
        default="math",
        help="Domain whose prompt instruction template should be used.",
    )
    parser.add_argument(
        "--method",
        choices=("llama", "mistral"),
        default="llama",
        help="Prompt packing style.",
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
        help="Prompt truncation length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per reasoning round.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum reasoning rounds, aligned with inference_tool_prompt_tag_suppressed_kvcache.py.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of highest-cosine question tokens to summarize per question.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to save the combined similarity JSON.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}.")

    questions: list[str] = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            questions.append(item)
            continue
        if isinstance(item, dict):
            if isinstance(item.get("problem"), str):
                questions.append(item["problem"])
                continue
            if isinstance(item.get("task"), str):
                questions.append(item["task"])
                continue
        raise ValueError(
            f"Unsupported question entry at index {index}: expected string or object with 'problem'/'task'."
        )
    return questions


def build_messages_for_question(question: str, domain: str, method: str) -> list[dict[str, str]]:
    instruction = base.build_domain_instruction(domain)
    return base.build_messages(
        instruction=instruction,
        input_text=base.build_steering_aligned_input(question),
        method=method,
    )


def render_prompt_text(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return base.render_messages_without_template(messages)


def encode_prompt_text_with_offsets(tokenizer, prompt_text: str) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    token_ids = encoded.input_ids[0].detach().cpu()
    offsets = [tuple(pair) for pair in encoded.offset_mapping[0].detach().cpu().tolist()]
    return token_ids, offsets


def prepare_selected_layer(
    model,
    steering_vector_dir: Path,
    steering_payload_name: str,
    steering_layer: str,
) -> tuple[torch.Tensor, list[int], tuple[int, int], int]:
    steering_vectors, layer_indices = base.load_steering_payload(steering_vector_dir / steering_payload_name)
    layer_map = base.resolve_layer_map(layer_indices, len(base.get_transformer_layers(model)))
    selection_summary = base.select_pairs_for_explicit_saved_layers(
        layer_indices=layer_indices,
        layer_map=layer_map,
        steering_layer_spec=steering_layer,
    )
    selected_pairs = selection_summary["selected_pairs"]
    if len(selected_pairs) != 1:
        raise ValueError(
            f"Expected exactly one selected steering layer from {steering_layer!r}, got {len(selected_pairs)}."
        )
    model_layer_idx, vector_idx = selected_pairs[0]
    saved_layer = layer_indices[vector_idx]
    return steering_vectors, layer_indices, (model_layer_idx, vector_idx), saved_layer


def compute_question_token_records(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    model_layer_idx: int,
    offsets: list[tuple[int, int]],
    question_span: tuple[int, int],
) -> list[dict[str, Any]]:
    with torch.inference_mode():
        outputs = model(
            input_ids=prompt_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model forward pass did not return hidden states.")

    layer_hidden = hidden_states[model_layer_idx + 1][0].detach().float().cpu()
    steering_vector = steering_vector.detach().float().cpu()
    norms = torch.linalg.vector_norm(layer_hidden, dim=1)
    vector_norm = torch.linalg.vector_norm(steering_vector)
    cosine_values = F.cosine_similarity(
        layer_hidden,
        steering_vector.unsqueeze(0).expand_as(layer_hidden),
        dim=1,
    )

    token_ids = prompt_ids[0].detach().cpu().tolist()
    special_ids = set(tokenizer.all_special_ids)
    question_start, question_end = question_span
    records: list[dict[str, Any]] = []
    for position, token_id in enumerate(token_ids):
        char_start, char_end = offsets[position]
        in_question_span = char_end > question_start and char_start < question_end
        token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        records.append(
            {
                "position": position,
                "token_id": int(token_id),
                "token_str": token_str,
                "token_text": token_text,
                "is_special": bool(token_id in special_ids),
                "char_start": int(char_start),
                "char_end": int(char_end),
                "in_question_span": bool(in_question_span),
                "hidden_norm": float(norms[position].item()),
                "steering_norm": float(vector_norm.item()),
                "cosine_similarity": float(cosine_values[position].item()),
            }
        )
    return records


def summarize_top_tokens(token_records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(token_records, key=lambda item: item["cosine_similarity"], reverse=True)
    return ranked[:top_k]


def compute_hidden_state_similarity(hidden_state: torch.Tensor, steering_vector: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            hidden_state.detach().float().cpu().unsqueeze(0),
            steering_vector.detach().float().cpu().unsqueeze(0),
            dim=1,
        ).item()
    )


def generate_until_code_heading(
    model,
    tokenizer,
    input_messages: list[dict[str, str]],
    device: torch.device,
    max_seq_length: int,
    max_new_tokens: int,
    max_steps: int,
    model_layer_idx: int,
    steering_vector: torch.Tensor,
) -> dict[str, Any]:
    raw_rounds: list[str] = []
    canonical_steps: list[dict[str, Any]] = []
    heading_events: list[dict[str, Any]] = []

    for round_index in range(1, max_steps + 1):
        prompt_ids = base.build_prompt_ids(tokenizer, input_messages, device, max_seq_length)
        generated_text = ""
        current_input_ids = prompt_ids
        attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
        past_key_values = None

        for generation_step in range(1, max_new_tokens + 1):
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
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token_id = int(next_token.item())
            next_token_text = tokenizer.decode(next_token[0], skip_special_tokens=False)

            event = None
            if should_consider_heading and outputs.hidden_states is not None:
                hidden_state = outputs.hidden_states[model_layer_idx + 1][0, -1, :].detach().float().cpu()
                cosine = compute_hidden_state_similarity(hidden_state, steering_vector)
                event = {
                    "round_index": int(round_index),
                    "generation_step": int(generation_step),
                    "generated_text_before_heading_completion": generated_text,
                    "generated_text_before_tail": generated_text[-160:],
                    "next_token_id": next_token_id,
                    "next_token_text": next_token_text,
                    "cosine_similarity": float(cosine),
                }

            generated_text += next_token_text

            if event is not None:
                event["generated_text_after_heading_completion"] = generated_text
                event["generated_text_after_tail"] = generated_text[-160:]
                heading_events.append(event)
                if CODE_HEADING_PATTERN.search(generated_text.rstrip()):
                    return {
                        "found_code_heading": True,
                        "code_heading_trigger": {
                            **event,
                            "round_output_prefix": generated_text,
                            "round_output_prefix_tail": generated_text[-240:],
                        },
                        "raw_rounds": raw_rounds + [generated_text],
                        "canonical_steps_before_code": canonical_steps,
                        "all_heading_events": heading_events,
                    }

            if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
                break

            current_input_ids = next_token
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=1,
            )

        assistant_output = generated_text.strip()
        raw_rounds.append(assistant_output)

        new_steps = base.canonicalize_steps(
            base.normalize_round_steps(assistant_output),
            start_step_number=len(canonical_steps) + 1,
        )

        first_tool_idx = next(
            (idx for idx, step in enumerate(new_steps) if step.get("type") == "tool"),
            None,
        )
        if first_tool_idx is not None and first_tool_idx + 1 < len(new_steps):
            new_steps = new_steps[: first_tool_idx + 1]

        canonical_steps.extend(new_steps)

        if new_steps and new_steps[-1].get("tool_name") == "Code":
            return {
                "found_code_heading": False,
                "code_heading_trigger": None,
                "raw_rounds": raw_rounds,
                "canonical_steps_before_code": canonical_steps,
                "all_heading_events": heading_events,
                "note": "A Code step was parsed canonically, but the ### -> Code next-token trigger was not captured explicitly.",
            }

        if new_steps and new_steps[-1]["name"] == "Final Response":
            return {
                "found_code_heading": False,
                "code_heading_trigger": None,
                "raw_rounds": raw_rounds,
                "canonical_steps_before_code": canonical_steps,
                "all_heading_events": heading_events,
                "note": "Generation reached Final Response before any Code heading.",
            }

        if not new_steps:
            return {
                "found_code_heading": False,
                "code_heading_trigger": None,
                "raw_rounds": raw_rounds,
                "canonical_steps_before_code": canonical_steps,
                "all_heading_events": heading_events,
                "note": "No canonical steps were parsed from the generated round.",
            }

        input_messages[-1]["content"] = (
            input_messages[-1]["content"].strip()
            + "\n"
            + base.format_steps(new_steps).strip()
            + base.build_continue_prompt(len(canonical_steps) + 1)
        )

    return {
        "found_code_heading": False,
        "code_heading_trigger": None,
        "raw_rounds": raw_rounds,
        "canonical_steps_before_code": canonical_steps,
        "all_heading_events": heading_events,
        "note": f"Did not reach a Code heading within max_steps={max_steps}.",
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    print("Loading questions...")
    questions = load_questions(args.questions_path)

    print("Loading model and tokenizer...")
    tokenizer, model, device = base.load_model_and_tokenizer(args.model_name_or_path, args.device)

    print("Loading steering payload and selecting layer...")
    steering_vectors, layer_indices, selected_pair, saved_layer = prepare_selected_layer(
        model=model,
        steering_vector_dir=args.steering_vector_dir,
        steering_payload_name=args.steering_payload_name,
        steering_layer=args.steering_layer,
    )
    model_layer_idx, vector_idx = selected_pair
    steering_vector = steering_vectors[vector_idx]

    results: list[dict[str, Any]] = []
    for question_index, question in enumerate(questions, start=1):
        print(f"Processing question {question_index}/{len(questions)}...")
        messages = build_messages_for_question(question, args.domain, args.method)
        prompt_ids = base.build_prompt_ids(tokenizer, messages, device, args.max_seq_length)
        prompt_text = render_prompt_text(tokenizer, messages)
        prompt_text_token_ids, offsets = encode_prompt_text_with_offsets(tokenizer, prompt_text)
        prompt_id_list = prompt_ids[0].detach().cpu().tolist()
        if prompt_text_token_ids.tolist() != prompt_id_list:
            raise ValueError("Prompt text tokenization does not match prompt ids from build_prompt_ids.")
        question_start = prompt_text.find(question)
        if question_start == -1:
            raise ValueError("Could not locate the full question string inside the rendered prompt text.")
        question_span = (question_start, question_start + len(question))
        token_records = compute_question_token_records(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            steering_vector=steering_vector,
            model_layer_idx=model_layer_idx,
            offsets=offsets,
            question_span=question_span,
        )
        question_token_records = [item for item in token_records if item["in_question_span"]]
        if not question_token_records:
            raise ValueError("No prompt tokens were mapped to the question span.")

        cosine_values = [item["cosine_similarity"] for item in token_records]
        question_cosine_values = [item["cosine_similarity"] for item in question_token_records]

        code_heading_info = generate_until_code_heading(
            model=model,
            tokenizer=tokenizer,
            input_messages=deepcopy(messages),
            device=device,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            max_steps=args.max_steps,
            model_layer_idx=model_layer_idx,
            steering_vector=steering_vector,
        )

        results.append(
            {
                "question_index": question_index,
                "question": question,
                "domain": args.domain,
                "method": args.method,
                "steering_payload_name": args.steering_payload_name,
                "saved_steering_layer": int(saved_layer),
                "model_layer_index": int(model_layer_idx),
                "messages": messages,
                "prompt_text": prompt_text,
                "question_char_span": {
                    "start": int(question_span[0]),
                    "end": int(question_span[1]),
                },
                "num_prompt_tokens": len(token_records),
                "num_question_tokens": len(question_token_records),
                "cosine_summary": {
                    "min": float(min(cosine_values)),
                    "max": float(max(cosine_values)),
                    "mean": float(sum(cosine_values) / len(cosine_values)),
                },
                "question_cosine_summary": {
                    "min": float(min(question_cosine_values)),
                    "max": float(max(question_cosine_values)),
                    "mean": float(sum(question_cosine_values) / len(question_cosine_values)),
                },
                "top_positive_question_tokens": summarize_top_tokens(question_token_records, args.top_k),
                "token_records": token_records,
                "code_heading_generation": code_heading_info,
            }
        )

    payload = {
        "questions_path": str(args.questions_path),
        "model_name_or_path": args.model_name_or_path,
        "steering_vector_dir": str(args.steering_vector_dir),
        "steering_payload_name": args.steering_payload_name,
        "saved_steering_layer": int(saved_layer),
        "model_layer_index": int(model_layer_idx),
        "available_layer_indices": layer_indices,
        "domain": args.domain,
        "method": args.method,
        "results": results,
    }
    save_json(args.output, payload)
    print(f"Saved question-token and code-heading similarities to {args.output}")


if __name__ == "__main__":
    main()
