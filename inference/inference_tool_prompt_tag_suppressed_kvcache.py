#!/usr/bin/env python3
"""Run SteeringMark tool-prompt inference with KV-cache heading-time tool-specific suppression."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_schema_utils import (  # noqa: E402
    CODE_HEADINGS,
    SCHEMAS,
    json_action_parse_diagnostics,
    parse_json_action_steps,
    parse_markdown_blocks,
    render_step_for_schema,
    resolve_code_heading,
    resolve_schema,
    has_schema_trigger_prefix,
)
try:
    from utils_askuser import simulate_user_response
except ModuleNotFoundError:  # Math/code ablations do not need AskUser support.
    def simulate_user_response(_task: str, _question: str) -> str:
        raise RuntimeError("AskUser simulation requires the optional OpenAI dependency.")

from utils_code import execute_code, extract_python_code_block
from utils_serper import search_serper


TIMEOUT_FALLBACK = (
    "Still do not get an answer after exceeding maximum step time! "
    "Please judge the answer for this question as wrong."
)
LEGACY_HEADING_STEP_PATTERN = re.compile(
    r"(?ms)^###\s*(?P<tag>Reasoning|Search|Code|AskUser|FinalResponse|Final Response)(?:[ \t]+(?P<title>[^\n]*))?[ \t]*\n(?P<body>.*?)(?=^###\s*(?:Reasoning|Search|Code|AskUser|FinalResponse|Final Response)(?:[ \t]+[^\n]*)?[ \t]*\n|\Z)"
)
VALID_TAGS = {"Reasoning", "Search", "Code", "AskUser", "FinalResponse"}
DOMAIN_TO_TOOL_GROUP = {
    "math": "code",
    "time": "search",
    "intention": "search",
}
GENERIC_STEP_NAMES = {"Reasoning Step", "Tool Step", "Final Response"}
DEFAULT_STEP_TITLES = {
    "Reasoning": "Reasoning",
    "Search": "Search",
    "Code": "Code",
    "AskUser": "AskUser",
    "FinalResponse": "Final Response",
}
FINAL_RESPONSE_PATTERNS = (
    r"Final Answer\s*:",
    r"###\s*Final Response\b",
    r"Final Response\s*:",
    r"The final answer is\s*:?",
    r"The answer is\s*:?",
)
LEGACY_REASONING_HEADERS = (
    "### Continue your reasoning",
    "### Continue Reasoning Steps",
    "### Reasoning Steps",
    "### Reasoning",
)
DOMAIN_ALLOWED_TOOL_HEADINGS = {
    "math": ["### Code"],
    "time": ["### Search", "### AskUser"],
    "intention": ["### Search", "### AskUser"],
}
MATCHED_DOMAIN_ALLOWED_TOOL_HEADINGS = {
    "math": ["### Code"],
    "time": ["### Search"],
    "intention": ["### AskUser"],
}
DOMAIN_TOOL_USAGE = {
    "math": [
        "- Use code snippet ```python ... ``` inside a `### Code` step when computation is needed.",
    ],
    "time": [
        "- Use a `### Search` step whose content is a concise web search query when external information is needed.",
        "- Use a `### AskUser` step whose content is the exact question that should be asked to the user when user-specific clarification is needed.",
    ],
    "intention": [
        "- Use a `### Search` step whose content is a concise web search query when external information is needed.",
        "- Use a `### AskUser` step whose content is the exact question that should be asked to the user when user-specific clarification is needed.",
    ],
}
DOMAIN_TOOL_OUTPUT_GUIDANCE = {
    "math": [
        "- If you need the tool, place the executable snippet inside a `### Code` step using ```python ... ```.",
    ],
    "time": [
        "- If you use `### Search`, place only the search query inside the step.",
        "- If you use `### AskUser`, place only the user-facing question inside the step.",
    ],
    "intention": [
        "- If you use `### Search`, place only the search query inside the step.",
        "- If you use `### AskUser`, place only the user-facing question inside the step.",
    ],
}


def log_message(text: str, quiet: bool, always: bool = False) -> None:
    if always or not quiet:
        if tqdm is not None:
            tqdm.write(text, file=sys.stderr)
        else:
            print(text, file=sys.stderr, flush=True)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_tool_usage_counts(tool_usage_counts: dict[str, int]) -> str:
    return (
        f"Code={tool_usage_counts.get('Code', 0)} "
        f"Search={tool_usage_counts.get('Search', 0)} "
        f"AskUser={tool_usage_counts.get('AskUser', 0)}"
    )


def log_tool_usage_update(
    tool_name: str,
    step_time: int,
    task_tool_usage_counts: dict[str, int],
    tool_usage_counts: dict[str, int],
    quiet: bool,
) -> None:
    log_message(
        (
            f"[ToolStats] step={step_time} used={tool_name} "
            f"task[{format_tool_usage_counts(task_tool_usage_counts)}] "
            f"total[{format_tool_usage_counts(tool_usage_counts)}]"
        ),
        quiet,
        always=True,
    )


def log_example_tool_summary(
    status: str,
    step_time: int,
    task_tool_usage_counts: dict[str, int],
    tool_usage_counts: dict[str, int],
    quiet: bool,
) -> None:
    log_message(
        (
            f"[ToolStats] {status} step={step_time} "
            f"task[{format_tool_usage_counts(task_tool_usage_counts)}] "
            f"total[{format_tool_usage_counts(tool_usage_counts)}]"
        ),
        quiet,
        always=True,
    )


def build_progress_postfix(
    log: dict[str, int],
    skipped_count: int,
    tool_usage_counts: dict[str, int],
    step_time: int | None = None,
    task_tool_usage_counts: dict[str, int] | None = None,
    last_tool: str | None = None,
) -> dict[str, Any]:
    postfix: dict[str, Any] = {
        "ok": log["success"],
        "fail": log["fail"],
        "skip": skipped_count,
        "tools_total": format_tool_usage_counts(tool_usage_counts),
    }
    if step_time is not None:
        postfix["step"] = step_time
    if last_tool is not None:
        postfix["last_tool"] = last_tool
    if task_tool_usage_counts is not None and any(task_tool_usage_counts.get(name, 0) for name in ("Code", "Search", "AskUser")):
        postfix["tools_task"] = format_tool_usage_counts(task_tool_usage_counts)
    return postfix


def summarize_token_diagnostics(token_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    suppressed_steps = [
        item["generation_step"]
        for item in token_diagnostics
        if item.get("suppression_active")
    ]
    trigger_infos = [item["trigger_info"] for item in token_diagnostics if item.get("trigger_info") is not None]
    return {
        "num_generation_steps": len(token_diagnostics),
        "num_suppressed_steps": len(suppressed_steps),
        "first_suppressed_step": suppressed_steps[0] if suppressed_steps else None,
        "last_suppressed_step": suppressed_steps[-1] if suppressed_steps else None,
        "final_fragment_tail": token_diagnostics[-1]["generated_fragment_tail"] if token_diagnostics else "",
        "first_trigger_info": trigger_infos[0] if trigger_infos else None,
        "max_mean_selected_layer_cosine": (
            max(info["mean_selected_layer_cosine"] for info in trigger_infos) if trigger_infos else None
        ),
    }


def build_trigger_info(
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
        saved_layer = layer_indices[vector_idx]
        per_layer.append(
            {
                "model_layer": int(model_layer_idx),
                "saved_layer": int(saved_layer),
                "cosine_similarity": float(cosine),
            }
        )
    mean_cosine = sum(item["cosine_similarity"] for item in per_layer) / len(per_layer) if per_layer else None
    return {
        "context_tail_before_trigger": generated_text[-160:],
        "mean_selected_layer_cosine": float(mean_cosine) if mean_cosine is not None else None,
        "per_layer_cosine": per_layer,
    }


def build_result_record(
    task: str,
    ground_truth: str,
    steps: list[dict[str, Any]],
    raw: list[str],
    steering_diagnostics: list[dict[str, Any]],
    selected_layers: list[int],
) -> dict[str, Any]:
    return {
        "task": task,
        "predict": steps,
        "ground_truth": ground_truth,
        "raw": raw,
        "steering_diagnostics": steering_diagnostics,
        "selected_steering_layers": selected_layers,
    }


def build_error_record(
    task: str,
    ground_truth: str,
    steps: list[dict[str, Any]],
    raw: list[str],
    step_time: int,
    error: Exception,
    steering_diagnostics: list[dict[str, Any]],
    selected_layers: list[int],
) -> dict[str, Any]:
    return {
        "task": task,
        "ground_truth": ground_truth,
        "predict_partial": steps,
        "raw": raw,
        "step_time": step_time,
        "error": str(error),
        "error_type": type(error).__name__,
        "traceback": traceback.format_exc(),
        "steering_diagnostics": steering_diagnostics,
        "selected_steering_layers": selected_layers,
    }


def build_messages(instruction: str, input_text: str, method: str) -> list[dict[str, str]]:
    if method == "mistral":
        return [{"role": "user", "content": instruction.strip() + "\n\n" + input_text.strip()}]
    if method == "llama":
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ]
    raise ValueError(f"Unsupported method: {method}")


def extract_task_from_input(input_text: str) -> str:
    if "### Task" in input_text:
        remainder = input_text.split("### Task", 1)[1]
        if "###" in remainder:
            return remainder.split("###", 1)[0].strip()
        return remainder.strip()
    return input_text.strip()


def build_steering_aligned_input(task: str) -> str:
    return f"### Task\n{task}\n"


CONSERVATIVE_GENERAL_POLICY_LINES = [
    "- Prefer solving with your own reasoning whenever possible.",
    "- Use the tool only when it is strictly necessary to answer correctly.",
    "- Do not call the tool for routine calculations, simple deductions, or information that can be derived from the prompt and your parametric knowledge.",
    "- If the task requires current external information, exact computation beyond reliable mental arithmetic, or missing user-specific details, then use the appropriate tool.",
]
CONSERVATIVE_DOMAIN_POLICY_LINES = {
    "math": "- Use `### Code` only for computations that are too long, error-prone, or require exact programmatic verification. Do not use `### Code` for simple arithmetic or algebra that can be solved reliably by reasoning.",
    "time": "- Use `### Search` only when the answer depends on up-to-date or external information unavailable from the prompt. Do not use `### Search` for stable general knowledge or deductions from the provided context.",
    "intention": "- Use `### AskUser` only when a missing user-specific detail is essential. Do not ask the user when the intent can be reasonably inferred from the conversation.",
}


def conservative_policy_text(domain: str) -> str:
    return "\n".join([*CONSERVATIVE_GENERAL_POLICY_LINES, CONSERVATIVE_DOMAIN_POLICY_LINES[domain]])


def allowed_tool_headings_for(domain: str, code_heading: str, heading_interface: str) -> list[str]:
    if heading_interface not in {"default", "matched_prompt"}:
        raise ValueError(f"Unsupported heading interface: {heading_interface}")
    if domain == "math":
        return [f"### {code_heading}"]
    if heading_interface == "matched_prompt":
        return MATCHED_DOMAIN_ALLOWED_TOOL_HEADINGS[domain]
    return DOMAIN_ALLOWED_TOOL_HEADINGS[domain]


def domain_tool_usage_for(domain: str, code_heading: str, heading_interface: str) -> tuple[str, str]:
    if domain == "math" and code_heading == "Action_B":
        return (
            "- Use `### Action_B` when external computation is needed. Put executable Python in that step.",
            "- If external computation is needed, place the executable Python snippet inside a `### Action_B` step using ```python ... ```.",
        )
    if domain == "math":
        return (
            f"- Use `### {code_heading}` when computation is needed. Put executable Python in a fenced ```python ... ``` block in that step.",
            f"- If you need the tool, place the executable Python snippet inside a `### {code_heading}` step using ```python ... ```.",
        )
    if heading_interface == "matched_prompt" and domain == "time":
        return (
            "- Use a `### Search` step whose content is a concise web search query when external information is needed.",
            "- If you use `### Search`, place only the search query inside the step.",
        )
    if heading_interface == "matched_prompt" and domain == "intention":
        return (
            "- Use a `### AskUser` step whose content is the exact question that should be asked to the user when user-specific clarification is needed.",
            "- If you use `### AskUser`, place only the user-facing question inside the step.",
        )
    return "\n".join(DOMAIN_TOOL_USAGE[domain]), "\n".join(DOMAIN_TOOL_OUTPUT_GUIDANCE[domain])


def build_domain_instruction(
    domain: str,
    eval_schema: str = "markdown",
    code_heading: str = "Code",
    prompt_policy: str = "base_tool",
    heading_interface: str = "default",
) -> str:
    eval_schema = resolve_schema(eval_schema)
    code_heading = resolve_code_heading(code_heading)
    if prompt_policy not in {"base_tool", "conservative_tool"}:
        raise ValueError(f"Unsupported prompt policy: {prompt_policy}")
    if heading_interface not in {"default", "matched_prompt"}:
        raise ValueError(f"Unsupported heading interface: {heading_interface}")
    conservative_policy_block = (
        conservative_policy_text(domain) + "\n"
        if prompt_policy == "conservative_tool"
        else ""
    )
    if eval_schema == "json":
        if domain == "math":
            tool_usage = "- Use action `code` when computation is needed. Put executable Python directly in `content`; fenced ```python ... ``` blocks are also accepted."
            allowed_actions = "reasoning, code, final"
        else:
            tool_usage = "- Use action `search` for concise web search queries. Use action `askuser` for user-specific clarification."
            allowed_actions = "reasoning, code, final"
        return f"""### Task
You are a highly capable assistant designed to solve tasks effectively using your knowledge and the tool set available for this domain.

### Principles
1. Reason Independently:
- Leverage your own knowledge to analyze and solve reasoning steps whenever possible. Use tools only when necessary.
{conservative_policy_block}2. Tool Usage:
{tool_usage}
3. Step-by-Step Approach:
- Work through reasoning systematically. Rely on your knowledge until a gap requires tool support.
- After a tool call, continue reasoning using the tool output when available.
4. Goal-Oriented Resolution:
- End with a concise final answer.

### Output Guidelines
- Output one JSON object per line.
- Each JSON object must contain `action` and `content` fields.
- Allowed action values for this domain are: {allowed_actions}.
- Do not output Markdown headings such as `### Reasoning` or `### Code`.
- End with an object whose action is `final`."""
    allowed_tool_headings = allowed_tool_headings_for(domain, code_heading, heading_interface)
    allowed_heading_text = ", ".join(f"`{heading}`" for heading in ["### Reasoning", *allowed_tool_headings, "### Final Response"])
    tool_usage, tool_output_guidance = domain_tool_usage_for(domain, code_heading, heading_interface)
    conservative_output_guideline = (
        "- Keep tools available, but follow the prompt policy above when deciding whether to use them.\n"
        if prompt_policy == "conservative_tool"
        else ""
    )
    return f"""### Task
You are a highly capable assistant designed to solve tasks effectively using your knowledge and the tool set available for this domain.

### Principles
1. Reason Independently:
- Leverage your own knowledge to analyze and solve reasoning steps whenever possible. Use the tool only when necessary.
{conservative_policy_block}2. Tool Usage:
{tool_usage}
3. Step-by-Step Approach:
- Work through reasoning systematically, breaking down the task into manageable steps. Rely on your knowledge until a gap is identified that requires tool support.
- Use only the domain-appropriate tool when needed.
4. Goal-Oriented Resolution:
- Conclude your reasoning process by achieving a clear, accurate, and succinct solution based on your independent analysis and any tool findings.

### Output Guidelines
- Your answer must begin with `### Reasoning`.
- After that, every step must begin with one of these section headers on its own line: {allowed_heading_text}.
{tool_output_guidance}
- Do not emit any other tool headings.
{conservative_output_guideline}- After a tool call, continue reasoning using the tool output when available.
- End with a `### Final Response` step that directly answers the task."""


def preprocess_dataset(
    data_path: Path,
    max_num: int,
    start_id: int,
    method: str,
    domain: str,
    eval_schema: str = "markdown",
    code_heading: str = "Code",
    prompt_policy: str = "base_tool",
    heading_interface: str = "default",
) -> list[dict[str, Any]]:
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    max_num = len(data) if max_num == -1 else max_num
    dataset = []
    instruction = build_domain_instruction(
        domain,
        eval_schema=eval_schema,
        code_heading=code_heading,
        prompt_policy=prompt_policy,
        heading_interface=heading_interface,
    )
    for entry in data[start_id:]:
        task = extract_task_from_input(entry["input"])
        dataset.append(
            {
                "input": build_messages(
                    instruction,
                    build_steering_aligned_input(task),
                    method,
                ),
                "ground_truth": entry["output"],
                "task": task,
                "source_instruction_head": entry.get("instruction", "")[:120],
            }
        )
        if len(dataset) >= max_num:
            break

    print(f"Length of data: {len(dataset)}")
    return dataset


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
    if len(messages) == 1:
        return messages[0]["content"].strip()
    return "\n\n".join(message["content"].strip() for message in messages if message["content"].strip())


def build_prompt_ids(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    device: torch.device,
    max_seq_length: int,
) -> torch.Tensor:
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_ids = coerce_token_ids_to_tensor(prompt_ids)
    else:
        prompt_text = render_messages_without_template(messages)
        prompt_ids = coerce_token_ids_to_tensor(tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt"))

    if prompt_ids.shape[1] > max_seq_length:
        prompt_ids = prompt_ids[:, -max_seq_length:]
    return prompt_ids.to(device)


def extract_first_parentheses_content(text: str) -> str | None:
    start = text.find("(")
    if start == -1:
        return None
    stack: list[str] = []
    content: list[str] = []
    for char in text[start + 1 :]:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack:
                return "".join(content).strip()
            stack.pop()
        content.append(char)
    return None


def parse_tagged_steps(text: str, code_heading: str = "Code") -> list[dict[str, Any]]:
    steps = []
    for block in parse_markdown_blocks(strip_generation_artifacts(text), code_heading=code_heading):
        tag = block["tag"]
        if tag not in VALID_TAGS:
            continue
        steps.append(
            {
                "step_number": None,
                "tag": tag,
                "title": block.get("title", ""),
                "body": block.get("body", "").rstrip(),
                "block_text": block.get("block_text", "").rstrip(),
            }
        )
    return steps

def normalize_legacy_heading_tag(tag: str) -> str:
    compact = tag.replace(" ", "")
    if compact == "FinalResponse":
        return "FinalResponse"
    return tag


def parse_legacy_heading_steps(text: str, code_heading: str = "Code") -> list[dict[str, Any]]:
    return parse_tagged_steps(text, code_heading=code_heading)

def strip_generation_artifacts(text: str) -> str:
    cleaned = text
    if "### Output Guidelines" in cleaned:
        cleaned = cleaned.split("### Output Guidelines", 1)[0]
    if "** Input **" in cleaned:
        cleaned = cleaned.replace("** Input **", "")
    if "** Output **" in cleaned:
        cleaned = cleaned.replace("** Output **", "")
    return cleaned.strip()


def find_earliest_final_marker(text: str) -> tuple[int, int, str | None]:
    best_match = None
    for pattern in FINAL_RESPONSE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        if best_match is None or match.start() < best_match.start():
            best_match = match

    if best_match is None:
        return -1, -1, None
    return best_match.start(), best_match.end(), best_match.group(0)


def compose_step_body(step: dict[str, Any]) -> str:
    parts = []
    title = (step.get("title") or "").strip()
    body = (step.get("body") or "").strip()
    if title:
        parts.append(title)
    if body:
        parts.append(body)
    return "\n".join(parts).strip()


def append_merged_reasoning_step(
    normalized: list[dict[str, Any]],
    pending_reasoning: list[dict[str, Any]],
) -> None:
    if not pending_reasoning:
        return

    merged_body = "\n\n".join(
        body
        for body in (compose_step_body(step) for step in pending_reasoning)
        if body
    ).strip()
    if not merged_body:
        pending_reasoning.clear()
        return

    normalized.append(
        {
            "name": "Reasoning Step",
            "type": "normal",
            "tool_name": None,
            "reasoning": merged_body,
            "block_text": merged_body,
        }
    )
    pending_reasoning.clear()


def normalize_round_steps(text: str, schema: str = "markdown", code_heading: str = "Code") -> list[dict[str, Any]]:
    schema = resolve_schema(schema)
    code_heading = resolve_code_heading(code_heading)
    if schema == "json":
        parsed_json = parse_json_action_steps(strip_generation_artifacts(text))
        if parsed_json:
            normalized = []
            pending_reasoning = []
            for step in parsed_json:
                tag = step["tag"]
                if tag == "Reasoning":
                    pending_reasoning.append(step)
                    continue
                append_merged_reasoning_step(normalized, pending_reasoning)
                if tag == "FinalResponse":
                    normalized.append(
                        {
                            "name": "Final Response",
                            "type": "normal",
                            "tool_name": None,
                            "reasoning": compose_step_body(step),
                            "block_text": step["block_text"],
                        }
                    )
                    break
                normalized.append(
                    {
                        "name": "Tool Step",
                        "type": "tool",
                        "tool_name": tag,
                        "reasoning": compose_step_body(step),
                        "block_text": step["block_text"],
                    }
                )
                break
            append_merged_reasoning_step(normalized, pending_reasoning)
            return normalized

    cleaned_text = strip_generation_artifacts(text)
    parsed = parse_tagged_steps(cleaned_text, code_heading=code_heading)
    if not parsed:
        parsed = parse_legacy_heading_steps(cleaned_text, code_heading=code_heading)
    if parsed:
        normalized = []
        pending_reasoning = []
        keep_full_round = any(step["tag"] == "FinalResponse" for step in parsed)
        for step in parsed:
            tag = step["tag"]
            if tag == "Reasoning":
                pending_reasoning.append(step)
                continue
            append_merged_reasoning_step(normalized, pending_reasoning)
            if tag == "FinalResponse":
                normalized.append(
                    {
                        "name": "Final Response",
                        "type": "normal",
                        "tool_name": None,
                        "reasoning": compose_step_body(step),
                        "block_text": step["block_text"],
                    }
                )
                break
            normalized.append(
                {
                    "name": "Tool Step",
                    "type": "tool",
                    "tool_name": tag,
                    "reasoning": compose_step_body(step),
                    "block_text": step["block_text"],
                }
            )
            if not keep_full_round:
                break
        append_merged_reasoning_step(normalized, pending_reasoning)
        return normalized

    final_start, final_end, final_marker = find_earliest_final_marker(cleaned_text)
    if final_marker is not None:
        before_final = cleaned_text[:final_start]
        after_final = cleaned_text[final_end:].lstrip(" :\n\t")
        results = []
        if before_final.strip():
            results.append(
                {
                    "name": "Reasoning Step",
                    "type": "normal",
                    "tool_name": None,
                    "reasoning": before_final.strip(),
                    "block_text": before_final.strip(),
                }
            )
        results.append(
            {
                "name": "Final Response",
                "type": "normal",
                "tool_name": None,
                "reasoning": after_final.strip(),
                "block_text": f"### Final Response\n{after_final.strip()}",
            }
        )
        return results

    return [
        {
            "name": "Reasoning Step",
            "type": "normal",
            "tool_name": None,
            "reasoning": cleaned_text.strip(),
            "block_text": cleaned_text.strip(),
        }
    ]


def clean_step_reasoning(text: str) -> str:
    cleaned = text.strip()
    while True:
        updated = cleaned
        for header in LEGACY_REASONING_HEADERS:
            if updated.startswith(header):
                updated = updated[len(header) :].lstrip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def get_step_tag(step: dict[str, Any]) -> str:
    if step.get("type") == "tool":
        tool_name = step.get("tool_name")
        if tool_name in VALID_TAGS:
            return tool_name
        raise ValueError(f"Unsupported tool step tag: {tool_name}")
    if step.get("name") == "Final Response":
        return "FinalResponse"
    return "Reasoning"


def get_step_title(step: dict[str, Any], tag: str, code_heading: str = "Code") -> str:
    if tag == "Code":
        return code_heading
    return DEFAULT_STEP_TITLES[tag]


def render_canonical_step(step: dict[str, Any], step_number: int, schema: str = "markdown", code_heading: str = "Code") -> str:
    tag = get_step_tag(step)
    step_for_render = dict(step)
    step_for_render["tag"] = tag
    return render_step_for_schema(step_for_render, schema=schema, code_heading=code_heading)


def canonicalize_steps(
    steps: list[dict[str, Any]],
    start_step_number: int,
    schema: str = "markdown",
    code_heading: str = "Code",
) -> list[dict[str, Any]]:
    canonical_steps = []
    for offset, step in enumerate(steps):
        normalized_step = deepcopy(step)
        normalized_step["step_number"] = start_step_number + offset
        if normalized_step.get("type") != "tool":
            normalized_step["reasoning"] = clean_step_reasoning(normalized_step.get("reasoning", ""))
        normalized_step["block_text"] = render_canonical_step(
            normalized_step,
            normalized_step["step_number"],
            schema=schema,
            code_heading=code_heading,
        )
        canonical_steps.append(normalized_step)
    return canonical_steps


def format_steps(steps: list[dict[str, Any]], schema: str = "markdown", code_heading: str = "Code") -> str:
    results = []
    for step in steps:
        block_text = render_canonical_step(
            step,
            step.get("step_number", 0),
            schema=schema,
            code_heading=code_heading,
        )
        results.append(block_text)
    return "\n\n".join(result for result in results if result.strip())

def extract_executable_python(text: str) -> str:
    code = extract_python_code_block(text)
    if code is not None:
        return code.strip()
    return text.strip()


def parse_code_content(text: str) -> str:
    code = extract_executable_python(text)
    if not code:
        return ""
    code_lines = code.split("\n")
    new_lines = []
    for line in code_lines:
        if line.strip().startswith("#"):
            continue
        if line.startswith("print("):
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


def sanitize_search_query(text: str) -> str:
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- Output:"):
            break
        cleaned_lines.append(line)
        break
    if cleaned_lines:
        return cleaned_lines[0]
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def build_continue_prompt(next_step_number: int, schema: str = "markdown", code_heading: str = "Code") -> str:
    schema = resolve_schema(schema)
    code_heading = resolve_code_heading(code_heading)
    if schema == "json":
        return (
            "\n\n<CONTINUE>\n"
            "Output only the next JSON action object(s), one per line.\n"
            "Use action values from: reasoning, code, search, askuser, final.\n"
            "Do not repeat or explain these instructions.\n"
        )
    return (
        "\n\n<CONTINUE>\n"
        "Output only the next step(s).\n"
        f"Start immediately with one of: ### Reasoning, ### Search, ### {code_heading}, ### AskUser, ### Final Response.\n"
        "Do not repeat or explain these instructions.\n"
    )

def infer_domain(data_path: Path, domain_override: str | None) -> str:
    if domain_override is not None:
        return domain_override
    name = data_path.name.lower()
    if "math" in name or "gsm" in name or "mint" in name:
        return "math"
    if "time" in name:
        return "time"
    if "intention" in name:
        return "intention"
    raise ValueError("Could not infer domain from --data-path; please set --domain explicitly.")


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


def load_model_and_tokenizer(model_name_or_path: str, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    use_cuda = device in {"auto", "cuda"} and torch.cuda.is_available()
    target_device = torch.device("cuda" if use_cuda else "cpu")
    tokenizer = load_tokenizer_with_compat(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"dtype": "auto"}
    use_device_map = device == "auto" and use_cuda and importlib.util.find_spec("accelerate") is not None
    if use_device_map:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if not use_device_map:
        model = model.to(target_device)
    model.eval()
    return tokenizer, model, target_device


def get_transformer_layers(model: AutoModelForCausalLM) -> list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported model architecture: could not locate transformer layers.")


def resolve_layer_map(layer_indices: list[int], num_model_layers: int) -> dict[int, int]:
    if not layer_indices:
        raise ValueError("Steering payload does not contain any layer indices.")
    if min(layer_indices) == 1 and max(layer_indices) == num_model_layers:
        return {layer_index - 1: idx for idx, layer_index in enumerate(layer_indices)}
    if min(layer_indices) == 0 and max(layer_indices) == num_model_layers - 1:
        return {layer_index: idx for idx, layer_index in enumerate(layer_indices)}
    if 0 in layer_indices:
        raise ValueError("Embedding-layer steering is not supported during inference hooks.")
    if max(layer_indices) < num_model_layers:
        return {layer_index: idx for idx, layer_index in enumerate(layer_indices)}
    raise ValueError(
        f"Could not map steering layer indices {layer_indices[:5]}... to a model with {num_model_layers} layers."
    )


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
                raise ValueError(f"Invalid steering layer range {value!r}: start must be <= end.")
            candidates = range(start, end + 1)
        else:
            candidates = [int(value)]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)
    if not selected:
        raise ValueError("--steering_layers cannot be empty when provided.")
    return selected


def load_steering_payload(path: Path) -> tuple[torch.Tensor, list[int]]:
    payload = torch.load(path, map_location="cpu")
    if "steering_vectors" not in payload:
        raise ValueError(f"Steering payload at {path} is missing 'steering_vectors'.")
    steering_vectors = payload["steering_vectors"]
    layer_indices = payload.get("layer_indices")
    if layer_indices is None:
        raise ValueError(f"Steering payload at {path} is missing 'layer_indices'.")
    return steering_vectors, list(layer_indices)


def select_top_layers_by_vector_norm(
    layer_map: dict[int, int],
    steering_vectors: torch.Tensor,
    max_steering_layers: int,
) -> dict[str, Any]:
    scored_pairs = []
    for model_layer_idx, vector_idx in layer_map.items():
        norm = torch.linalg.vector_norm(steering_vectors[vector_idx].float()).item()
        scored_pairs.append((model_layer_idx, vector_idx, norm))
    scored_pairs.sort(key=lambda item: item[2], reverse=True)
    selected = sorted(scored_pairs[:max_steering_layers], key=lambda item: item[0])
    selected_pairs = [(model_layer_idx, vector_idx) for model_layer_idx, vector_idx, _ in selected]
    return {
        "selected_pairs": selected_pairs,
        "selected_layers": [layer_idx for layer_idx, _ in selected_pairs],
    }


def select_pairs_for_explicit_saved_layers(
    layer_indices: list[int],
    layer_map: dict[int, int],
    steering_layer_spec: str,
) -> dict[str, Any]:
    requested_saved_layers = parse_layer_spec(steering_layer_spec)
    saved_to_pair = {
        layer_indices[vector_idx]: (model_layer_idx, vector_idx)
        for model_layer_idx, vector_idx in layer_map.items()
    }
    missing = [layer_id for layer_id in requested_saved_layers if layer_id not in saved_to_pair]
    if missing:
        raise ValueError(
            f"Requested steering layers {missing} are not available in the payload. "
            f"Available payload layers: {min(layer_indices)}-{max(layer_indices)}"
        )
    selected_pairs = [saved_to_pair[layer_id] for layer_id in requested_saved_layers]
    return {
        "selected_pairs": selected_pairs,
        "selected_layers": [model_layer_idx for model_layer_idx, _ in selected_pairs],
        "selected_saved_layers": requested_saved_layers,
    }


def has_open_heading_prefix(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith("###")


def has_open_schema_trigger_prefix(text: str, schema: str) -> bool:
    return has_schema_trigger_prefix(text, schema=schema)


class TagTriggeredSuppressionHookManager:
    """Apply suppress-only steering on the last position once generation enters a `###` heading prefix."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        steering_vectors: torch.Tensor,
        selected_pairs: list[tuple[int, int]],
        strength: float,
    ) -> None:
        self.scale = 0.0
        self.handles: list[Any] = []
        self.strength = strength
        self.layers = get_transformer_layers(model)
        self.vectors = steering_vectors.cpu()

        for model_layer_idx, vector_idx in selected_pairs:
            vector = self.vectors[vector_idx]
            handle = self.layers[model_layer_idx].register_forward_hook(self._make_hook(vector))
            self.handles.append(handle)

    def set_scale(self, scale: float) -> None:
        self.scale = scale

    def _make_hook(self, vector: torch.Tensor):
        def hook(_module, _inputs, output):
            if self.scale == 0.0:
                return output

            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return output

            delta = vector.to(device=hidden.device, dtype=hidden.dtype) * self.scale * self.strength
            if hidden.shape[1] > 0:
                hidden = hidden.clone()
                hidden[:, -1, :] -= delta

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def greedy_generate_with_tag_triggered_suppression_kv_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    hook_manager: TagTriggeredSuppressionHookManager,
    suppress_scale: float,
    max_new_tokens: int,
    selected_pairs: list[tuple[int, int]],
    steering_vectors: torch.Tensor,
    layer_indices: list[int],
    eval_schema: str = "markdown",
) -> tuple[str, list[dict[str, Any]]]:
    generated_text = ""
    diagnostics: list[dict[str, Any]] = []
    generated_ids: list[torch.Tensor] = []
    attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
    current_input_ids = prompt_ids
    past_key_values = None

    for generation_step in range(max_new_tokens):
        should_suppress = has_open_schema_trigger_prefix(generated_text, eval_schema)
        hook_manager.set_scale(suppress_scale if should_suppress else 0.0)
        with torch.inference_mode():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=should_suppress,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_token)

        next_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
        generated_text += next_text
        trigger_info = None
        if should_suppress and outputs.hidden_states is not None:
            trigger_info = build_trigger_info(
                generated_text=generated_text,
                hidden_states=outputs.hidden_states,
                selected_pairs=selected_pairs,
                steering_vectors=steering_vectors,
                layer_indices=layer_indices,
            )

        diagnostics.append(
            {
                "generation_step": generation_step + 1,
                "suppression_active": should_suppress,
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


def greedy_generate_without_steering_kv_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, list[dict[str, Any]]]:
    generated_text = ""
    diagnostics: list[dict[str, Any]] = []
    generated_ids: list[torch.Tensor] = []
    attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
    current_input_ids = prompt_ids
    past_key_values = None

    for generation_step in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_token)

        next_text = tokenizer.decode(next_token[0], skip_special_tokens=False)
        generated_text += next_text
        diagnostics.append(
            {
                "generation_step": generation_step + 1,
                "suppression_active": False,
                "steering_disabled": True,
                "generated_fragment_tail": generated_text[-40:],
                "trigger_info": None,
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

    if generated_ids:
        output_ids = torch.cat(generated_ids, dim=1)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    else:
        output_text = ""
    return output_text, diagnostics


def inference(args: argparse.Namespace) -> None:
    print("Loading model and tokenizer...")
    tokenizer, model, device = load_model_and_tokenizer(args.model_name_or_path, args.device)

    domain = infer_domain(args.data_path, args.domain)
    eval_schema = resolve_schema(args.eval_schema, fallback=args.schema or "markdown")
    extract_schema = resolve_schema(args.extract_schema, fallback="markdown")
    code_heading = resolve_code_heading(args.code_heading)

    print("Loading and preprocessing dataset...")
    dataset = preprocess_dataset(
        data_path=args.data_path,
        max_num=args.max_test_num,
        start_id=args.test_start_id,
        method=args.method,
        domain=domain,
        eval_schema=eval_schema,
        code_heading=code_heading,
        prompt_policy=args.prompt_policy,
        heading_interface=args.heading_interface,
    )

    tool_group = DOMAIN_TO_TOOL_GROUP[domain]
    steering_payload_name = args.steering_payload_name or f"step_mark_{tool_group}.pt"
    selected_pairs: list[tuple[int, int]] = []
    selected_layers: list[int] = []
    selected_saved_layers: list[int] = []
    steering_vectors: torch.Tensor | None = None
    layer_indices: list[int] = []
    if args.disable_steering:
        print(f"Using domain={domain} -> tool_group={tool_group}")
        print("Steering disabled; no steering payload will be loaded.")
    else:
        steering_vector_path = args.steering_vector_dir / steering_payload_name
        print(f"Using domain={domain} -> tool_group={tool_group}")
        print(f"Loading steering payload from {steering_vector_path}")
        steering_vectors, layer_indices = load_steering_payload(steering_vector_path)
        layer_map = resolve_layer_map(layer_indices, len(get_transformer_layers(model)))
        if args.steering_layers is not None:
            selection_summary = select_pairs_for_explicit_saved_layers(
                layer_indices=layer_indices,
                layer_map=layer_map,
                steering_layer_spec=args.steering_layers,
            )
        else:
            selection_summary = select_top_layers_by_vector_norm(
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

    hook_manager = None
    if not args.disable_steering:
        if steering_vectors is None:
            raise RuntimeError("Steering vectors were not loaded for a steered run.")
        hook_manager = TagTriggeredSuppressionHookManager(
            model=model,
            steering_vectors=steering_vectors,
            selected_pairs=selected_pairs,
            strength=args.steering_strength,
        )

    log = {"fail": 0, "success": 0}
    example_count = 0
    skipped_count = 0
    tool_usage_counts: dict[str, int] = defaultdict(int)

    progress = tqdm(dataset, desc="Inference", dynamic_ncols=True) if tqdm is not None else dataset

    try:
        for example in progress:
            input_messages = deepcopy(example["input"])
            ground_truth = example["ground_truth"]
            task = example["task"]

            example_count += 1
            code_file = args.code_exec_dir / f"{example_count}_{task[:3]}.py"

            if task in existing_tasks or task in existing_error_tasks:
                skipped_count += 1
                if tqdm is not None:
                    progress.set_postfix(build_progress_postfix(log, skipped_count, tool_usage_counts), refresh=False)
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
                                "reasoning": TIMEOUT_FALLBACK,
                                "block_text": TIMEOUT_FALLBACK,
                            }
                        )
                        results.append(
                            build_result_record(
                                task=task,
                                ground_truth=ground_truth,
                                steps=steps,
                                raw=raw,
                                steering_diagnostics=steering_diagnostics,
                                selected_layers=selected_layers,
                            )
                        )
                        save_json(args.save_path, results)
                        log_example_tool_summary("timeout", step_time, task_tool_usage_counts, tool_usage_counts, args.quiet)
                        if tqdm is not None:
                            progress.set_postfix(
                                build_progress_postfix(
                                    log,
                                    skipped_count,
                                    tool_usage_counts,
                                    step_time=step_time,
                                    task_tool_usage_counts=task_tool_usage_counts,
                                ),
                                refresh=False,
                            )
                        break

                    prompt_ids = build_prompt_ids(tokenizer, input_messages, device, args.max_seq_length)
                    if args.disable_steering:
                        assistant_output, token_diagnostics = greedy_generate_without_steering_kv_cache(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_ids=prompt_ids,
                            max_new_tokens=args.max_new_tokens,
                        )
                    else:
                        if hook_manager is None or steering_vectors is None:
                            raise RuntimeError("Steering state was not initialized for a steered run.")
                        assistant_output, token_diagnostics = greedy_generate_with_tag_triggered_suppression_kv_cache(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_ids=prompt_ids,
                            hook_manager=hook_manager,
                            suppress_scale=args.suppress_scale,
                            max_new_tokens=args.max_new_tokens,
                            selected_pairs=selected_pairs,
                            steering_vectors=steering_vectors,
                            layer_indices=layer_indices,
                            eval_schema=eval_schema,
                        )
                    raw.append(assistant_output)
                    schema_parse_diagnostics = (
                        json_action_parse_diagnostics(strip_generation_artifacts(assistant_output))
                        if eval_schema == "json"
                        else None
                    )
                    steering_diagnostics.append(
                        {
                            "step": step_time,
                            "gating_mode": (
                                "disabled"
                                if args.disable_steering
                                else "tag_triggered_domain_specific" if args.steering_payload_name is None else "tag_triggered_custom_payload"
                            ),
                            "domain": domain,
                            "tool_group": tool_group,
                            "steering_payload_name": steering_payload_name,
                            "selected_layers": selected_layers,
                            "selected_saved_layers": selected_saved_layers,
                            "suppress_scale": args.suppress_scale,
                            "extract_schema": extract_schema,
                            "eval_schema": eval_schema,
                            "code_heading": code_heading,
                            "ablation": args.ablation,
                            "prompt_policy": args.prompt_policy,
                            "disable_steering": args.disable_steering,
                            "schema_parse_diagnostics": schema_parse_diagnostics,
                            "token_diagnostics": token_diagnostics if args.token_diagnostics_mode == "full" else None,
                            "token_diagnostics_summary": summarize_token_diagnostics(token_diagnostics),
                        }
                    )

                    log_message("\n" + "+" * 10 + " Round Response " + "+" * 10, args.quiet)
                    log_message(assistant_output, args.quiet)

                    new_steps = canonicalize_steps(
                        normalize_round_steps(assistant_output, schema=eval_schema, code_heading=code_heading),
                        start_step_number=len(steps) + 1,
                        schema=eval_schema,
                        code_heading=code_heading,
                    )

                    first_tool_idx = next(
                        (idx for idx, step in enumerate(new_steps) if step.get("type") == "tool"),
                        None,
                    )
                    if first_tool_idx is not None and first_tool_idx + 1 < len(new_steps):
                        # Any content after the first tool was generated without seeing the tool output,
                        # so keep only the executable prefix and let the model continue next round.
                        new_steps = new_steps[: first_tool_idx + 1]

                    if new_steps and new_steps[-1]["type"] == "tool":
                        tool_name = new_steps[-1]["tool_name"]
                        tool_usage_counts[tool_name] += 1
                        task_tool_usage_counts[tool_name] += 1
                        if tool_name == "AskUser":
                            response = simulate_user_response(task, new_steps[-1]["reasoning"])
                            new_steps[-1]["output"] = response
                        elif tool_name == "Search":
                            link = "intention" in str(args.data_path)
                            search_query = sanitize_search_query(new_steps[-1]["reasoning"])
                            new_steps[-1]["reasoning"] = search_query
                            new_steps[-1]["block_text"] = render_canonical_step(new_steps[-1], new_steps[-1]["step_number"], schema=eval_schema, code_heading=code_heading)
                            response = search_serper(search_query, link=link, num=args.search_top_k)
                            new_steps[-1]["output"] = response
                        elif tool_name == "Code":
                            code_body = extract_executable_python(new_steps[-1]["reasoning"])
                            if not code_body:
                                response = "Error: No valid Python code found."
                            else:
                                code_content = f"```python\n{all_previous_code}\n{code_body}\n```"
                                response = execute_code(code_content, code_file)
                            if not response.startswith("Error"):
                                new_code = parse_code_content(new_steps[-1]["reasoning"])
                                if new_code:
                                    all_previous_code += new_code + "\n"
                            new_steps[-1]["output"] = response
                        else:
                            raise AssertionError(f"Unknown tool name: {tool_name}")
                        new_steps[-1]["block_text"] = render_canonical_step(
                            new_steps[-1],
                            new_steps[-1]["step_number"],
                            schema=eval_schema,
                            code_heading=code_heading,
                        )
                        log_tool_usage_update(tool_name, step_time, task_tool_usage_counts, tool_usage_counts, args.quiet)
                        log_message("\n" + "+" * 10 + f" Tool {tool_name} Response " + "+" * 10, args.quiet)
                        log_message(response, args.quiet)
                        if tqdm is not None:
                            progress.set_postfix(
                                build_progress_postfix(
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
                            build_result_record(
                                task=task,
                                ground_truth=ground_truth,
                                steps=steps,
                                raw=raw,
                                steering_diagnostics=steering_diagnostics,
                                selected_layers=selected_layers,
                            )
                        )
                        save_json(args.save_path, results)
                        log_example_tool_summary("success", step_time, task_tool_usage_counts, tool_usage_counts, args.quiet)
                        if tqdm is not None:
                            progress.set_postfix(
                                {
                                    "ok": log["success"],
                                    "fail": log["fail"],
                                    "step": step_time,
                                    "tools_total": format_tool_usage_counts(tool_usage_counts),
                                    "tools_task": format_tool_usage_counts(task_tool_usage_counts),
                                },
                                refresh=False,
                            )
                        break

                    input_messages[-1]["content"] = (
                        input_messages[-1]["content"].strip()
                        + "\n"
                        + format_steps(new_steps, schema=eval_schema, code_heading=code_heading).strip()
                        + build_continue_prompt(len(steps) + len(new_steps) + 1, schema=eval_schema, code_heading=code_heading)
                    )
                    steps.extend(new_steps)

                except Exception as error:  # pragma: no cover - runtime path
                    log["fail"] += 1
                    error_results.append(
                        build_error_record(
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
                    save_json(error_save_path, error_results)
                    log_example_tool_summary("error", step_time, task_tool_usage_counts, tool_usage_counts, args.quiet)
                    log_message(str(error), args.quiet)
                    if tqdm is not None:
                        progress.set_postfix(
                            build_progress_postfix(
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
        if hook_manager is not None:
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
    parser.add_argument(
        "--steering_payload_name",
        type=str,
        default=None,
        help="Optional steering payload filename inside --steering_vector_dir, e.g. 'step_mark_all.pt'. Overrides domain-specific payload selection.",
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
    parser.add_argument(
        "--prompt-policy",
        choices=("base_tool", "conservative_tool"),
        default="base_tool",
        help="Natural-language tool-use policy while preserving the same heading-anchor interface.",
    )
    parser.add_argument(
        "--heading-interface",
        choices=("default", "matched_prompt"),
        default="default",
        help="Tool headings exposed in the prompt. matched_prompt uses one target tool per domain.",
    )
    parser.add_argument(
        "--schema",
        choices=SCHEMAS,
        default=None,
        help="Alias for --eval-schema. Defaults to markdown.",
    )
    parser.add_argument(
        "--extract-schema",
        choices=SCHEMAS,
        default="markdown",
        help="Schema metadata for the loaded steering vector.",
    )
    parser.add_argument(
        "--eval-schema",
        choices=SCHEMAS,
        default=None,
        help="Generation/output schema used during evaluation. Defaults to --schema or markdown.",
    )
    parser.add_argument(
        "--code-heading",
        choices=CODE_HEADINGS,
        default="Code",
        help="Markdown heading name that represents the code tool during evaluation.",
    )
    parser.add_argument(
        "--ablation",
        choices=("none", "cross_format", "heading_rename", "matched_prompt"),
        default="none",
        help="Optional ablation metadata saved into diagnostics.",
    )
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
        "--disable-steering",
        action="store_true",
        help="Run the same tool loop and prompt format without loading or applying activation steering.",
    )
    parser.add_argument("--search_top_k", type=int, default=3, help="Top-k web results for Search")
    parser.add_argument(
        "--token_diagnostics_mode",
        choices=("summary", "full"),
        default="summary",
        help="How much per-token suppression trace to save. Default keeps only a compact summary.",
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
