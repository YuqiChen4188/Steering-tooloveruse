#!/usr/bin/env python3
"""Build math single-turn GPT datasets with paired no-tool and code-tool outputs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from tqdm import tqdm


SOURCE_PATH = Path(
    "/data/yuqi/Steering-tooloveruse/data_train/base_reasoning_raw/domain_math_base_reasoning_raw.json"
)
DEFAULT_OUTPUT_DIR = Path("/data/yuqi/Steering-tooloveruse/steering_data/math_single_turn_gpt")

NO_TOOL_SYSTEM_PROMPT_C2 = """You are an advanced assistant designed to solve tasks autonomously using your knowledge and reasoning. Clearly articulate your thought process and reasoning steps before presenting the final response to ensure transparency and accuracy.
In the field '### Reasoning Steps', clearly articulate your thought process and reasoning steps towards the final answer. Then you should present a succinct and accurate final response in the field '### Final Response'."""

CODE_TOOL_SINGLE_TURN_SYSTEM_PROMPT_A2 = """### Task
You are a highly capable assistant designed to solve tasks effectively using your knowledge and the available code tool.
### Principles
1. Reason Independently:
• Leverage your own knowledge to analyze and solve reasoning steps whenever possible. Use the code tool only when necessary.
2. Tool Usage:
• Use code snippet ```python ... ``` to write, execute a Python code snippet, and retrieve the result from its printed output.
3. Step-by-Step Approach:
• Work through reasoning systematically, breaking down the task into manageable steps. Rely on your knowledge until a gap is identified that requires code support. Employ the code tool to address the gap and integrate the result into your solution.
4. Goal-Oriented Resolution:
• Conclude your reasoning process by achieving a clear, accurate, and succinct solution based on your independent analysis and the code result.
### Output Guidelines
• Continue directly after "### Reasoning Steps".
• If you use the code tool, include exactly one executable ```python``` block with all required imports.
• Complete the whole answer in a single response. Do not stop early or wait for another round.
• After the code block, include exactly one line in the format "- Tool Output: <result>".
• Finally give a succinct and accurate final response after "### Final Response".
• Do not mention or use any tool other than the code tool."""

CODE_TOOL_STAGE1_DRAFT_SYSTEM_PROMPT = """### Task
You are preparing a code-assisted math solution draft.
### Principles
1. Reason briefly and only as needed to set up the computation.
2. Use exactly one executable ```python``` block to perform the needed calculation.
3. Do not mention or use Search, AskUser, browsing, or any tool other than the Python code block.
### Output Guidelines
• Continue directly after "### Reasoning Steps".
• Include at most a short setup explanation before the code block.
• Include exactly one executable ```python``` block with all required imports.
• Print the value needed for the solution.
• Stop immediately after the closing ``` of the Python block.
• Do not include "- Tool Output:".
• Do not include "### Final Response"."""

CODE_TOOL_STAGE2_FINALIZER_SYSTEM_PROMPT = """You are formatting a completed single-turn training sample for a math problem.
Follow these rules exactly:
1. Continue directly after "### Reasoning Steps".
2. Include concise reasoning before and/or after the code when helpful.
3. Include exactly one Python code block, and copy it exactly from the provided code snippet.
4. Include exactly one line formatted as "- Tool Output: <result>" using the provided tool output string exactly.
5. Do not mention Search, AskUser, browsing, or any tool other than the Python code block.
6. End with "### Final Response" followed by the final answer.
7. Output only the final sample text."""

INPUT_TEMPLATE = "### Task\n{problem}\n\n### Reasoning Steps\n"
FINAL_RESPONSE_PATTERNS = (
    r"###\s*Final Response\b",
    r"Final Response\s*:",
    r"Final Answer\s*:",
    r"The final answer is\s*:?",
    r"The answer is\s*:?",
)
DEFAULT_PREAMBLE = """import math
import cmath
import statistics
import itertools
import functools
import collections
import heapq
import bisect
from fractions import Fraction
from decimal import Decimal, getcontext
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate math single-turn GPT datasets with paired no-tool and code-tool outputs."
        )
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pair outputs, raw outputs, and build summary.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for each generation stage.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="Seconds to sleep between failed attempts.",
    )
    parser.add_argument(
        "--code-timeout-sec",
        type=float,
        default=20.0,
        help="Timeout in seconds for executing generated Python code.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, rebuild outputs from scratch instead of resuming partial outputs.",
    )
    return parser.parse_args()


def load_secret() -> dict[str, Any]:
    secret_path = Path(__file__).resolve().parents[1] / "secret.json"
    with secret_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_client() -> OpenAI:
    secret = load_secret()
    return OpenAI(api_key=secret["api_key"], base_url=secret["base_url"])


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json_dict(path: Path, overwrite: bool) -> dict[str, Any]:
    if overwrite or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return data


def load_source_dataset() -> dict[str, Any]:
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON at {SOURCE_PATH}")
    return data


def extract_problem_text(record: dict[str, Any]) -> str:
    data = record.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Expected record['data'] to be a dict.")
    problem = data.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Expected a non-empty data.problem field.")
    return problem.strip()


def build_messages(system_prompt: str, user_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]


def call_model_once(client: OpenAI, messages: list[dict[str, str]], model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.000001,
        n=1,
    )
    return (response.choices[0].message.content or "").strip()


def normalize_text(text: str) -> str:
    text = text.replace("** Input **", "")
    text = text.replace("** Output **", "")
    text = text.strip()
    if text.startswith("### Reasoning Steps"):
        text = text[len("### Reasoning Steps") :].lstrip(" \n\t:")
    return text.strip()


def has_disallowed_tool_marker(text: str) -> bool:
    return "AskUser(" in text or "Search(" in text


def count_python_code_blocks(text: str) -> int:
    return len(re.findall(r"```python\s*\n.*?```", text, flags=re.DOTALL))


def extract_python_code(text: str) -> str | None:
    matches = re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    return matches[0].strip()


def find_final_response_match(text: str) -> re.Match[str] | None:
    for pattern in FINAL_RESPONSE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return match
    return None


def display_tool_output(text: str) -> str:
    return text.replace("\r\n", "\n").rstrip("\n").replace("\n", "\\n").strip()


def execute_python_code(code: str, timeout_sec: float) -> str:
    code_to_run = DEFAULT_PREAMBLE + "\n" + code.strip() + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        prefix="math_single_turn_",
        dir="/tmp",
        delete=False,
    ) as handle:
        handle.write(code_to_run)
        temp_path = Path(handle.name)

    try:
        result = subprocess.run(
            ["python", str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Generated code exceeded the {timeout_sec:g}-second timeout."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "unknown execution error"
        raise RuntimeError(f"Generated code failed to execute: {stderr}") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    output = result.stdout.replace("\r\n", "\n").strip()
    if not output:
        raise RuntimeError("Generated code did not print any output.")
    return output


def generate_with_retries(
    *,
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    max_retries: int,
    retry_sleep: float,
    validator: Callable[[str], str | None],
) -> tuple[str, int]:
    last_error = "unknown generation error"
    for attempt in range(1, max_retries + 1):
        try:
            response_text = call_model_once(client=client, messages=messages, model=model)
            failure_reason = validator(response_text)
            if failure_reason is None:
                return response_text, attempt
            last_error = failure_reason
        except Exception as exc:
            last_error = str(exc)
        if attempt < max_retries:
            time.sleep(retry_sleep)
    raise RuntimeError(last_error)


def validate_no_tool_output(text: str) -> str | None:
    normalized = normalize_text(text)
    if has_disallowed_tool_marker(normalized):
        return "No-tool output mentioned Search or AskUser."
    if count_python_code_blocks(normalized) > 0:
        return "No-tool output included a Python code block."
    if find_final_response_match(normalized) is None:
        return "No-tool output did not contain ### Final Response."
    return None


def validate_stage1_code_output(text: str) -> str | None:
    normalized = normalize_text(text)
    if has_disallowed_tool_marker(normalized):
        return "Stage-1 code output mentioned Search or AskUser."
    if count_python_code_blocks(normalized) != 1:
        return "Stage-1 code output must contain exactly one Python code block."
    if "- Tool Output:" in normalized:
        return "Stage-1 code output must not include tool output text."
    if find_final_response_match(normalized) is not None:
        return "Stage-1 code output must not include a final response."
    return None


def validate_code_final_output(text: str) -> str | None:
    normalized = normalize_text(text)
    if has_disallowed_tool_marker(normalized):
        return "Final code-tool output mentioned Search or AskUser."
    if count_python_code_blocks(normalized) != 1:
        return "Final code-tool output must contain exactly one Python code block."
    if "- Tool Output:" not in normalized:
        return "Final code-tool output is missing - Tool Output:."
    if find_final_response_match(normalized) is None:
        return "Final code-tool output did not contain ### Final Response."
    return None


def split_reasoning_and_final(text: str) -> tuple[str, str]:
    normalized = normalize_text(text)
    match = find_final_response_match(normalized)
    if match is None:
        raise ValueError("Output does not contain a final response marker.")
    reasoning_text = normalized[: match.start()].strip()
    final_text = normalized[match.end() :].lstrip(" \n\t:").strip()
    if not final_text:
        raise ValueError("Output final response is empty.")
    return reasoning_text, final_text


def split_paragraphs(text: str) -> list[str]:
    if not text.strip():
        return []
    chunks = re.split(r"\n\s*\n+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def build_no_tool_pair(instruction: str, problem: str, output_text: str) -> dict[str, str]:
    return {
        "instruction": instruction,
        "input": INPUT_TEMPLATE.format(problem=problem),
        "output": normalize_text(output_text),
    }


def convert_no_tool_output_to_raw(
    output_text: str,
    original_record: dict[str, Any],
) -> dict[str, Any]:
    reasoning_text, final_text = split_reasoning_and_final(output_text)
    steps: list[dict[str, Any]] = []
    for index, chunk in enumerate(split_paragraphs(reasoning_text), start=1):
        steps.append(
            {
                "step": index,
                "name": f"Reasoning Step {index}",
                "type": "normal",
                "reasoning": chunk,
            }
        )
    steps.append(
        {
            "step": len(steps) + 1,
            "name": "Final Response",
            "type": "normal",
            "reasoning": final_text,
        }
    )
    return {
        "data": original_record["data"],
        "tool_involved": False,
        "thought": reasoning_text if reasoning_text else final_text,
        "reasoning_complete": steps,
    }


def convert_code_output_to_raw(
    output_text: str,
    original_record: dict[str, Any],
    executed_output: str,
) -> dict[str, Any]:
    reasoning_text, final_text = split_reasoning_and_final(output_text)
    code_match = re.search(r"```python\s*\n(.*?)```", reasoning_text, flags=re.DOTALL)
    if code_match is None:
        raise ValueError("Code-tool output is missing its Python code block.")

    code = code_match.group(1).strip()
    before_code = reasoning_text[: code_match.start()].strip()
    after_code = reasoning_text[code_match.end() :].strip()
    tool_output_line_match = re.search(
        r"(?m)^- Tool Output:\s*(.*)$",
        after_code,
    )
    if tool_output_line_match is None:
        raise ValueError("Code-tool output is missing the tool output section.")

    before_tool_output = after_code[: tool_output_line_match.start()].strip()
    after_tool_output = after_code[tool_output_line_match.end() :].strip()

    steps: list[dict[str, Any]] = []
    step_id = 1
    normal_reasoning_chunks = []

    for chunk in split_paragraphs(before_code):
        normal_reasoning_chunks.append(chunk)
        steps.append(
            {
                "step": step_id,
                "name": f"Reasoning Step {step_id}",
                "type": "normal",
                "reasoning": chunk,
            }
        )
        step_id += 1

    steps.append(
        {
            "step": step_id,
            "name": "Code Tool Step",
            "type": "code",
            "reasoning": code,
            "simulate_response": executed_output,
        }
    )
    step_id += 1

    if before_tool_output:
        normal_reasoning_chunks.append(before_tool_output)
        steps.append(
            {
                "step": step_id,
                "name": f"Reasoning Step {step_id}",
                "type": "normal",
                "reasoning": before_tool_output,
            }
        )
        step_id += 1

    for chunk in split_paragraphs(after_tool_output):
        normal_reasoning_chunks.append(chunk)
        steps.append(
            {
                "step": step_id,
                "name": f"Reasoning Step {step_id}",
                "type": "normal",
                "reasoning": chunk,
            }
        )
        step_id += 1

    steps.append(
        {
            "step": step_id,
            "name": "Final Response",
            "type": "normal",
            "reasoning": final_text,
        }
    )

    thought = "\n\n".join(normal_reasoning_chunks).strip() or final_text
    return {
        "data": original_record["data"],
        "tool_involved": True,
        "thought": thought,
        "reasoning_complete": steps,
    }


def raw_record_to_pair(record: dict[str, Any], instruction: str, problem: str) -> dict[str, str]:
    parts: list[str] = []
    for step in record.get("reasoning_complete", []):
        if step["name"] == "Final Response":
            continue
        step_type = step.get("type", "normal")
        if step_type == "code":
            code = step.get("reasoning", "").strip()
            parts.append(f"```python\n{code}\n```")
            output_text = display_tool_output(str(step.get("simulate_response", "")).strip())
            parts.append(f"- Tool Output: {output_text}")
        else:
            reasoning = str(step.get("reasoning", "")).strip()
            if reasoning:
                parts.append(reasoning)

    final_text = ""
    for step in record.get("reasoning_complete", []):
        if step["name"] == "Final Response":
            final_text = str(step.get("reasoning", "")).strip()
            break
    if not final_text:
        raise ValueError("Raw record is missing a final response step.")

    reasoning_prefix = "\n\n".join(part for part in parts if part).strip()
    if reasoning_prefix:
        output = f"{reasoning_prefix}\n\n### Final Response\n{final_text}"
    else:
        output = f"### Final Response\n{final_text}"

    return {
        "instruction": instruction,
        "input": INPUT_TEMPLATE.format(problem=problem),
        "output": output,
    }


def build_pairs_from_raw(
    task_ids: list[str],
    source_dataset: dict[str, Any],
    raw_records: dict[str, Any],
    instruction: str,
) -> list[dict[str, str]]:
    pairs = []
    for task_id in task_ids:
        record = raw_records.get(task_id)
        if record is None:
            continue
        problem = extract_problem_text(source_dataset[task_id])
        pairs.append(raw_record_to_pair(record=record, instruction=instruction, problem=problem))
    return pairs


def build_summary_payload(
    *,
    args: argparse.Namespace,
    task_ids: list[str],
    no_tool_raw: dict[str, Any],
    code_tool_raw: dict[str, Any],
    no_tool_stats: dict[str, Any],
    code_tool_stats: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "model": args.model,
        "source_path": str(SOURCE_PATH),
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "input_template": INPUT_TEMPLATE,
        "code_timeout_sec": args.code_timeout_sec,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "prompts": {
            "no_tool_system_prompt_c2": NO_TOOL_SYSTEM_PROMPT_C2,
            "code_tool_single_turn_system_prompt_a2": CODE_TOOL_SINGLE_TURN_SYSTEM_PROMPT_A2,
            "code_tool_stage1_draft_system_prompt": CODE_TOOL_STAGE1_DRAFT_SYSTEM_PROMPT,
            "code_tool_stage2_finalizer_system_prompt": CODE_TOOL_STAGE2_FINALIZER_SYSTEM_PROMPT,
        },
        "outputs": {
            "domain_math_no_tool_pairs": str(output_dir / "domain_math_no_tool_pairs.json"),
            "domain_math_code_tool_pairs": str(output_dir / "domain_math_code_tool_pairs.json"),
            "domain_math_no_tool_raw": str(output_dir / "domain_math_no_tool_raw.json"),
            "domain_math_code_tool_raw": str(output_dir / "domain_math_code_tool_raw.json"),
        },
        "no_tool": {
            "completed_count": len(no_tool_raw),
            "failed_count": len(no_tool_stats["failed_ids"]),
            "failed_ids": no_tool_stats["failed_ids"],
            "retry_attempts_total": no_tool_stats["retry_attempts_total"],
            "validation_failures": dict(no_tool_stats["validation_failures"]),
            "fallback_count": 0,
            "fallback_ids": [],
        },
        "code_tool": {
            "completed_count": len(code_tool_raw),
            "failed_count": len(code_tool_stats["failed_ids"]),
            "failed_ids": code_tool_stats["failed_ids"],
            "retry_attempts_total": code_tool_stats["retry_attempts_total"],
            "validation_failures": dict(code_tool_stats["validation_failures"]),
            "execution_failures": dict(code_tool_stats["execution_failures"]),
            "fallback_count": 0,
            "fallback_ids": [],
        },
    }


def save_all_outputs(
    *,
    output_dir: Path,
    task_ids: list[str],
    source_dataset: dict[str, Any],
    no_tool_raw: dict[str, Any],
    code_tool_raw: dict[str, Any],
    no_tool_stats: dict[str, Any],
    code_tool_stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    no_tool_pairs = build_pairs_from_raw(
        task_ids=task_ids,
        source_dataset=source_dataset,
        raw_records=no_tool_raw,
        instruction=NO_TOOL_SYSTEM_PROMPT_C2,
    )
    code_tool_pairs = build_pairs_from_raw(
        task_ids=task_ids,
        source_dataset=source_dataset,
        raw_records=code_tool_raw,
        instruction=CODE_TOOL_SINGLE_TURN_SYSTEM_PROMPT_A2,
    )

    save_json(output_dir / "domain_math_no_tool_raw.json", no_tool_raw)
    save_json(output_dir / "domain_math_code_tool_raw.json", code_tool_raw)
    save_json(output_dir / "domain_math_no_tool_pairs.json", no_tool_pairs)
    save_json(output_dir / "domain_math_code_tool_pairs.json", code_tool_pairs)

    summary = build_summary_payload(
        args=args,
        task_ids=task_ids,
        no_tool_raw=no_tool_raw,
        code_tool_raw=code_tool_raw,
        no_tool_stats=no_tool_stats,
        code_tool_stats=code_tool_stats,
        output_dir=output_dir,
    )
    save_json(output_dir / "domain_math_single_turn_build_summary.json", summary)


def stage1_code_draft_messages(problem: str) -> list[dict[str, str]]:
    return build_messages(
        system_prompt=CODE_TOOL_STAGE1_DRAFT_SYSTEM_PROMPT,
        user_input=INPUT_TEMPLATE.format(problem=problem),
    )


def stage2_code_final_messages(problem: str, code: str, tool_output: str) -> list[dict[str, str]]:
    user_prompt = (
        f"### Problem\n{problem}\n\n"
        "### Code Snippet\n"
        f"```python\n{code}\n```\n\n"
        "### Tool Output\n"
        f"{display_tool_output(tool_output)}\n\n"
        "### Required Format\n"
        "Continue directly after '### Reasoning Steps'. "
        "Use the exact code snippet and exact tool output above."
    )
    return build_messages(
        system_prompt=CODE_TOOL_STAGE2_FINALIZER_SYSTEM_PROMPT,
        user_input=user_prompt,
    )


def build_no_tool_record(
    *,
    client: OpenAI,
    problem: str,
    original_record: dict[str, Any],
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> dict[str, Any]:
    try:
        messages = build_messages(
            system_prompt=NO_TOOL_SYSTEM_PROMPT_C2,
            user_input=INPUT_TEMPLATE.format(problem=problem),
        )
        output_text, attempts = generate_with_retries(
            client=client,
            messages=messages,
            model=args.model,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
            validator=validate_no_tool_output,
        )
        stats["retry_attempts_total"] += attempts - 1
        normalized = normalize_text(output_text)
        return convert_no_tool_output_to_raw(output_text=normalized, original_record=original_record)
    except Exception as exc:
        stats["validation_failures"][str(exc)] += 1
        raise


def build_code_tool_record(
    *,
    client: OpenAI,
    problem: str,
    original_record: dict[str, Any],
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> dict[str, Any]:
    last_error = "unknown code-tool error"
    for attempt in range(1, args.max_retries + 1):
        try:
            stage1_text, _stage1_attempts = generate_with_retries(
                client=client,
                messages=stage1_code_draft_messages(problem),
                model=args.model,
                max_retries=1,
                retry_sleep=args.retry_sleep,
                validator=validate_stage1_code_output,
            )
            code = extract_python_code(normalize_text(stage1_text))
            if not code:
                raise RuntimeError("Stage-1 code output did not contain a valid Python block.")

            executed_output = execute_python_code(code=code, timeout_sec=args.code_timeout_sec)

            stage2_text, _stage2_attempts = generate_with_retries(
                client=client,
                messages=stage2_code_final_messages(problem, code, executed_output),
                model=args.model,
                max_retries=1,
                retry_sleep=args.retry_sleep,
                validator=validate_code_final_output,
            )

            normalized = normalize_text(stage2_text)
            final_code = extract_python_code(normalized)
            if final_code != code.strip():
                raise RuntimeError("Final code-tool output changed the Python code block.")
            expected_output_line = f"- Tool Output: {display_tool_output(executed_output)}"
            if expected_output_line not in normalized:
                raise RuntimeError("Final code-tool output changed the tool output string.")

            stats["retry_attempts_total"] += attempt - 1
            return convert_code_output_to_raw(
                output_text=normalized,
                original_record=original_record,
                executed_output=executed_output,
            )
        except Exception as exc:
            last_error = str(exc)
            if "Search or AskUser" in last_error:
                stats["validation_failures"][last_error] += 1
            elif "Python code block" in last_error or "### Final Response" in last_error or "- Tool Output:" in last_error:
                stats["validation_failures"][last_error] += 1
            elif "execute" in last_error or "timeout" in last_error or "print any output" in last_error:
                stats["execution_failures"][last_error] += 1
            else:
                stats["validation_failures"][last_error] += 1
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep)
    raise RuntimeError(last_error)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = load_source_dataset()
    task_ids = list(source_dataset.keys())

    no_tool_raw = load_json_dict(args.output_dir / "domain_math_no_tool_raw.json", args.overwrite)
    code_tool_raw = load_json_dict(args.output_dir / "domain_math_code_tool_raw.json", args.overwrite)

    no_tool_stats = {
        "failed_ids": [],
        "retry_attempts_total": 0,
        "validation_failures": Counter(),
    }
    code_tool_stats = {
        "failed_ids": [],
        "retry_attempts_total": 0,
        "validation_failures": Counter(),
        "execution_failures": Counter(),
    }

    client = build_client()

    for task_id in tqdm(task_ids, desc="Generating math single-turn pairs"):
        original_record = source_dataset[task_id]
        problem = extract_problem_text(original_record)

        if task_id not in no_tool_raw:
            try:
                no_tool_raw[task_id] = build_no_tool_record(
                    client=client,
                    problem=problem,
                    original_record=original_record,
                    args=args,
                    stats=no_tool_stats,
                )
                save_all_outputs(
                    output_dir=args.output_dir,
                    task_ids=task_ids,
                    source_dataset=source_dataset,
                    no_tool_raw=no_tool_raw,
                    code_tool_raw=code_tool_raw,
                    no_tool_stats=no_tool_stats,
                    code_tool_stats=code_tool_stats,
                    args=args,
                )
            except Exception:
                no_tool_stats["failed_ids"].append(task_id)

        if task_id not in code_tool_raw:
            try:
                code_tool_raw[task_id] = build_code_tool_record(
                    client=client,
                    problem=problem,
                    original_record=original_record,
                    args=args,
                    stats=code_tool_stats,
                )
                save_all_outputs(
                    output_dir=args.output_dir,
                    task_ids=task_ids,
                    source_dataset=source_dataset,
                    no_tool_raw=no_tool_raw,
                    code_tool_raw=code_tool_raw,
                    no_tool_stats=no_tool_stats,
                    code_tool_stats=code_tool_stats,
                    args=args,
                )
            except Exception:
                code_tool_stats["failed_ids"].append(task_id)

    save_all_outputs(
        output_dir=args.output_dir,
        task_ids=task_ids,
        source_dataset=source_dataset,
        no_tool_raw=no_tool_raw,
        code_tool_raw=code_tool_raw,
        no_tool_stats=no_tool_stats,
        code_tool_stats=code_tool_stats,
        args=args,
    )

    summary_path = args.output_dir / "domain_math_single_turn_build_summary.json"
    print(
        json.dumps(
            {
                "task_count": len(task_ids),
                "no_tool_completed": len(no_tool_raw),
                "code_tool_completed": len(code_tool_raw),
                "no_tool_failed": len(no_tool_stats["failed_ids"]),
                "code_tool_failed": len(code_tool_stats["failed_ids"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
