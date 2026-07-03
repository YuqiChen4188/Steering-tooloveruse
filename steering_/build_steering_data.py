#!/usr/bin/env python3
"""Build full labeled tool trajectories for SteeringMark steering experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TOOL_INSTRUCTION = """### Task
You are a highly capable assistant designed to solve tasks effectively using your knowledge and available tools.

### Principles
1. Reason Independently:
- Leverage your own knowledge to analyze and solve reasoning steps whenever possible. Use external tools only when necessary.
2. Tool Usage:
- Use code snippet ```python ... ``` to write executable Python when computation is needed.
- Use Search by writing a `### Search` step whose content is a concise web search query.
- Use AskUser by writing a `### AskUser` step whose content is the exact question that should be asked to the user.
3. Step-by-Step Approach:
- Work through reasoning systematically, breaking down the task into manageable steps. Rely on your knowledge until a gap is identified that requires tool support.
- Employ tools to address gaps and integrate the findings into your solution.
4. Goal-Oriented Resolution:
- Conclude your reasoning process by achieving a clear, accurate, and succinct solution based on your independent analysis and insights gained from tools.

### Output Guidelines
- Your answer must begin with `### Reasoning`.
- After that, every step must begin with one of these section headers on its own line: `### Reasoning`, `### Search`, `### Code`, `### AskUser`, `### Final Response`.
- If you need to use the code tool, place the executable snippet inside a `### Code` step using ```python ... ```.
- If you need to use search, place only the search query inside a `### Search` step.
- If you need to ask the user for information, place only the user-facing question inside a `### AskUser` step.
- After a tool call, continue reasoning using the tool output when available.
- End with a `### Final Response` step that directly answers the task."""


RAW_FILES = (
    ("math", "domain_math_raw.json"),
    ("time", "domain_time_raw.json"),
    ("intention", "domain_intention_raw.json"),
)

TOOL_TYPES = ("search", "code", "askuser")
TOOL_TYPE_TO_TAG = {
    "normal": "Reasoning",
    "search": "Search",
    "code": "Code",
    "askuser": "AskUser",
}
OUTPUT_FILE_NAMES = {
    "search": "steering_data_search_20.json",
    "code": "steering_data_code_20.json",
    "askuser": "steering_data_askuser_20.json",
}
SUMMARY_FILE_NAME = "steering_data_summary.json"
MERGED_FILE_NAME = "steering_data_tool.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def build_task_text(sample: dict[str, Any]) -> str:
    problem = sample["data"]["problem"].strip()
    return f"### Task\n{problem}\n"


def is_final_step(step_idx: int, total_steps: int) -> bool:
    return step_idx == total_steps - 1


def extract_tool_query(text: str, tool_name: str) -> str:
    pattern = rf"^\s*{re.escape(tool_name)}\((.*)\)\s*$"
    match = re.match(pattern, text.strip(), flags=re.DOTALL)
    if not match:
        return text.strip()
    return match.group(1).strip()


def normalize_code_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        return stripped
    return f"```python\n{stripped}\n```"


def format_step_block(step: dict[str, Any], final_step: bool) -> str:
    if final_step:
        answer = (step.get("reasoning") or "").strip()
        return f"### Final Response\n{answer}"

    step_type = str(step.get("type", "normal")).strip().lower()
    tag = TOOL_TYPE_TO_TAG.get(step_type, "Reasoning")
    content = (step.get("reasoning") or "").strip()

    if step_type == "search":
        content = extract_tool_query(content, "Search")
    elif step_type == "askuser":
        content = extract_tool_query(content, "AskUser")
    elif step_type == "code":
        content = normalize_code_block(content)

    block = f"### {tag}\n{content}"
    if step_type in TOOL_TYPES:
        tool_output = step.get("simulate_response", "")
        block = f"{block}\n- Output: {tool_output}"
    return block


def format_full_trajectory(sample: dict[str, Any]) -> str:
    steps = sample.get("reasoning_complete", [])
    if not steps:
        raise ValueError("Sample is missing reasoning_complete.")
    blocks = [
        format_step_block(step, is_final_step(idx, len(steps)))
        for idx, step in enumerate(steps)
    ]
    return "\n\n".join(blocks)


def contains_tool_type(sample: dict[str, Any], tool_type: str) -> bool:
    steps = sample.get("reasoning_complete", [])
    return any(str(step.get("type", "")).strip().lower() == tool_type for step in steps)


def build_example(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "instruction": TOOL_INSTRUCTION,
        "input": build_task_text(sample),
        "output": format_full_trajectory(sample),
    }


def collect_tool_datasets(raw_dir: Path, limit_per_tool: int) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    datasets = {tool_type: [] for tool_type in TOOL_TYPES}
    selection_meta = {tool_type: [] for tool_type in TOOL_TYPES}

    for domain, file_name in RAW_FILES:
        file_path = raw_dir / file_name
        data = load_json(file_path)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object at {file_path}")

        for sample_id in sorted(data):
            sample = data[sample_id]
            for tool_type in TOOL_TYPES:
                if len(datasets[tool_type]) >= limit_per_tool:
                    continue
                if not contains_tool_type(sample, tool_type):
                    continue
                datasets[tool_type].append(build_example(sample))
                selection_meta[tool_type].append(
                    {
                        "domain": domain,
                        "source_file": file_name,
                        "sample_id": sample_id,
                        "step_types": [
                            str(step.get("type", "")).strip().lower()
                            for step in sample.get("reasoning_complete", [])
                        ],
                    }
                )

    for tool_type, examples in datasets.items():
        if len(examples) < limit_per_tool:
            raise ValueError(
                f"Only collected {len(examples)} trajectories for {tool_type}, "
                f"but {limit_per_tool} were requested."
            )

    summary = {
        "limit_per_tool": limit_per_tool,
        "counts": {tool_type: len(examples) for tool_type, examples in datasets.items()},
        "selections": selection_meta,
    }
    return datasets, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build full labeled steering trajectories from SteeringMark/data_raw, "
            "with separate 20-trajectory subsets for Search, Code, and AskUser."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root directory of the SteeringMark project.",
    )
    parser.add_argument(
        "--limit-per-tool",
        type=int,
        default=20,
        help="Number of complete trajectories to save for each tool type.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to SteeringMark/steering_.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_dir = project_root / "data_raw"
    output_dir = args.output_dir.resolve() if args.output_dir else (project_root / "steering_")

    datasets, summary = collect_tool_datasets(raw_dir=raw_dir, limit_per_tool=args.limit_per_tool)

    for tool_type, file_name in OUTPUT_FILE_NAMES.items():
        write_json(output_dir / file_name, datasets[tool_type])

    merged = datasets["search"] + datasets["code"] + datasets["askuser"]
    write_json(output_dir / MERGED_FILE_NAME, merged)
    write_json(output_dir / SUMMARY_FILE_NAME, summary)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "search_output": str(output_dir / OUTPUT_FILE_NAMES["search"]),
                "code_output": str(output_dir / OUTPUT_FILE_NAMES["code"]),
                "askuser_output": str(output_dir / OUTPUT_FILE_NAMES["askuser"]),
                "merged_output": str(output_dir / MERGED_FILE_NAME),
                "summary_output": str(output_dir / SUMMARY_FILE_NAME),
                "counts": summary["counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
