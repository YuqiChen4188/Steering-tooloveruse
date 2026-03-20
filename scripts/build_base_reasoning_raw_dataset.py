#!/usr/bin/env python3
"""Rebuild no-tool raw datasets with the SMART appendix C.2 base reasoning prompt."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


RAW_DATASETS = {
    "intention": Path("/data/yuqi/Steering-tooloveruse/data_raw/domain_intention_raw.json"),
    "math": Path("/data/yuqi/Steering-tooloveruse/data_raw/domain_math_raw.json"),
    "time": Path("/data/yuqi/Steering-tooloveruse/data_raw/domain_time_raw.json"),
}

TOOL_STEP_TYPES = {"askuser", "search", "code"}

REASONING_PROMPT = """You are an advanced assistant designed to solve tasks autonomously using your knowledge and reasoning. Clearly articulate your thought process and reasoning steps before presenting the final response to ensure transparency and accuracy.
In the field '### Reasoning Steps', clearly articulate your thought process and reasoning steps towards the final answer. Then you should present a succinct and accurate final response in the field '### Final Response'."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample tool-used raw examples and rebuild them with the SMART appendix "
            "C.2 Base Model Reasoning Prompt."
        )
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name.")
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=100,
        help="Number of tool-used samples to select from each raw domain dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum OpenAI retry attempts per sample.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="Seconds to sleep between failed OpenAI attempts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, ignore any existing partial outputs and rebuild from scratch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/yuqi/Steering-tooloveruse/data_train/base_reasoning_raw"),
        help="Directory for selected raw subsets, rebuilt raw outputs, and summary metadata.",
    )
    return parser.parse_args()


def load_secret() -> dict[str, Any]:
    secret_path = Path(__file__).resolve().parents[1] / "secret.json"
    with secret_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_client() -> OpenAI:
    secret = load_secret()
    return OpenAI(api_key=secret["api_key"], base_url=secret["base_url"])


def load_raw_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_problem_text(record: dict[str, Any]) -> str:
    data = record.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("Expected raw record['data'] to be a dict.")
    for key in ("problem", "question", "input", "task"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in data.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Could not extract problem text from raw record.")


def uses_tool(record: dict[str, Any]) -> bool:
    for step in record.get("reasoning_complete", []):
        if not isinstance(step, dict):
            continue
        if (step.get("type") or "").lower() in TOOL_STEP_TYPES:
            return True
    return False


def sample_tool_used_subset(
    raw_dataset: dict[str, Any], samples_per_domain: int, seed: int
) -> dict[str, Any]:
    tool_used_ids = [task_id for task_id, record in raw_dataset.items() if uses_tool(record)]
    if len(tool_used_ids) < samples_per_domain:
        raise ValueError(
            f"Requested {samples_per_domain} tool-used samples, but only found "
            f"{len(tool_used_ids)}."
        )
    sampled_ids = random.Random(seed).sample(tool_used_ids, samples_per_domain)
    return {task_id: raw_dataset[task_id] for task_id in sampled_ids}


def build_messages(problem: str) -> list[dict[str, str]]:
    user_prompt = f"### Task\n{problem.strip()}\n### Reasoning Steps"
    return [
        {"role": "system", "content": REASONING_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_model(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str,
    max_retries: int,
    retry_sleep: float,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.000001,
                n=1,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(retry_sleep)
    raise RuntimeError("Unreachable retry loop exit.")


def normalize_response_text(text: str) -> str:
    text = text.replace("** Input **", "")
    text = text.replace("** Output **", "")
    text = text.strip()
    return text


def split_reasoning_and_final(text: str) -> tuple[str, str, bool]:
    cleaned = normalize_response_text(text)

    final_patterns = [
        r"###\s*Final Response\b",
        r"Final Response:\s*",
        r"Final Answer:\s*",
        r"The final answer is\s*:?",
        r"The answer is\s*:?",
    ]
    final_match = None
    for pattern in final_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match is not None:
            final_match = match
            break

    if final_match is None:
        return "", cleaned.strip(), False

    before = cleaned[: final_match.start()].strip()
    after = cleaned[final_match.end() :].lstrip(" \n\t:").strip()
    before = re.sub(r"^\s*###\s*Reasoning Steps\b", "", before, flags=re.IGNORECASE).strip()
    return before, after, True


def split_reasoning_steps(reasoning_text: str) -> list[str]:
    if not reasoning_text:
        return []

    text = reasoning_text.strip()
    numbered = re.split(r"\n(?=\d+\.\s+)", text)
    if len(numbered) > 1:
        return [chunk.strip() for chunk in numbered if chunk.strip()]

    paragraph_chunks = re.split(r"\n\s*\n+", text)
    return [chunk.strip() for chunk in paragraph_chunks if chunk.strip()]


def convert_response_to_raw_fields(response_text: str, original_record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    reasoning_text, final_text, used_explicit_final = split_reasoning_and_final(response_text)
    reasoning_steps = split_reasoning_steps(reasoning_text)

    steps = []
    step_id = 1
    for chunk in reasoning_steps:
        steps.append(
            {
                "step": step_id,
                "name": f"Reasoning Step {step_id}",
                "type": "normal",
                "reasoning": chunk,
            }
        )
        step_id += 1

    final_reasoning = final_text if final_text else response_text.strip()
    steps.append(
        {
            "step": step_id,
            "name": "Final Response",
            "type": "normal",
            "reasoning": final_reasoning,
        }
    )

    rebuilt = {
        "data": original_record["data"],
        "tool_involved": False,
        "reasoning_complete": steps,
    }

    if "thought" in original_record:
        rebuilt["thought"] = reasoning_text if reasoning_text else final_reasoning

    return rebuilt, used_explicit_final


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_existing_output(path: Path, overwrite: bool) -> dict[str, Any]:
    if overwrite or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_domain_outputs(
    domain: str,
    selected_subset: dict[str, Any],
    args: argparse.Namespace,
    client: OpenAI,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_path = args.output_dir / f"domain_{domain}_tool_used_selected_raw.json"
    rebuilt_path = args.output_dir / f"domain_{domain}_base_reasoning_raw.json"

    save_json(selected_path, selected_subset)
    rebuilt = load_existing_output(rebuilt_path, args.overwrite)

    domain_summary = {
        "selected_count": len(selected_subset),
        "completed_count": len(rebuilt),
        "failed_ids": [],
        "fallback_ids": [],
    }

    iterable = list(selected_subset.items())
    for task_id, original_record in tqdm(iterable, desc=f"Reanswering {domain}", leave=False):
        if task_id in rebuilt:
            continue

        problem = extract_problem_text(original_record)
        messages = build_messages(problem)

        try:
            response_text = call_model(
                client=client,
                messages=messages,
                model=args.model,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            rebuilt_record, used_explicit_final = convert_response_to_raw_fields(
                response_text=response_text,
                original_record=original_record,
            )
            rebuilt[task_id] = rebuilt_record
            if not used_explicit_final:
                domain_summary["fallback_ids"].append(task_id)
            save_json(rebuilt_path, rebuilt)
        except Exception:
            domain_summary["failed_ids"].append(task_id)

    domain_summary["completed_count"] = len(rebuilt)
    domain_summary["failed_count"] = len(domain_summary["failed_ids"])
    domain_summary["fallback_count"] = len(domain_summary["fallback_ids"])
    return rebuilt, domain_summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client()

    summary: dict[str, Any] = {
        "model": args.model,
        "samples_per_domain": args.samples_per_domain,
        "seed": args.seed,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "prompt_name": "SMART Appendix C.2 Base Model Reasoning Prompt",
        "prompt_text": REASONING_PROMPT,
        "selection_rule": "tool-used iff any reasoning_complete step has type in {'askuser','search','code'}",
        "domains": {},
    }

    for domain, path in RAW_DATASETS.items():
        raw_dataset = load_raw_dataset(path)
        selected_subset = sample_tool_used_subset(
            raw_dataset=raw_dataset,
            samples_per_domain=args.samples_per_domain,
            seed=args.seed,
        )

        rebuilt, domain_summary = build_domain_outputs(
            domain=domain,
            selected_subset=selected_subset,
            args=args,
            client=client,
        )
        domain_summary["output_count"] = len(rebuilt)
        domain_summary["selected_path"] = str(
            args.output_dir / f"domain_{domain}_tool_used_selected_raw.json"
        )
        domain_summary["rebuilt_path"] = str(
            args.output_dir / f"domain_{domain}_base_reasoning_raw.json"
        )
        summary["domains"][domain] = domain_summary

    summary["aggregate"] = {
        "selected_count": sum(info["selected_count"] for info in summary["domains"].values()),
        "completed_count": sum(info["completed_count"] for info in summary["domains"].values()),
        "failed_count": sum(info["failed_count"] for info in summary["domains"].values()),
        "fallback_count": sum(info["fallback_count"] for info in summary["domains"].values()),
    }

    summary_path = args.output_dir / "base_reasoning_raw_build_summary.json"
    save_json(summary_path, summary)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
