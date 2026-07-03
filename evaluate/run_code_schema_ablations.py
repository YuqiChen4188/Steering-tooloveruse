#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = PROJECT_ROOT / "inference" / "inference_tool_prompt_tag_suppressed_kvcache.py"
CSV_COLUMNS = [
    "model",
    "domain",
    "ablation",
    "extract_schema",
    "eval_schema",
    "code_heading",
    "method",
    "alpha",
    "layer",
    "tool_avg_use",
    "accuracy",
    "num_examples",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or summarize Code schema ablations.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--domain", default="math")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--markdown-vector-dir", type=Path, required=True)
    parser.add_argument("--json-vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results/code_schema_ablations"))
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--method", default="activation_addition")
    parser.add_argument("--prompt-method", default="llama")
    parser.add_argument("--max-test-num", type=int, default=-1)
    parser.add_argument("--test-start-id", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def slug(value: Any) -> str:
    text = str(value).replace("/", "_").replace(" ", "_").replace(".", "p")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})


def model_label(args: argparse.Namespace) -> str:
    if args.model_label:
        return args.model_label
    return Path(args.model_name_or_path).name.replace("-Instruct", "")


def build_conditions() -> list[dict[str, str]]:
    conditions: list[dict[str, str]] = []
    for extract_schema, eval_schema in [
        ("markdown", "markdown"),
        ("markdown", "json"),
        ("json", "json"),
        ("json", "markdown"),
    ]:
        conditions.append(
            {
                "ablation": "cross_format",
                "extract_schema": extract_schema,
                "eval_schema": eval_schema,
                "code_heading": "Code",
            }
        )
    for code_heading in ["Code", "Compute", "Execute", "Action_B"]:
        conditions.append(
            {
                "ablation": "heading_rename",
                "extract_schema": "markdown",
                "eval_schema": "markdown",
                "code_heading": code_heading,
            }
        )
    return conditions


def result_path(args: argparse.Namespace, condition: dict[str, str]) -> Path:
    schema_pair = condition["extract_schema"] + "_to_" + condition["eval_schema"]
    heading_part = "heading_" + condition["code_heading"]
    name = "_".join(
        [
            slug(model_label(args)),
            slug(args.domain),
            condition["ablation"],
            schema_pair,
            heading_part,
            f"layer_{slug(args.layer)}",
            f"alpha_{slug(args.alpha)}",
        ]
    )
    return args.output_dir / f"{name}.json"

def vector_dir_for(args: argparse.Namespace, extract_schema: str) -> Path:
    return args.json_vector_dir if extract_schema == "json" else args.markdown_vector_dir


def build_command(args: argparse.Namespace, condition: dict[str, str], save_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--model_name_or_path",
        args.model_name_or_path,
        "--data_path",
        str(args.data_path),
        "--steering_vector_dir",
        str(vector_dir_for(args, condition["extract_schema"])),
        "--steering_payload_name",
        "step_mark_code.pt",
        "--save_path",
        str(save_path),
        "--domain",
        args.domain,
        "--steering_layers",
        str(args.layer),
        "--suppress_scale",
        str(args.alpha),
        "--method",
        args.prompt_method,
        "--extract-schema",
        condition["extract_schema"],
        "--eval-schema",
        condition["eval_schema"],
        "--schema",
        condition["eval_schema"],
        "--code-heading",
        condition["code_heading"],
        "--ablation",
        condition["ablation"],
        "--max_test_num",
        str(args.max_test_num),
        "--test_start_id",
        str(args.test_start_id),
        "--device",
        args.device,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--max_steps",
        str(args.max_steps),
        "--quiet",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def count_code_tools(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        for step in record.get("predict", []):
            if step.get("type") == "tool" and step.get("tool_name") == "Code":
                total += 1
    return total


def summarize_result(path: Path) -> tuple[str, str, int, str]:
    records = load_json_list(path)
    judge_path = path.with_name(path.stem + "_judge.json")
    judged = load_json_list(judge_path)
    metric_records = judged or records
    num_examples = len(metric_records)
    if num_examples:
        tool_avg_use = count_code_tools(metric_records) / num_examples
        tool_avg_text = f"{tool_avg_use:.6f}"
    else:
        tool_avg_text = ""
    notes: list[str] = []
    accuracy_text = ""
    if judged:
        correct = sum(1 for record in judged if record.get("judge") == "correct")
        total = sum(1 for record in judged if record.get("judge") in {"correct", "wrong"})
        accuracy_text = f"{(correct / total):.6f}" if total else ""
    else:
        notes.append("missing_judge")
    if not records:
        notes.append("missing_result")
    return tool_avg_text, accuracy_text, num_examples, ";".join(notes)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in build_conditions():
        save_path = result_path(args, condition)
        command = build_command(args, condition, save_path)
        run_note = ""
        if args.dry_run:
            print(" ".join(command))
            run_note = "dry_run"
        elif not args.summarize_only and (args.overwrite or not save_path.exists()):
            subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
            run_note = "ran"
        elif args.summarize_only:
            run_note = "summarize_only"
        else:
            run_note = "existing_result"

        tool_avg_use, accuracy, num_examples, metric_notes = summarize_result(save_path)
        notes = ";".join(piece for piece in [run_note, metric_notes] if piece)
        rows.append(
            {
                "model": model_label(args),
                "domain": args.domain,
                "ablation": condition["ablation"],
                "extract_schema": condition["extract_schema"],
                "eval_schema": condition["eval_schema"],
                "code_heading": condition["code_heading"],
                "method": args.method,
                "alpha": str(args.alpha),
                "layer": str(args.layer),
                "tool_avg_use": tool_avg_use,
                "accuracy": accuracy,
                "num_examples": str(num_examples),
                "notes": notes,
            }
        )

    with args.csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
