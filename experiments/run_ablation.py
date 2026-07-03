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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_schema_utils import json_action_parse_diagnostics, parse_markdown_blocks  # noqa: E402

INFERENCE_SCRIPT = PROJECT_ROOT / "inference" / "inference_tool_prompt_tag_suppressed_kvcache.py"
ORTHOGONALIZATION_SCRIPT = PROJECT_ROOT / "inference" / "inference_tool_prompt_tag_orthogonalized_kvcache.py"
MATCHED_PROMPT_METHODS = {"base", "activation_addition", "orthogonalization"}
TARGET_TOOL_BY_DOMAIN = {
    "math": "Code",
    "time": "Search",
    "intention": "AskUser",
}
CROSS_FORMAT_CSV_COLUMNS = [
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
    "malformed_output_rate",
    "num_examples",
    "notes",
]
HEADING_RENAME_CSV_COLUMNS = [
    "model",
    "domain",
    "ablation",
    "extraction_code_heading",
    "eval_code_heading",
    "method",
    "alpha",
    "layer",
    "tool_avg_use",
    "accuracy",
    "num_examples",
    "malformed_output_rate",
    "notes",
]
MATCHED_PROMPT_CSV_COLUMNS = [
    "model",
    "domain",
    "method",
    "prompt_policy",
    "alpha",
    "layer",
    "tool_avg_use",
    "accuracy",
    "num_examples",
    "malformed_output_rate",
    "over_suppression_rate",
    "notes",
]


CROSS_FORMAT_CONDITIONS = [
    {"name": "markdown_to_markdown", "extract_schema": "markdown", "eval_schema": "markdown", "code_heading": "Code"},
    {"name": "markdown_to_json", "extract_schema": "markdown", "eval_schema": "json", "code_heading": "Code"},
    {"name": "json_to_json", "extract_schema": "json", "eval_schema": "json", "code_heading": "Code"},
    {"name": "json_to_markdown", "extract_schema": "json", "eval_schema": "markdown", "code_heading": "Code"},
]
HEADING_RENAME_CONDITIONS = [
    {
        "name": "heading_code",
        "extract_schema": "markdown",
        "eval_schema": "markdown",
        "extraction_code_heading": "Code",
        "eval_code_heading": "Code",
    },
    {
        "name": "heading_compute",
        "extract_schema": "markdown",
        "eval_schema": "markdown",
        "extraction_code_heading": "Code",
        "eval_code_heading": "Compute",
    },
    {
        "name": "heading_execute",
        "extract_schema": "markdown",
        "eval_schema": "markdown",
        "extraction_code_heading": "Code",
        "eval_code_heading": "Execute",
    },
    {
        "name": "heading_action_b",
        "extract_schema": "markdown",
        "eval_schema": "markdown",
        "extraction_code_heading": "Code",
        "eval_code_heading": "Action_B",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SteeringMark ablations.")
    parser.add_argument("--ablation", choices=("cross_format", "heading_rename"), default="cross_format")
    parser.add_argument("--baseline", choices=("matched_prompt",), default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--domain", choices=("math", "time", "intention"), default="math")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--markdown-vector-dir", type=Path, required=True)
    parser.add_argument("--json-vector-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--output", dest="output_path", type=Path, default=None)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--method", choices=("base", "activation_addition", "orthogonalization"), default=None)
    parser.add_argument("--prompt-policy", choices=("base_tool", "conservative_tool"), default=None)
    parser.add_argument("--prompt-method", default="llama")
    parser.add_argument("--max-test-num", type=int, default=-1)
    parser.add_argument("--test-start-id", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output_path is not None:
        args.csv_path = args.output_path

    if args.baseline == "matched_prompt":
        if args.output_dir is None:
            args.output_dir = Path("results/ablations/matched_prompt_outputs")
        if args.csv_path is None:
            args.csv_path = Path("results/ablations/matched_prompt_baseline.csv")
        return args

    if args.method is None:
        args.method = "activation_addition"
    if args.prompt_policy is None:
        args.prompt_policy = "base_tool"
    if args.ablation == "cross_format":
        if args.json_vector_dir is None:
            parser.error("--json-vector-dir is required for --ablation cross_format")
        if args.output_dir is None:
            args.output_dir = Path("results/ablations/cross_format_outputs")
        if args.csv_path is None:
            args.csv_path = Path("results/ablations/cross_format_results.csv")
    else:
        if args.output_dir is None:
            args.output_dir = Path("results/ablations/heading_rename_outputs")
        if args.csv_path is None:
            args.csv_path = Path("results/ablations/heading_rename_results.csv")
    return args


def slug(value: Any) -> str:
    text = str(value).replace("/", "_").replace(" ", "_").replace(".", "p")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})


def model_label(args: argparse.Namespace) -> str:
    if args.model_label:
        return args.model_label
    return Path(args.model_name_or_path).name.replace("-Instruct", "")


def vector_dir_for(args: argparse.Namespace, extract_schema: str) -> Path:
    if extract_schema == "json" and args.json_vector_dir is None:
        raise ValueError("JSON extraction requires --json-vector-dir")
    return args.json_vector_dir if extract_schema == "json" else args.markdown_vector_dir


def condition_eval_code_heading(condition: dict[str, str]) -> str:
    if "eval_code_heading" in condition:
        return condition["eval_code_heading"]
    return condition["code_heading"]


def build_matched_prompt_conditions(args: argparse.Namespace) -> list[dict[str, str]]:
    methods = [args.method] if args.method is not None else ["base", "activation_addition"]
    prompt_policies = [args.prompt_policy] if args.prompt_policy is not None else ["base_tool", "conservative_tool"]
    conditions = []
    for method in methods:
        if method not in MATCHED_PROMPT_METHODS:
            raise ValueError(f"Unsupported matched prompt method: {method}")
        for prompt_policy in prompt_policies:
            conditions.append(
                {
                    "name": f"{method}_{prompt_policy}",
                    "method": method,
                    "prompt_policy": prompt_policy,
                    "extract_schema": "markdown",
                    "eval_schema": "markdown",
                    "code_heading": "Code",
                }
            )
    return conditions


def conditions_for(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.baseline == "matched_prompt":
        return build_matched_prompt_conditions(args)
    if args.ablation == "heading_rename":
        return HEADING_RENAME_CONDITIONS
    return CROSS_FORMAT_CONDITIONS


def csv_columns_for(args: argparse.Namespace) -> list[str]:
    if args.baseline == "matched_prompt":
        return MATCHED_PROMPT_CSV_COLUMNS
    if args.ablation == "heading_rename":
        return HEADING_RENAME_CSV_COLUMNS
    return CROSS_FORMAT_CSV_COLUMNS


def effective_method(args: argparse.Namespace, condition: dict[str, str]) -> str:
    return condition.get("method") or args.method or "activation_addition"


def effective_prompt_policy(args: argparse.Namespace, condition: dict[str, str]) -> str:
    return condition.get("prompt_policy") or args.prompt_policy or "base_tool"


def effective_alpha(args: argparse.Namespace, condition: dict[str, str]) -> str:
    return "0" if effective_method(args, condition) == "base" else str(args.alpha)


def effective_layer(args: argparse.Namespace, condition: dict[str, str]) -> str:
    return "" if effective_method(args, condition) == "base" else str(args.layer)


def result_path(args: argparse.Namespace, condition: dict[str, str]) -> Path:
    if args.baseline == "matched_prompt":
        name = "_".join(
            [
                slug(model_label(args)),
                slug(args.domain),
                "matched_prompt",
                slug(effective_method(args, condition)),
                slug(effective_prompt_policy(args, condition)),
                f"layer_{slug(effective_layer(args, condition) or 'none')}",
                f"alpha_{slug(effective_alpha(args, condition))}",
                f"n_{slug(args.max_test_num)}",
            ]
        )
        return args.output_dir / f"{name}.json"

    name = "_".join(
        [
            slug(model_label(args)),
            slug(args.domain),
            args.ablation,
            condition["name"],
            f"layer_{slug(args.layer)}",
            f"alpha_{slug(args.alpha)}",
            f"n_{slug(args.max_test_num)}",
        ]
    )
    return args.output_dir / f"{name}.json"


def build_matched_prompt_command(args: argparse.Namespace, condition: dict[str, str], save_path: Path) -> list[str]:
    method = effective_method(args, condition)
    prompt_policy = effective_prompt_policy(args, condition)
    script = ORTHOGONALIZATION_SCRIPT if method == "orthogonalization" else INFERENCE_SCRIPT
    command = [
        sys.executable,
        str(script),
        "--model_name_or_path",
        args.model_name_or_path,
        "--data_path",
        str(args.data_path),
        "--save_path",
        str(save_path),
        "--domain",
        args.domain,
        "--method",
        args.prompt_method,
        "--prompt-policy",
        prompt_policy,
        "--heading-interface",
        "matched_prompt",
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
    if method == "base":
        command.extend(["--disable-steering", "--suppress_scale", "0"])
    else:
        command.extend(
            [
                "--steering_vector_dir",
                str(args.markdown_vector_dir),
                "--steering_layers",
                str(args.layer),
                "--suppress_scale",
                str(args.alpha),
            ]
        )
    if method != "orthogonalization":
        command.extend(["--ablation", "matched_prompt"])
    if args.overwrite:
        command.append("--overwrite")
    return command


def build_command(args: argparse.Namespace, condition: dict[str, str], save_path: Path) -> list[str]:
    if args.baseline == "matched_prompt":
        return build_matched_prompt_command(args, condition, save_path)

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
        "--prompt-policy",
        args.prompt_policy,
        "--extract-schema",
        condition["extract_schema"],
        "--eval-schema",
        condition["eval_schema"],
        "--schema",
        condition["eval_schema"],
        "--code-heading",
        condition_eval_code_heading(condition),
        "--ablation",
        args.ablation,
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


def target_tool_for_domain(domain: str) -> str:
    if domain not in TARGET_TOOL_BY_DOMAIN:
        raise ValueError(f"Unsupported domain for target tool metrics: {domain}")
    return TARGET_TOOL_BY_DOMAIN[domain]


def count_target_tools(records: list[dict[str, Any]], target_tool: str) -> int:
    total = 0
    for record in records:
        for step in record.get("predict", []):
            if step.get("type") == "tool" and step.get("tool_name") == target_tool:
                total += 1
    return total


def record_target_tool_count(record: dict[str, Any], target_tool: str) -> int:
    return sum(
        1
        for step in record.get("predict", [])
        if step.get("type") == "tool" and step.get("tool_name") == target_tool
    )


def count_code_tools(records: list[dict[str, Any]]) -> int:
    return count_target_tools(records, "Code")


def malformed_output_rate(records: list[dict[str, Any]], eval_schema: str) -> str:
    if eval_schema != "json":
        return ""
    malformed = 0
    candidates = 0
    for record in records:
        for raw_output in record.get("raw", []):
            diagnostics = json_action_parse_diagnostics(str(raw_output))
            malformed += diagnostics["malformed_json_action_lines"]
            candidates += diagnostics["json_action_candidates"]
    if candidates == 0:
        return "0.000000"
    return f"{(malformed / candidates):.6f}"


def has_parseable_final_response(record: dict[str, Any]) -> bool:
    for step in record.get("predict", []):
        if step.get("name") == "Final Response":
            return True
    for raw_output in record.get("raw", []):
        blocks = parse_markdown_blocks(str(raw_output), code_heading="Code")
        if any(block.get("tag") == "FinalResponse" for block in blocks):
            return True
    return False


def matched_markdown_malformed_output_rate(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    malformed = sum(1 for record in records if not has_parseable_final_response(record))
    return f"{(malformed / len(records)):.6f}"


def matched_malformed_output_rate(records: list[dict[str, Any]], eval_schema: str) -> str:
    if eval_schema == "json":
        return malformed_output_rate(records, eval_schema)
    return matched_markdown_malformed_output_rate(records)


def intention_summary_score(record: dict[str, Any]) -> float | None:
    summary_results = record.get("summary_results")
    if not isinstance(summary_results, list) or not summary_results:
        return None
    values = [1.0 if str(item.get("judgment", "")).strip().lower() == "yes" else 0.0 for item in summary_results]
    return sum(values) / len(values) if values else None


def summarize_matched_prompt_result(path: Path, domain: str, eval_schema: str) -> tuple[str, str, str, str, int, str]:
    records = load_json_list(path)
    judge_path = path.with_name(path.stem + "_judge.json")
    judged = load_json_list(judge_path)
    metric_records = judged or records
    num_examples = len(metric_records)
    target_tool = target_tool_for_domain(domain)
    tool_avg_text = f"{(count_target_tools(metric_records, target_tool) / num_examples):.6f}" if num_examples else ""

    notes: list[str] = []
    accuracy_text = ""
    over_suppression_text = ""
    if judged:
        if domain == "intention":
            scored = [(record, intention_summary_score(record)) for record in judged]
            scored = [(record, score) for record, score in scored if score is not None]
            if scored:
                accuracy_text = f"{(sum(score for _record, score in scored) / len(scored)):.6f}"
            else:
                notes.append("missing_intention_summary_judge")
        else:
            judged_records = [record for record in judged if record.get("judge") in {"correct", "wrong"}]
            if judged_records:
                correct = sum(1 for record in judged_records if record.get("judge") == "correct")
                accuracy_text = f"{(correct / len(judged_records)):.6f}"
    else:
        notes.append("missing_judge")
    if not records:
        notes.append("missing_result")

    return (
        tool_avg_text,
        accuracy_text,
        matched_malformed_output_rate(records, eval_schema),
        over_suppression_text,
        num_examples,
        ";".join(notes),
    )


def summarize_result(path: Path, eval_schema: str) -> tuple[str, str, str, int, str]:
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

    return tool_avg_text, accuracy_text, malformed_output_rate(records, eval_schema), num_examples, ";".join(notes)


def build_matched_prompt_row(
    args: argparse.Namespace,
    condition: dict[str, str],
    tool_avg_use: str,
    accuracy: str,
    malformed_rate: str,
    over_suppression_rate: str,
    num_examples: int,
    notes: str,
) -> dict[str, str]:
    return {
        "model": model_label(args),
        "domain": args.domain,
        "method": effective_method(args, condition),
        "prompt_policy": effective_prompt_policy(args, condition),
        "alpha": effective_alpha(args, condition),
        "layer": effective_layer(args, condition),
        "tool_avg_use": tool_avg_use,
        "accuracy": accuracy,
        "num_examples": str(num_examples),
        "malformed_output_rate": malformed_rate,
        "over_suppression_rate": over_suppression_rate,
        "notes": notes,
    }


def build_row(
    args: argparse.Namespace,
    condition: dict[str, str],
    tool_avg_use: str,
    accuracy: str,
    malformed_rate: str,
    num_examples: int,
    notes: str,
) -> dict[str, str]:
    base = {
        "model": model_label(args),
        "domain": args.domain,
        "ablation": args.ablation,
        "method": args.method or "activation_addition",
        "alpha": str(args.alpha),
        "layer": str(args.layer),
        "tool_avg_use": tool_avg_use,
        "accuracy": accuracy,
        "malformed_output_rate": malformed_rate,
        "num_examples": str(num_examples),
        "notes": notes,
    }
    if args.ablation == "heading_rename":
        return {
            **base,
            "extraction_code_heading": condition["extraction_code_heading"],
            "eval_code_heading": condition["eval_code_heading"],
        }
    return {
        **base,
        "extract_schema": condition["extract_schema"],
        "eval_schema": condition["eval_schema"],
        "code_heading": condition["code_heading"],
    }


def record_task_score(record: dict[str, Any], domain: str) -> float | None:
    if domain == "intention":
        return intention_summary_score(record)
    judge = record.get("judge")
    if judge == "correct":
        return 1.0
    if judge == "wrong":
        return 0.0
    return None


def judged_records_by_task(path: Path, domain: str) -> dict[str, dict[str, Any]]:
    judge_path = path.with_name(path.stem + "_judge.json")
    judged = load_json_list(judge_path)
    by_task = {}
    for record in judged:
        task = record.get("task")
        if not task:
            continue
        if record_task_score(record, domain) is not None:
            by_task[str(task)] = record
    return by_task


def append_note(row: dict[str, str], note: str) -> None:
    existing = row.get("notes", "")
    pieces = [piece for piece in [existing, note] if piece]
    row["notes"] = ";".join(pieces)


def apply_pairwise_over_suppression_rates(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    path_by_key: dict[tuple[str, str], Path],
) -> None:
    if args.baseline != "matched_prompt" or args.domain not in {"time", "intention"}:
        return

    target_tool = target_tool_for_domain(args.domain)
    for row in rows:
        if row.get("prompt_policy") != "conservative_tool":
            continue
        method = row.get("method", "")
        base_path = path_by_key.get((method, "base_tool"))
        conservative_path = path_by_key.get((method, "conservative_tool"))
        if base_path is None or conservative_path is None:
            append_note(row, "over_suppression_missing_base_policy_pair")
            continue

        base_by_task = judged_records_by_task(base_path, args.domain)
        conservative_by_task = judged_records_by_task(conservative_path, args.domain)
        paired_tasks = sorted(set(base_by_task) & set(conservative_by_task))
        if not paired_tasks:
            append_note(row, "over_suppression_missing_pair_judge")
            continue

        over_suppressed = 0
        for task in paired_tasks:
            base_record = base_by_task[task]
            conservative_record = conservative_by_task[task]
            base_score = record_task_score(base_record, args.domain)
            conservative_score = record_task_score(conservative_record, args.domain)
            if base_score is None or conservative_score is None:
                continue
            if (
                record_target_tool_count(base_record, target_tool) > 0
                and record_target_tool_count(conservative_record, target_tool) == 0
                and conservative_score < base_score
            ):
                over_suppressed += 1
        row["over_suppression_rate"] = f"{(over_suppressed / len(paired_tasks)):.6f}"


def summarize_condition(args: argparse.Namespace, condition: dict[str, str], save_path: Path) -> dict[str, str]:
    if args.baseline == "matched_prompt":
        tool_avg_use, accuracy, malformed_rate, over_suppression_rate, num_examples, metric_notes = summarize_matched_prompt_result(
            save_path,
            args.domain,
            condition["eval_schema"],
        )
        return build_matched_prompt_row(
            args,
            condition,
            tool_avg_use,
            accuracy,
            malformed_rate,
            over_suppression_rate,
            num_examples,
            metric_notes,
        )

    tool_avg_use, accuracy, malformed_rate, num_examples, metric_notes = summarize_result(
        save_path,
        condition["eval_schema"],
    )
    return build_row(args, condition, tool_avg_use, accuracy, malformed_rate, num_examples, metric_notes)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    path_by_key: dict[tuple[str, str], Path] = {}
    for condition in conditions_for(args):
        save_path = result_path(args, condition)
        command = build_command(args, condition, save_path)
        run_note = ""
        if args.dry_run:
            print(" ".join(command))
            run_note = "dry_run"
        elif not args.summarize_only and (args.overwrite or not save_path.exists()):
            if args.overwrite:
                for stale_path in [save_path, save_path.with_name(save_path.stem + "_errors.json")]:
                    stale_path.unlink(missing_ok=True)
            subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
            run_note = "ran"
        elif args.summarize_only:
            run_note = "summarize_only"
        else:
            run_note = "existing_result"

        row = summarize_condition(args, condition, save_path)
        row["notes"] = ";".join(piece for piece in [run_note, row.get("notes", "")] if piece)
        rows.append(row)
        if args.baseline == "matched_prompt":
            path_by_key[(row["method"], row["prompt_policy"])] = save_path

    apply_pairwise_over_suppression_rates(args, rows, path_by_key)

    with args.csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns_for(args))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
