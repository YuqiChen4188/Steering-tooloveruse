#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_ablation as ablation  # noqa: E402

MODEL_CONFIGS = {
    "Llama-3.1-8B": {
        "model_path": "/data/yuqi/Steering-tooloveruse/model_weights/Llama-3.1-8B-Instruct",
        "vector_dir": "steering_vector/Llama_3_8_vector_heading",
        "layer": "21",
    },
    "Mistral-7B": {
        "model_path": "/data/yuqi/Steering-tooloveruse/model_weights/Mistral-7B-Instruct-v0.3",
        "vector_dir": "steering_vector/Mistral_7B_vector_heading",
        "layer": "16",
    },
}
DOMAIN_ALIASES = {
    "math": "math",
    "time": "time",
    "intention": "intention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched prompt-only baseline experiments.")
    parser.add_argument("--models", nargs="+", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--domains", nargs="+", required=True, help="Domains: Math Time Intention")
    parser.add_argument(
        "--prompt-policies",
        nargs="+",
        choices=("base_tool", "conservative_tool"),
        default=["base_tool", "conservative_tool"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("base", "activation_addition", "orthogonalization"),
        default=["base", "activation_addition"],
    )
    parser.add_argument("--max-examples", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=Path("results/ablations/matched_prompt_baseline.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ablations/matched_prompt_outputs"))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prompt-method", default="llama")
    parser.add_argument("--test-start-id", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_domain(value: str) -> str:
    key = value.strip().lower()
    if key not in DOMAIN_ALIASES:
        raise ValueError(f"Unsupported domain {value!r}. Use Math, Time, or Intention.")
    return DOMAIN_ALIASES[key]


def data_path_for(domain: str) -> Path:
    return Path(f"data_inference/domain_{domain}_tool_prompt.json")


def namespace_for(args: argparse.Namespace, model: str, domain: str) -> argparse.Namespace:
    config = MODEL_CONFIGS[model]
    return argparse.Namespace(
        baseline="matched_prompt",
        model_name_or_path=config["model_path"],
        model_label=model,
        domain=domain,
        data_path=data_path_for(domain),
        markdown_vector_dir=Path(config["vector_dir"]),
        json_vector_dir=None,
        output_dir=args.output_dir,
        csv_path=args.output,
        output_path=args.output,
        layer=config["layer"],
        alpha=args.alpha,
        method=None,
        prompt_policy=None,
        prompt_method=args.prompt_method,
        max_test_num=args.max_examples,
        test_start_id=args.test_start_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
        summarize_only=args.summarize_only,
        dry_run=args.dry_run,
    )


def condition_for(method: str, prompt_policy: str) -> dict[str, str]:
    return {
        "name": f"{method}_{prompt_policy}",
        "method": method,
        "prompt_policy": prompt_policy,
        "extract_schema": "markdown",
        "eval_schema": "markdown",
        "code_heading": "Code",
    }


def run_condition(args: argparse.Namespace, run_args: argparse.Namespace, condition: dict[str, str]) -> tuple[dict[str, str], Path]:
    save_path = ablation.result_path(run_args, condition)
    command = ablation.build_matched_prompt_command(run_args, condition, save_path)
    run_note = ""
    if args.dry_run:
        print(" ".join(command))
        run_note = "dry_run"
    elif args.summarize_only:
        run_note = "summarize_only"
    elif args.overwrite or not save_path.exists():
        if args.overwrite:
            for stale_path in [save_path, save_path.with_name(save_path.stem + "_errors.json")]:
                stale_path.unlink(missing_ok=True)
        subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
        run_note = "ran"
    else:
        run_note = "existing_result"

    tool_avg_use, accuracy, malformed_rate, over_suppression_rate, num_examples, metric_notes = ablation.summarize_matched_prompt_result(
        save_path,
        run_args.domain,
        condition["eval_schema"],
    )
    row = ablation.build_matched_prompt_row(
        run_args,
        condition,
        tool_avg_use,
        accuracy,
        malformed_rate,
        over_suppression_rate,
        num_examples,
        ";".join(piece for piece in [run_note, metric_notes] if piece),
    )
    return row, save_path


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for model in args.models:
        for domain_input in args.domains:
            domain = normalize_domain(domain_input)
            run_args = namespace_for(args, model, domain)
            group_rows: list[dict[str, str]] = []
            path_by_key: dict[tuple[str, str], Path] = {}
            for method in args.methods:
                for prompt_policy in args.prompt_policies:
                    condition = condition_for(method, prompt_policy)
                    row, save_path = run_condition(args, run_args, condition)
                    group_rows.append(row)
                    path_by_key[(method, prompt_policy)] = save_path
            ablation.apply_pairwise_over_suppression_rates(run_args, group_rows, path_by_key)
            rows.extend(group_rows)

    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ablation.MATCHED_PROMPT_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
