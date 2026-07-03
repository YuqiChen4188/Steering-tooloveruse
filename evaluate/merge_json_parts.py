#!/usr/bin/env python3
"""Merge sharded JSON result files like `*_part1.json`, `*_part2.json` into one file."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


PART_PATTERN = re.compile(r"part(\d+)")


def extract_part_index(path: str) -> int:
    matches = PART_PATTERN.findall(Path(path).stem)
    if not matches:
        raise ValueError(f"Could not find part index in file name: {path}")
    return int(matches[-1])


def infer_output_path(pattern: str) -> Path:
    if "*" in pattern:
        output_name = pattern.replace("part*.json", "all.json")
    else:
        output_name = PART_PATTERN.sub("all", pattern)
    return Path(output_name)


def merge_payloads(payloads: list[Any]) -> Any:
    if not payloads:
        raise ValueError("No JSON payloads were loaded.")

    first = payloads[0]
    if isinstance(first, list):
        merged: list[Any] = []
        for payload in payloads:
            if not isinstance(payload, list):
                raise TypeError("All input files must have the same top-level JSON type.")
            merged.extend(payload)
        return merged

    if isinstance(first, dict):
        merged_dict: dict[str, Any] = {}
        for payload in payloads:
            if not isinstance(payload, dict):
                raise TypeError("All input files must have the same top-level JSON type.")
            overlap = set(merged_dict) & set(payload)
            if overlap:
                overlap_preview = ", ".join(sorted(list(overlap))[:5])
                raise ValueError(f"Duplicate keys found when merging dict JSON files: {overlap_preview}")
            merged_dict.update(payload)
        return merged_dict

    raise TypeError(f"Unsupported top-level JSON type: {type(first).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge JSON part files into one file.")
    parser.add_argument(
        "pattern",
        help="Glob pattern for part files, for example: inference_results/llama8b_/domain_math_heading_layer21_scale1p0_part*.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path. Defaults to replacing `part*.json` with `all.json`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.pattern), key=extract_part_index)
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    payloads = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payloads.append(json.load(f))

    merged = merge_payloads(payloads)
    output_path = Path(args.output) if args.output else infer_output_path(args.pattern)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    merged_count = len(merged) if isinstance(merged, (list, dict)) else "unknown"
    print(f"Merged {len(paths)} files into {output_path}")
    print(f"Top-level type: {type(merged).__name__}, size: {merged_count}")


if __name__ == "__main__":
    main()
