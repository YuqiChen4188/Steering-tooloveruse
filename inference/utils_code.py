"""Minimal Code helper for SteeringMark inference."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


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


def extract_python_code_block(text: str) -> str | None:
    opening_match = re.search(r"```python[ \t]*\n", text)
    if not opening_match:
        return None

    code_start = opening_match.end()
    closing_match = re.search(r"\n```", text[code_start:])
    if closing_match:
        code_end = code_start + closing_match.start()
        return text[code_start:code_end].strip()
    return text[code_start:].strip()


def execute_code(code_text: str, file_name: str | Path) -> str:
    code = extract_python_code_block(code_text)
    if code is None:
        return "Error: No valid Python code block found."

    code = DEFAULT_PREAMBLE + "\n" + code
    file_path = Path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_path.write_text(code, encoding="utf-8")
    except Exception as exc:
        return f"Error: Could not write to file. {exc}"

    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        output = result.stdout
    except subprocess.TimeoutExpired:
        output = "Error: Execution time exceeded the 20-second limit."
    except subprocess.CalledProcessError as exc:
        output = f"Error: {exc.stderr.strip()}"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    if len(output) > 256:
        output = output[:128] + "..." + output[-128:]
    return output
