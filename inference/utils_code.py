"""Minimal Code helper copied for Steering-tooloveruse inference."""

from __future__ import annotations

import os
import re
import subprocess
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


def execute_code(code_text: str, file_name: str | Path) -> str:
    code_match = re.search(r"```python\n(.*?)```", code_text, re.DOTALL)
    if not code_match:
        return "Error: No valid Python code block found."

    code = DEFAULT_PREAMBLE + "\n" + code_match.group(1)
    file_path = Path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_path.write_text(code, encoding="utf-8")
    except Exception as exc:
        return f"Error: Could not write to file. {exc}"

    try:
        result = subprocess.run(
            ["python", str(file_path)],
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
