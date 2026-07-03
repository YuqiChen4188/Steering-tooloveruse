"""Shared schema helpers for SteeringMark tool-use ablations."""

from __future__ import annotations

import json
import re
from typing import Any


SCHEMAS = ("markdown", "json")
CODE_HEADINGS = ("Code", "Compute", "Execute", "Action_B")
CANONICAL_TOOL_TAGS = {"Code", "Search", "AskUser"}
MARKDOWN_BASE_TAGS = ("Reasoning", "Search", "AskUser", "FinalResponse", "Final Response")
JSON_ACTION_TO_TAG = {
    "reasoning": "Reasoning",
    "code": "Code",
    "final": "FinalResponse",
}
ACTION_TO_TAG = {
    "reasoning": "Reasoning",
    "think": "Reasoning",
    "search": "Search",
    "code": "Code",
    "compute": "Code",
    "execute": "Code",
    "action_b": "Code",
    "askuser": "AskUser",
    "ask_user": "AskUser",
    "ask": "AskUser",
    "final": "FinalResponse",
    "finalresponse": "FinalResponse",
    "final_response": "FinalResponse",
}
TAG_TO_JSON_ACTION = {
    "Reasoning": "reasoning",
    "Code": "code",
    "FinalResponse": "final",
}
TAG_TO_ACTION = {
    "Reasoning": "reasoning",
    "Search": "search",
    "Code": "code",
    "AskUser": "askuser",
    "FinalResponse": "final",
}


def resolve_schema(schema: str | None, fallback: str = "markdown") -> str:
    value = (schema or fallback).strip().lower()
    if value not in SCHEMAS:
        raise ValueError(f"Unsupported schema {schema!r}. Allowed: {', '.join(SCHEMAS)}")
    return value


def resolve_code_heading(code_heading: str) -> str:
    if code_heading not in CODE_HEADINGS:
        raise ValueError(f"Unsupported code heading {code_heading!r}. Allowed: {', '.join(CODE_HEADINGS)}")
    return code_heading


def normalize_action_name(action: Any) -> str | None:
    if action is None:
        return None
    text = str(action).strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", text)


def action_to_tag(action: Any, strict_json: bool = False) -> str | None:
    normalized = normalize_action_name(action)
    if normalized is None:
        return None
    if strict_json:
        return JSON_ACTION_TO_TAG.get(normalized)
    return ACTION_TO_TAG.get(normalized)


def tag_to_action(tag: str, strict_json: bool = False) -> str:
    compact = tag.replace(" ", "")
    if compact == "FinalResponse":
        return "final"
    if strict_json:
        if tag not in TAG_TO_JSON_ACTION:
            raise ValueError(f"Tag {tag!r} cannot be rendered in the JSON cross-format schema.")
        return TAG_TO_JSON_ACTION[tag]
    return TAG_TO_ACTION[tag]


def normalize_markdown_tag(tag: str, code_heading: str = "Code") -> str:
    compact = tag.replace(" ", "")
    if compact == "FinalResponse":
        return "FinalResponse"
    if tag == code_heading:
        return "Code"
    return tag


def markdown_heading_pattern(code_heading: str = "Code") -> re.Pattern[str]:
    resolve_code_heading(code_heading)
    tags = list(MARKDOWN_BASE_TAGS)
    if code_heading not in tags:
        tags.insert(2, code_heading)
    if "Code" not in tags:
        tags.insert(2, "Code")
    tag_alt = "|".join(re.escape(tag) for tag in tags)
    return re.compile(
        rf"(?ms)^###\s*(?P<tag>{tag_alt})(?:[ \t]+(?P<title>[^\n]*))?[ \t]*\n"
        rf"(?P<body>.*?)(?=^###\s*(?:{tag_alt})(?:[ \t]+[^\n]*)?[ \t]*\n|\Z)"
    )


def parse_markdown_blocks(text: str, code_heading: str = "Code") -> list[dict[str, Any]]:
    pattern = markdown_heading_pattern(code_heading)
    blocks: list[dict[str, Any]] = []
    for match in pattern.finditer(text.strip()):
        raw_tag = match.group("tag")
        canonical_tag = normalize_markdown_tag(raw_tag, code_heading=code_heading)
        blocks.append(
            {
                "tag": canonical_tag,
                "raw_tag": raw_tag,
                "title": (match.group("title") or "").strip(),
                "body": match.group("body").rstrip(),
                "block_text": match.group(0).rstrip(),
            }
        )
    return blocks


def render_markdown_step(step: dict[str, Any], code_heading: str = "Code") -> str:
    resolve_code_heading(code_heading)
    tag = step.get("tag")
    if tag is None:
        if step.get("type") == "tool":
            tag = step.get("tool_name")
        elif step.get("name") == "Final Response":
            tag = "FinalResponse"
        else:
            tag = "Reasoning"
    if tag == "FinalResponse":
        heading = "Final Response"
    elif tag == "Code":
        heading = code_heading
    else:
        heading = str(tag)

    body = str(step.get("reasoning", step.get("body", ""))).strip()
    lines = [f"### {heading}"]
    if body:
        lines.append(body)
    if "output" in step:
        lines.append(f"- Output: {step['output']}")
    return "\n".join(lines).rstrip()


def render_json_action_step(step: dict[str, Any]) -> str:
    tag = step.get("tag")
    if tag is None:
        if step.get("type") == "tool":
            tag = step.get("tool_name")
        elif step.get("name") == "Final Response":
            tag = "FinalResponse"
        else:
            tag = "Reasoning"
    obj: dict[str, Any] = {
        "action": tag_to_action(str(tag), strict_json=True),
        "content": str(step.get("reasoning", step.get("body", ""))).strip(),
    }
    if "output" in step:
        obj["output"] = step["output"]
    return json.dumps(obj, ensure_ascii=False)


def render_step_for_schema(step: dict[str, Any], schema: str, code_heading: str = "Code") -> str:
    schema = resolve_schema(schema)
    if schema == "json":
        return render_json_action_step(step)
    return render_markdown_step(step, code_heading=code_heading)


def replace_markdown_code_heading(text: str, code_heading: str = "Code") -> str:
    resolve_code_heading(code_heading)
    if code_heading == "Code":
        return text
    return re.sub(r"(?m)^###\s*Code(?=[ \t]*\n)", f"### {code_heading}", text)


def rewrite_instruction_for_schema(instruction: str, schema: str, code_heading: str = "Code") -> str:
    schema = resolve_schema(schema)
    resolve_code_heading(code_heading)
    if schema == "json":
        return """### Task
You are a highly capable assistant designed to solve tasks effectively using your knowledge and available tools.

### Principles
1. Reason Independently:
- Leverage your own knowledge to analyze and solve reasoning steps whenever possible. Use external tools only when necessary.
2. Tool Usage:
- Use action `code` when executable Python is needed. Put the executable snippet in the JSON `content` string.
3. Step-by-Step Approach:
- Work through reasoning systematically. Rely on your knowledge until a gap requires tool support.
- After a tool call, continue reasoning using the tool output when available.
4. Goal-Oriented Resolution:
- End with action `final` that directly answers the task.

### Output Guidelines
- Output one JSON object per line.
- Each object must contain exactly these core fields: `action` and `content`.
- Allowed action values are exactly `reasoning`, `code`, and `final`.
- Do not use Markdown section headings such as `### Reasoning` or `### Code`."""

    rewritten = replace_markdown_code_heading(instruction, code_heading=code_heading)
    if code_heading != "Code":
        rewritten = rewritten.replace("`### Code`", f"`### {code_heading}`")
        rewritten = rewritten.replace("### Code", f"### {code_heading}")
        rewritten = rewritten.replace("`Code`", f"`{code_heading}`")
        rewritten = rewritten.replace(" Code step", f" {code_heading} step")
    return rewritten


def markdown_output_to_json_actions(text: str, code_heading: str = "Code") -> str:
    blocks = parse_markdown_blocks(text, code_heading=code_heading)
    if not blocks:
        return text
    lines = []
    for block in blocks:
        lines.append(
            render_json_action_step(
                {
                    "tag": block["tag"],
                    "reasoning": block["body"],
                }
            )
        )
    return "\n".join(lines)


def convert_output_for_schema(text: str, schema: str, code_heading: str = "Code") -> str:
    schema = resolve_schema(schema)
    if schema == "json":
        return markdown_output_to_json_actions(text, code_heading=code_heading)
    return replace_markdown_code_heading(text, code_heading=code_heading)


def convert_record_for_schema(record: dict[str, str], schema: str, code_heading: str = "Code") -> dict[str, str]:
    return {
        **record,
        "instruction": rewrite_instruction_for_schema(record["instruction"], schema=schema, code_heading=code_heading),
        "output": convert_output_for_schema(record["output"], schema=schema, code_heading=code_heading),
    }


def _scan_json_action_objects(text: str, strict_json: bool = True) -> list[tuple[dict[str, Any], int, int]]:
    decoder = json.JSONDecoder()
    objects: list[tuple[dict[str, Any], int, int]] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        absolute_end = start + max(end, 1)
        if isinstance(obj, dict) and action_to_tag(obj.get("action"), strict_json=strict_json) is not None:
            objects.append((obj, start, absolute_end))
        idx = absolute_end
    return objects


def parse_json_action_objects(text: str, strict_json: bool = True) -> list[dict[str, Any]]:
    return [obj for obj, _start, _end in _scan_json_action_objects(text, strict_json=strict_json)]


def json_action_parse_diagnostics(text: str, strict_json: bool = True) -> dict[str, int]:
    objects = _scan_json_action_objects(text, strict_json=strict_json)
    parsed_spans = [(start, end) for _obj, start, end in objects]
    malformed_count = 0
    candidate_count = len(objects)
    line_start = 0
    for raw_line in text.splitlines(keepends=True):
        line_end = line_start + len(raw_line)
        stripped = raw_line.strip()
        looks_like_action = bool(stripped) and (
            "{" in stripped
            or "}" in stripped
            or '"action"' in stripped
            or "'action'" in stripped
            or stripped.lower().startswith(("action:", "action ="))
        )
        if looks_like_action:
            has_parsed = any(start >= line_start and end <= line_end for start, end in parsed_spans)
            if not has_parsed:
                malformed_count += 1
                candidate_count += 1
        line_start = line_end
    return {
        "json_action_objects": len(objects),
        "malformed_json_action_lines": malformed_count,
        "json_action_candidates": candidate_count,
    }


def parse_json_action_steps(text: str, strict_json: bool = True) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for obj in parse_json_action_objects(text, strict_json=strict_json):
        tag = action_to_tag(obj.get("action"), strict_json=strict_json)
        if tag is None:
            continue
        step = {
            "tag": tag,
            "body": str(obj.get("content", "")).rstrip(),
            "block_text": json.dumps(obj, ensure_ascii=False),
        }
        if "output" in obj:
            step["output"] = obj["output"]
        steps.append(step)
    return steps


def find_json_action_value_spans(text: str, action_name: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r'"action"\s*:\s*"(?P<action>[^"]+)"', text):
        tag = action_to_tag(match.group("action"), strict_json=True)
        target_tag = action_to_tag(action_name, strict_json=True)
        if tag != target_tag:
            continue
        value_start = match.start("action")
        value_end = match.end("action")
        spans.append((value_start, value_end))
    return spans


def has_open_json_action_prefix(text: str) -> bool:
    tail_start = max(text.rfind("\n"), text.rfind("{"))
    tail = text[tail_start + 1 :] if tail_start >= 0 else text
    if '"content"' in tail or "}" in tail:
        return False
    return re.search(r'"action"\s*:\s*"?[A-Za-z_]*$', tail.rstrip()) is not None


def has_schema_trigger_prefix(text: str, schema: str) -> bool:
    schema = resolve_schema(schema)
    if schema == "json":
        return has_open_json_action_prefix(text)
    return text.rstrip().endswith("###")
