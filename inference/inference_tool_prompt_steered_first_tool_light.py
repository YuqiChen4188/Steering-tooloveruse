#!/usr/bin/env python3
"""Run tool-prompt inference with first-tool-light / repeat-tool-heavy steering."""

from __future__ import annotations

import argparse
import json
import re
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_askuser import simulate_user_response
from utils_code import execute_code
from utils_serper import search_serper


TIMEOUT_FALLBACK = (
    "Still do not get an answer after exceeding maximum step time! "
    "Please judge the answer for this question as wrong."
)


def log_message(text: str, quiet: bool) -> None:
    """Route verbose runtime logs through tqdm so the progress bar stays usable."""
    if not quiet:
        tqdm.write(text)


def save_json(path: Path, data: Any) -> None:
    """Write JSON with UTF-8 and pretty formatting for easy inspection/debugging."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_first_parentheses_content(text: str) -> str | None:
    """Extract the first balanced `( ... )` span, used for AskUser/Search arguments."""
    start = text.find("(")
    if start == -1:
        return None
    stack: list[str] = []
    content: list[str] = []
    for char in text[start + 1 :]:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack:
                return "".join(content).strip()
            stack.pop()
        content.append(char)
    return None


def find_earliest_tool(text: str) -> tuple[int, str | None]:
    """Detect the earliest tool marker in the generated text."""
    substrings = {
        "AskUser": "AskUser(",
        "Search": "Search(",
        "Code": "```python",
    }
    occurrences = {key: text.find(value) for key, value in substrings.items()}
    valid_occurrences = {key: idx for key, idx in occurrences.items() if idx != -1}
    if not valid_occurrences:
        return -1, None

    earliest_key = min(valid_occurrences, key=valid_occurrences.get)
    return valid_occurrences[earliest_key], earliest_key


def find_earliest_final_marker(text: str) -> tuple[int, int, str | None]:
    """
    Detect final-answer markers, including light natural-language variants.

    We keep this matcher intentionally conservative: it covers the patterns we
    repeatedly observe in generation logs ("Final Answer:", "### Final Response",
    "The final answer is:", etc.) without trying to infer every possible answer-like
    sentence. This reduces false positives while fixing common non-stop failures.
    """
    patterns = (
        r"Final Answer\s*:",
        r"###\s*Final Response\b",
        r"Final Response\s*:",
        r"The final answer is\s*:?",
        r"The answer is\s*:?",
    )

    best_match = None
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        if best_match is None or match.start() < best_match.start():
            best_match = match

    if best_match is None:
        return -1, -1, None
    return best_match.start(), best_match.end(), best_match.group(0)


def parse_steps(text: str) -> list[dict[str, Any]]:
    """
    Parse one round of model output into normalized reasoning/tool/final-response steps.

    Important: to stay close to the original Open-SMARTAgent behavior, tool markers
    take precedence over final-answer markers. This means that if a response contains
    both a tool call and a final answer, we first treat it as a tool round.
    """
    if "### Output Guidelines" in text:
        text = text.split("### Output Guidelines")[0].strip()
    if "** Input **" in text:
        text = text.replace("** Input **", "")
    if "** Output **" in text:
        text = text.replace("** Output **", "")
    if "### Reasoning Steps" in text:
        text = text.replace("### Reasoning Steps", "").strip()
    if "### Continue your reasoning" in text:
        text = text.replace("### Continue your reasoning", "").strip()
    text = text.strip()

    tool_index, tool_name = find_earliest_tool(text)
    if tool_name is not None:
        before_tool = text[:tool_index].strip()
        if tool_name in {"AskUser", "Search"}:
            tool_content = extract_first_parentheses_content(text[tool_index:])
            tool_content = tool_content.strip() if tool_content else ""
        else:
            tool_content = text[tool_index:].split("```python", 1)[1].split("```", 1)[0].strip()
            tool_content = f"```python\n{tool_content}\n```"
        return [
            {
                "name": "Reasoning Step",
                "type": "normal",
                "tool_name": None,
                "reasoning": before_tool,
            },
            {
                "name": "Reasoning Step",
                "type": "tool",
                "tool_name": tool_name,
                "reasoning": tool_content,
            },
        ]

    final_start, final_end, final_marker = find_earliest_final_marker(text)
    if final_marker is not None:
        reasoning_before = text[:final_start]
        reasoning_after = text[final_end:].lstrip(" :\n\t")
        return [
            {
                "name": "Reasoning Step",
                "type": "normal",
                "tool_name": None,
                "reasoning": reasoning_before.strip(),
            },
            {
                "name": "Final Response",
                "type": "normal",
                "tool_name": None,
                "reasoning": reasoning_after.strip(),
            },
        ]

    return [
        {
            "name": "Reasoning Step",
            "type": "normal",
            "tool_name": None,
            "reasoning": text,
        }
    ]


def format_steps(steps: list[dict[str, Any]]) -> str:
    """Serialize normalized steps back into plain text for the next reasoning round."""
    results: list[str] = []
    for step in steps:
        if step["type"] == "normal":
            results.append(step["reasoning"].strip())
        elif step["tool_name"] in {"AskUser", "Search"}:
            results.append(f"{step['tool_name']}({step['reasoning']})")
        elif step["tool_name"] == "Code":
            results.append(step["reasoning"])
        if "output" in step:
            results.append(f"- Tool Output: {step['output']}")
    return "\n".join(results)


def build_messages(instruction: str, input_text: str, method: str) -> list[dict[str, str]]:
    """Build chat-style messages so extraction/inference match the target prompting format."""
    if method == "mistral":
        return [{"role": "user", "content": instruction.strip() + "\n\n" + input_text.strip()}]
    if method == "llama":
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ]
    raise ValueError(f"Unsupported method: {method}")


def preprocess_dataset(data_path: Path, max_num: int, start_id: int, method: str) -> list[dict[str, Any]]:
    """Load inference examples and convert them into message-formatted records."""
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    max_num = len(data) if max_num == -1 else max_num
    dataset = []
    for entry in data[start_id:]:
        task = entry["input"].split("### Task", 1)[1].split("###", 1)[0].strip()
        dataset.append(
            {
                "input": build_messages(entry["instruction"], entry["input"], method),
                "ground_truth": entry["output"],
                "task": task,
            }
        )
        if len(dataset) >= max_num:
            break

    print(f"Length of data: {len(dataset)}")
    return dataset


def render_messages_without_template(messages: list[dict[str, str]]) -> str:
    """Fallback prompt renderer for tokenizers that do not ship a chat template."""
    if len(messages) == 1:
        return messages[0]["content"].strip()
    return "\n\n".join(message["content"].strip() for message in messages if message["content"].strip())


def parse_code_content(text: str) -> str:
    """Remove comments/prints before caching previously generated code for later rounds."""
    code = text.split("```python", 1)[1].split("```", 1)[0].strip()
    code_lines = code.split("\n")
    new_lines = []
    for line in code_lines:
        if line.strip().startswith("#"):
            continue
        if line.startswith("print("):
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


def build_error_record(
    task: str,
    ground_truth: str,
    steps: list[dict[str, Any]],
    raw: list[str],
    step_time: int,
    error: Exception,
    steering_diagnostics: list[dict[str, Any]],
    selected_layers: list[int],
) -> dict[str, Any]:
    """Store failed examples separately so success JSON stays clean and resumable."""
    return {
        "task": task,
        "ground_truth": ground_truth,
        "predict_partial": steps,
        "raw": raw,
        "step_time": step_time,
        "error": str(error),
        "error_type": type(error).__name__,
        "traceback": traceback.format_exc(),
        "steering_diagnostics": steering_diagnostics,
        "selected_steering_layers": selected_layers,
    }


def build_result_record(
    task: str,
    ground_truth: str,
    steps: list[dict[str, Any]],
    raw: list[str],
    steering_diagnostics: list[dict[str, Any]],
    selected_layers: list[int],
) -> dict[str, Any]:
    """Attach steering diagnostics to successful generations for later analysis."""
    return {
        "task": task,
        "predict": steps,
        "ground_truth": ground_truth,
        "raw": raw,
        "steering_diagnostics": steering_diagnostics,
        "selected_steering_layers": selected_layers,
    }


def load_model_and_tokenizer(model_name_or_path: str, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """Load the base LM used for both hidden-state inspection and generation."""
    target_device = torch.device("cuda" if device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="auto")
    model = model.to(target_device)
    model.eval()
    return tokenizer, model, target_device


def get_transformer_layers(model: AutoModelForCausalLM) -> list[torch.nn.Module]:
    """Return the decoder block list so steering hooks can be attached per layer."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported model architecture: could not locate transformer layers.")


def resolve_layer_map(layer_indices: list[int], num_model_layers: int) -> dict[int, int]:
    """Map saved steering-vector layer ids to the current model's internal layer indices."""
    if not layer_indices:
        raise ValueError("Steering payload does not contain any layer indices.")

    if min(layer_indices) == 1 and max(layer_indices) == num_model_layers:
        return {layer_index - 1: idx for idx, layer_index in enumerate(layer_indices)}
    if min(layer_indices) == 0 and max(layer_indices) == num_model_layers - 1:
        return {layer_index: idx for idx, layer_index in enumerate(layer_indices)}
    if 0 in layer_indices:
        raise ValueError("Embedding-layer steering is not supported during inference hooks.")
    if max(layer_indices) < num_model_layers:
        return {layer_index: idx for idx, layer_index in enumerate(layer_indices)}
    raise ValueError(
        f"Could not map steering layer indices {layer_indices[:5]}... to a model with {num_model_layers} layers."
    )


def parse_layer_range_spec(spec: str) -> tuple[int, int]:
    """Parse an inclusive layer range like `16-20` or a single layer like `18`."""
    text = spec.strip()
    if not text:
        raise ValueError("Layer range spec cannot be empty.")

    if "-" in text:
        start_text, end_text = text.split("-", 1)
        start = int(start_text.strip())
        end = int(end_text.strip())
    else:
        start = int(text)
        end = start

    if start > end:
        raise ValueError(f"Invalid layer range {spec!r}: start must be <= end.")
    return start, end


def load_steering_payload(path: Path) -> tuple[torch.Tensor, list[int]]:
    """Load steering vectors and their saved layer indices from disk."""
    payload = torch.load(path, map_location="cpu")
    if "steering_vectors" not in payload:
        raise ValueError(f"Steering payload at {path} is missing 'steering_vectors'.")
    steering_vectors = payload["steering_vectors"]
    layer_indices = payload.get("layer_indices")
    if layer_indices is None:
        raise ValueError(f"Steering payload at {path} is missing 'layer_indices'.")
    return steering_vectors, list(layer_indices)


def select_pairs_for_layers(layer_map: dict[int, int], selected_layers: list[int]) -> list[tuple[int, int]]:
    """Restrict the full steering payload to the exact model layers chosen for intervention."""
    missing_layers = [layer for layer in selected_layers if layer not in layer_map]
    if missing_layers:
        raise ValueError(f"Selected layers {missing_layers} are not available in the steering payload.")
    return [(layer_idx, layer_map[layer_idx]) for layer_idx in selected_layers]


def select_pairs_for_layer_range(
    layer_indices: list[int],
    layer_map: dict[int, int],
    layer_range_spec: str,
) -> dict[str, Any]:
    """
    Select intervention layers from an inclusive range over the payload's saved layer ids.

    Example: if the payload stores layers [1..32], then `16-20` selects those saved ids,
    which are mapped back to the model's internal block indices for hook registration.
    """
    start, end = parse_layer_range_spec(layer_range_spec)
    saved_to_model: dict[int, int] = {}
    for model_layer_idx, vector_idx in layer_map.items():
        saved_to_model[layer_indices[vector_idx]] = model_layer_idx

    selected_saved_layers = [layer_id for layer_id in layer_indices if start <= layer_id <= end]
    if not selected_saved_layers:
        raise ValueError(
            f"Layer range {layer_range_spec!r} does not overlap payload layers "
            f"{min(layer_indices)}-{max(layer_indices)}."
        )

    expected_saved_layers = list(range(start, end + 1))
    missing_saved_layers = [layer_id for layer_id in expected_saved_layers if layer_id not in saved_to_model]
    if missing_saved_layers:
        raise ValueError(
            f"Layer range {layer_range_spec!r} includes unavailable payload layers: {missing_saved_layers}"
        )

    selected_pairs = [(saved_to_model[layer_id], layer_map[saved_to_model[layer_id]]) for layer_id in expected_saved_layers]
    return {
        "selected_pairs": selected_pairs,
        "selected_layers": [model_layer_idx for model_layer_idx, _ in selected_pairs],
        "selected_saved_layers": expected_saved_layers,
    }


def build_prompt_ids(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    device: torch.device,
    max_seq_length: int,
) -> torch.Tensor:
    """Tokenize the current prompt and keep only the prompt-side tokens (no gold output)."""
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt_text = render_messages_without_template(messages)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt").input_ids

    if prompt_ids.shape[1] > max_seq_length:
        prompt_ids = prompt_ids[:, -max_seq_length:]
    return prompt_ids.to(device)


def extract_last_prompt_layer_states(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    selected_layers: list[int],
) -> torch.Tensor:
    """
    Extract decision-slot hidden states.

    The decision slot is the last prompt position, i.e. the representation used to
    predict the first token of the next assistant continuation. We reuse this same
    location for steering extraction, probe training, and probe-time gating.
    """
    attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
    with torch.inference_mode():
        outputs = model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    layer_states = []
    for model_layer_idx in selected_layers:
        hidden_state = outputs.hidden_states[model_layer_idx + 1][0, -1, :].detach().float().cpu()
        layer_states.append(hidden_state)
    return torch.stack(layer_states, dim=0) if layer_states else torch.empty(0)


def compute_last_prompt_layer_cosines(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    selected_pairs: list[tuple[int, int]],
    steering_vectors: torch.Tensor,
) -> dict[int, float]:
    """
    Compare current hidden states against steering directions.

    This is the older gating signal: high cosine means the model is already aligned
    with the tool-using direction; low cosine means it is far from that direction.
    """
    if not selected_pairs:
        return {}

    selected_layers = [layer_idx for layer_idx, _ in selected_pairs]
    layer_states = extract_last_prompt_layer_states(model, prompt_ids, selected_layers)
    cosines: dict[int, float] = {}
    for local_idx, (model_layer_idx, vector_idx) in enumerate(selected_pairs):
        hidden_state = layer_states[local_idx].float()
        vector = steering_vectors[vector_idx].to(hidden_state.device).float()
        cosine = F.cosine_similarity(hidden_state.unsqueeze(0), vector.unsqueeze(0), dim=1).item()
        cosines[model_layer_idx] = cosine
    return cosines


def aggregate_similarity(layer_cosines: dict[int, float]) -> float:
    """Average the per-layer cosine similarities into one gating score for the current step."""
    if not layer_cosines:
        return 0.0
    return sum(layer_cosines.values()) / len(layer_cosines)


def select_top_layers_by_vector_norm(
    layer_map: dict[int, int],
    steering_vectors: torch.Tensor,
    max_steering_layers: int,
) -> dict[str, Any]:
    """
    Pick intervention layers using steering-vector strength alone.

    We rank layers by the L2 norm of their steering vectors and keep the top-k layers.
    This keeps layer selection independent from the current evaluation dataset.
    """
    scored_pairs = []
    for model_layer_idx, vector_idx in layer_map.items():
        norm = torch.linalg.vector_norm(steering_vectors[vector_idx].float()).item()
        scored_pairs.append((model_layer_idx, vector_idx, norm))

    scored_pairs.sort(key=lambda item: item[2], reverse=True)
    selected = sorted(scored_pairs[:max_steering_layers], key=lambda item: item[0])
    selected_pairs = [(model_layer_idx, vector_idx) for model_layer_idx, vector_idx, _norm in selected]

    return {
        "selected_pairs": selected_pairs,
        "selected_layers": [layer_idx for layer_idx, _ in selected_pairs],
        "layer_norms": {str(layer_idx): norm for layer_idx, _vector_idx, norm in scored_pairs},
    }


def calibrate_cosine_gating(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    examples: list[dict[str, Any]],
    max_seq_length: int,
    selected_pairs: list[tuple[int, int]],
    steering_vectors: torch.Tensor,
    low_quantile: float,
    high_quantile: float,
) -> dict[str, float]:
    """Estimate low/high cosine thresholds on a small prompt subset for cosine-based gating."""
    aggregate_scores: list[float] = []

    for example in examples:
        prompt_ids = build_prompt_ids(tokenizer, example["input"], device, max_seq_length)
        layer_cosines = compute_last_prompt_layer_cosines(model, prompt_ids, selected_pairs, steering_vectors)
        if layer_cosines:
            aggregate_scores.append(sum(layer_cosines.values()) / len(layer_cosines))

    if aggregate_scores:
        score_tensor = torch.tensor(aggregate_scores, dtype=torch.float32)
        low_threshold = torch.quantile(score_tensor, low_quantile).item()
        high_threshold = torch.quantile(score_tensor, high_quantile).item()
    else:
        low_threshold = 0.0
        high_threshold = 0.2

    return {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "num_calibration_examples": len(examples),
    }


def compute_first_tool_light_repeat_heavy_scale(
    score: float,
    low_threshold: float,
    high_threshold: float,
    base_negative_scale: float,
    prior_tool_calls: int,
    first_tool_factor: float,
    repeat_tool_start_factor: float,
    repeat_tool_increment: float,
    repeat_tool_max_factor: float,
) -> tuple[float, float, float]:
    """
    Convert a gating score into a suppress-only steering scale.

    Behavior:
    - score <= low_threshold: no intervention
    - low_threshold < score < high_threshold: interpolate suppression strength
    - score >= high_threshold: use the strongest suppression

    Suppression uses a two-stage schedule:
    1. before the first tool call, apply a lighter factor to avoid blocking
       necessary single-tool usage
    2. after tools have already been used, ramp suppression up faster to make
       repeated tool calls progressively harder
    """
    if score <= low_threshold:
        return 0.0, 0.0, 1.0

    if high_threshold <= low_threshold:
        similarity_factor = 1.0
    elif score >= high_threshold:
        similarity_factor = 1.0
    else:
        similarity_factor = (score - low_threshold) / (high_threshold - low_threshold)

    if prior_tool_calls == 0:
        tool_count_factor = first_tool_factor
    else:
        repeat_index = prior_tool_calls - 1
        tool_count_factor = min(
            repeat_tool_start_factor + repeat_tool_increment * repeat_index,
            repeat_tool_max_factor,
        )
    scale = -base_negative_scale * similarity_factor * tool_count_factor
    return scale, similarity_factor, tool_count_factor


class ConditionalSteeringHookManager:
    """
    Manage forward hooks that add steering vectors only when gating enables them.

    The key idea is that the hooks are always registered on the selected layers, but
    the actual vector addition is conditional (`self.enabled`). This lets us inspect
    the current state first, then decide whether to intervene on the next forward pass.
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        steering_vectors: torch.Tensor,
        selected_pairs: list[tuple[int, int]],
        strength: float,
        application_mode: str,
    ) -> None:
        self.scale = 0.0
        self.handles: list[Any] = []
        self.strength = strength
        self.application_mode = application_mode
        self.layers = get_transformer_layers(model)
        self.vectors = steering_vectors.cpu()
        self.selected_pairs = selected_pairs

        for model_layer_idx, vector_idx in self.selected_pairs:
            vector = self.vectors[vector_idx]
            handle = self.layers[model_layer_idx].register_forward_hook(self._make_hook(vector))
            self.handles.append(handle)

    def set_scale(self, scale: float) -> None:
        """
        Set the steering multiplier for the next model forward.

        `scale = 0` leaves the model unchanged,
        `scale < 0` suppresses tool use (-v).
        """
        self.scale = scale

    def _make_hook(self, vector: torch.Tensor):
        def hook(_module, _inputs, output):
            if self.scale == 0.0:
                return output

            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return output

            # Steering is implemented as an additive activation intervention.
            delta = vector.to(device=hidden.device, dtype=hidden.dtype) * self.strength * self.scale
            if self.application_mode == "last_prompt":
                # Preferred setting: only perturb the decision slot before the next step is generated.
                if hidden.shape[1] > 1:
                    hidden = hidden.clone()
                    hidden[:, -1, :] += delta
            elif self.application_mode == "all_tokens":
                # Less controlled alternative: add the steering direction to all token positions.
                hidden = hidden.clone()
                hidden += delta.view(1, 1, -1)
            else:
                raise ValueError(f"Unsupported steering application mode: {self.application_mode}")

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class LinearProbe(nn.Module):
    """Minimal binary classifier used for step-level tool-decision probing."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class ToolDecisionProbe:
    """
    Wrap a trained step-level tool-use probe.

    The probe operates on the same decision-slot hidden states used elsewhere in this
    project. At inference time it predicts whether the *next* step should call a tool.
    We use that probability as the primary gating signal when a probe checkpoint is given.
    """
    def __init__(self, checkpoint_path: Path, threshold_override: float | None = None) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.checkpoint_path = checkpoint_path
        self.feature_key = payload["feature_key"]
        self.per_layer_index = payload.get("per_layer_index")
        self.selected_layers = list(payload.get("selected_layers") or [])
        self.feature_mean = payload["feature_mean"].float()
        self.feature_std = payload["feature_std"].float()
        self.threshold = threshold_override if threshold_override is not None else float(payload["threshold"])
        self.model = LinearProbe(int(payload["input_dim"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def _build_feature(self, layer_states: torch.Tensor) -> torch.Tensor:
        """Project extracted layer states into the exact feature format used during probe training."""
        if self.feature_key == "X_mean":
            feature = layer_states.mean(dim=0)
        elif self.feature_key == "X_concat":
            feature = layer_states.reshape(-1)
        elif self.feature_key == "X_per_layer":
            if self.per_layer_index is None:
                raise ValueError("Probe checkpoint uses X_per_layer but does not store per_layer_index.")
            feature = layer_states[self.per_layer_index]
        else:
            raise ValueError(f"Unsupported probe feature key: {self.feature_key}")

        feature = feature.float()
        return (feature - self.feature_mean) / self.feature_std

    def predict_tool_probability(self, layer_states: torch.Tensor) -> float:
        """Return P(call_tool_now | current decision-slot hidden state)."""
        feature = self._build_feature(layer_states)
        with torch.inference_mode():
            logit = self.model(feature.unsqueeze(0)).squeeze(0)
        return torch.sigmoid(logit).item()


def resolve_probe_thresholds(
    probe: ToolDecisionProbe,
    low_override: float | None,
    high_override: float | None,
    margin: float,
) -> tuple[float, float]:
    """
    Convert a single probe threshold into a three-way gating band.

    By default we center the neutral zone around the threshold selected during
    probe validation. In the suppress-only setup we reuse this band as:
    - low probability  -> no intervention
    - middle band      -> partial suppression
    - high probability -> stronger suppression
    """
    base = probe.threshold
    low = low_override if low_override is not None else max(0.0, base - margin)
    high = high_override if high_override is not None else min(1.0, base + margin)
    if low > high:
        raise ValueError(f"Invalid probe thresholds: low={low} is greater than high={high}.")
    return low, high


def generate_assistant_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """Run one assistant generation round from the current prompt state."""
    attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
    generation_kwargs = {
        "input_ids": prompt_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)

    new_tokens = generated[0, prompt_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def append_reasoning(messages: list[dict[str, str]], new_steps: list[dict[str, Any]]) -> None:
    """Append the parsed round output back into the user-visible running trace."""
    messages[-1]["content"] = (
        messages[-1]["content"].strip()
        + "\n"
        + format_steps(new_steps).strip()
        + "\n\n### Continue your reasoning\n"
    )


def inference(args: argparse.Namespace) -> None:
    """
    Main inference loop.

    High-level flow:
    1. Select steering layers.
    2. Optionally load a trained step-level tool-decision probe.
    3. For each round, inspect the decision-slot hidden state first.
    4. Use either the probe score or cosine score to decide whether to enable steering.
    5. Generate, parse, execute tools if needed, and continue until final response or timeout.
    """
    print("Loading model and tokenizer...")
    tokenizer, model, device = load_model_and_tokenizer(args.model_name_or_path, args.device)
    generation_max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else args.max_seq_length

    print("Loading steering payload...")
    steering_vectors, layer_indices = load_steering_payload(args.steering_vector_path)
    model_layers = get_transformer_layers(model)
    layer_map = resolve_layer_map(layer_indices, len(model_layers))

    print("Loading and preprocessing dataset...")
    dataset = preprocess_dataset(
        data_path=args.data_path,
        max_num=args.max_test_num,
        start_id=args.test_start_id,
        method=args.method,
    )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    args.code_exec_dir.mkdir(parents=True, exist_ok=True)
    error_save_path = args.error_save_path or args.save_path.with_name(args.save_path.stem + "_errors.json")
    error_save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_path.exists():
        with args.save_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        existing_tasks = {result["task"] for result in results}
    else:
        results = []
        existing_tasks = set()

    if error_save_path.exists():
        with error_save_path.open("r", encoding="utf-8") as f:
            error_results = json.load(f)
        existing_error_tasks = {result["task"] for result in error_results}
    else:
        error_results = []
        existing_error_tasks = set()

    print(f"Length of existing data: {len(results)}")
    print(f"Length of existing errors: {len(error_results)}")

    pending_examples = [example for example in dataset if example["task"] not in existing_tasks]
    if not pending_examples:
        print("No pending examples to process.")
        return

    if args.steering_layer_range is not None:
        selection_summary = select_pairs_for_layer_range(
            layer_indices=layer_indices,
            layer_map=layer_map,
            layer_range_spec=args.steering_layer_range,
        )
    else:
        selection_summary = select_top_layers_by_vector_norm(
            layer_map=layer_map,
            steering_vectors=steering_vectors,
            max_steering_layers=args.max_steering_layers,
        )

    selected_pairs = selection_summary["selected_pairs"]
    selected_layers = selection_summary["selected_layers"]
    selected_saved_layers = selection_summary.get("selected_saved_layers", [])
    probe = None
    gating_mode = "similarity"
    probe_low_threshold = None
    probe_high_threshold = None

    if args.tool_decision_probe_path is not None:
        # Probe gating is preferred when available: it predicts whether this step should call a tool.
        probe = ToolDecisionProbe(
            checkpoint_path=args.tool_decision_probe_path,
            threshold_override=args.probe_threshold_override,
        )
        if probe.selected_layers and args.steering_layer_range is None:
            # Keep intervention/extraction layers consistent with the layers seen during probe training.
            selected_layers = probe.selected_layers
        gating_mode = "probe"
        probe_low_threshold, probe_high_threshold = resolve_probe_thresholds(
            probe=probe,
            low_override=args.probe_low_threshold,
            high_override=args.probe_high_threshold,
            margin=args.probe_neutral_margin,
        )

    selected_pairs = select_pairs_for_layers(layer_map, selected_layers)
    selected_layers = [layer_idx for layer_idx, _ in selected_pairs]
    selected_saved_layers = [layer_indices[vector_idx] for _layer_idx, vector_idx in selected_pairs]

    low_threshold = None
    high_threshold = None
    if gating_mode == "similarity":
        if (args.similarity_low_threshold is None) != (args.similarity_high_threshold is None):
            raise ValueError(
                "Please set both --similarity-low-threshold and --similarity-high-threshold, or leave both unset."
            )
        if (
            args.similarity_low_threshold is not None
            and args.similarity_high_threshold is not None
            and args.similarity_low_threshold > args.similarity_high_threshold
        ):
            raise ValueError(
                "Invalid similarity thresholds: "
                f"low={args.similarity_low_threshold} is greater than high={args.similarity_high_threshold}."
            )

        if args.similarity_low_threshold is not None and args.similarity_high_threshold is not None:
            low_threshold = args.similarity_low_threshold
            high_threshold = args.similarity_high_threshold
            print(
                "Using manual similarity thresholds with layers:",
                selected_saved_layers or selected_layers,
                "| similarity_low_threshold=",
                round(low_threshold, 4),
                "| similarity_high_threshold=",
                round(high_threshold, 4),
            )
        else:
            calibration_examples = pending_examples[: min(len(pending_examples), args.cosine_calibration_samples)]
            gating_summary = calibrate_cosine_gating(
                model=model,
                tokenizer=tokenizer,
                device=device,
                examples=calibration_examples,
                max_seq_length=args.max_seq_length,
                selected_pairs=selected_pairs,
                steering_vectors=steering_vectors,
                low_quantile=args.cosine_low_quantile,
                high_quantile=args.cosine_high_quantile,
            )
            low_threshold = gating_summary["low_threshold"]
            high_threshold = gating_summary["high_threshold"]
            print(
                "Using calibrated similarity thresholds with layers:",
                selected_saved_layers or selected_layers,
                "| similarity_low_threshold=",
                round(low_threshold, 4),
                "| similarity_high_threshold=",
                round(high_threshold, 4),
            )
    else:
        print(
            "Using tool-decision probe gating with layers:",
            selected_saved_layers or selected_layers,
            "| probe_low_threshold=",
            round(probe_low_threshold, 4),
            "| probe_high_threshold=",
            round(probe_high_threshold, 4),
        )

    hook_manager = ConditionalSteeringHookManager(
        model=model,
        steering_vectors=steering_vectors,
        selected_pairs=selected_pairs,
        strength=args.steering_strength,
        application_mode=args.steering_application,
    )

    log = {"fail": 0, "success": 0}

    try:
        example_count = 0
        for example in tqdm(dataset):
            input_messages = deepcopy(example["input"])
            ground_truth = example["ground_truth"]
            task = example["task"]

            example_count += 1
            code_file = args.code_exec_dir / f"{example_count}_{task[:3]}.py"

            if task in existing_tasks:
                continue

            steps: list[dict[str, Any]] = []
            raw: list[str] = []
            steering_diagnostics: list[dict[str, Any]] = []
            step_time = 0
            all_previous_code = ""

            while True:
                try:
                    step_time += 1
                    if step_time > args.max_steps:
                        steps.append(
                            {
                                "name": "Final Response",
                                "type": "normal",
                                "tool_name": None,
                                "reasoning": TIMEOUT_FALLBACK,
                            }
                        )
                        results.append(
                            build_result_record(
                                task=task,
                                ground_truth=ground_truth,
                                steps=steps,
                                raw=raw,
                                steering_diagnostics=steering_diagnostics,
                                selected_layers=selected_layers,
                            )
                        )
                        save_json(args.save_path, results)
                        break

                    prompt_ids = build_prompt_ids(tokenizer, input_messages, device, args.max_seq_length)
                    hook_manager.set_scale(0.0)
                    prior_tool_calls = sum(1 for step in steps if step.get("type") == "tool")

                    if gating_mode == "probe":
                        # Probe-based gating:
                        # suppress-only:
                        # high probability -> stronger -v
                        # low probability  -> no intervention
                        layer_states = extract_last_prompt_layer_states(model, prompt_ids, selected_layers)
                        tool_probability = probe.predict_tool_probability(layer_states)
                        steering_scale, similarity_factor, tool_count_factor = compute_first_tool_light_repeat_heavy_scale(
                            score=tool_probability,
                            low_threshold=probe_low_threshold,
                            high_threshold=probe_high_threshold,
                            base_negative_scale=args.negative_steering_scale,
                            prior_tool_calls=prior_tool_calls,
                            first_tool_factor=args.first_tool_factor,
                            repeat_tool_start_factor=args.repeat_tool_start_factor,
                            repeat_tool_increment=args.repeat_tool_increment,
                            repeat_tool_max_factor=args.repeat_tool_max_factor,
                        )
                        if steering_scale < 0.0:
                            steering_decision = "suppress_tool_high_probability"
                            apply_steering = True
                        else:
                            steering_decision = "skip_low_probability"
                            steering_scale = 0.0
                            apply_steering = False

                        steering_diagnostics.append(
                            {
                                "step": step_time,
                                "gating_mode": "probe",
                                "tool_probability": tool_probability,
                                "probe_low_threshold": probe_low_threshold,
                                "probe_high_threshold": probe_high_threshold,
                                "decision": steering_decision,
                                "steering_applied": apply_steering,
                                "steering_scale": steering_scale,
                                "prior_tool_calls": prior_tool_calls,
                                "similarity_factor": similarity_factor,
                                "tool_count_factor": tool_count_factor,
                                "first_tool_factor": args.first_tool_factor,
                                "repeat_tool_start_factor": args.repeat_tool_start_factor,
                                "repeat_tool_increment": args.repeat_tool_increment,
                                "repeat_tool_max_factor": args.repeat_tool_max_factor,
                            }
                        )
                    else:
                        # Similarity-based suppress-only gating:
                        # high similarity -> stronger -v
                        # low similarity  -> no intervention
                        layer_cosines = compute_last_prompt_layer_cosines(
                            model=model,
                            prompt_ids=prompt_ids,
                            selected_pairs=selected_pairs,
                            steering_vectors=steering_vectors,
                        )
                        aggregate_cosine = aggregate_similarity(layer_cosines)

                        steering_scale, similarity_factor, tool_count_factor = compute_first_tool_light_repeat_heavy_scale(
                            score=aggregate_cosine,
                            low_threshold=low_threshold,
                            high_threshold=high_threshold,
                            base_negative_scale=args.negative_steering_scale,
                            prior_tool_calls=prior_tool_calls,
                            first_tool_factor=args.first_tool_factor,
                            repeat_tool_start_factor=args.repeat_tool_start_factor,
                            repeat_tool_increment=args.repeat_tool_increment,
                            repeat_tool_max_factor=args.repeat_tool_max_factor,
                        )
                        if steering_scale < 0.0:
                            steering_decision = "apply_negative_high_similarity"
                            apply_steering = True
                        else:
                            steering_decision = "skip_low_similarity"
                            steering_scale = 0.0
                            apply_steering = False

                        steering_diagnostics.append(
                            {
                                "step": step_time,
                                "gating_mode": "similarity",
                                "aggregate_similarity": aggregate_cosine,
                                "layer_cosines": {str(k): v for k, v in layer_cosines.items()},
                                "selected_saved_layers": selected_saved_layers,
                                "selected_model_layers": selected_layers,
                                "low_threshold": low_threshold,
                                "high_threshold": high_threshold,
                                "decision": steering_decision,
                                "steering_applied": apply_steering,
                                "steering_scale": steering_scale,
                                "prior_tool_calls": prior_tool_calls,
                                "similarity_factor": similarity_factor,
                                "tool_count_factor": tool_count_factor,
                                "first_tool_factor": args.first_tool_factor,
                                "repeat_tool_start_factor": args.repeat_tool_start_factor,
                                "repeat_tool_increment": args.repeat_tool_increment,
                                "repeat_tool_max_factor": args.repeat_tool_max_factor,
                            }
                        )

                    hook_manager.set_scale(steering_scale)

                    if gating_mode == "probe":
                        tool_probability = steering_diagnostics[-1]["tool_probability"]
                        log_message(
                            (
                                f"[steering] decision={steering_decision} "
                                f"tool_probability={tool_probability:.4f} scale={steering_scale:.2f}"
                            ),
                            args.quiet,
                        )
                    else:
                        aggregate_cosine = steering_diagnostics[-1]["aggregate_similarity"]
                        log_message(
                            (
                                f"[steering] decision={steering_decision} "
                                f"aggregate_similarity={aggregate_cosine:.4f} scale={steering_scale:.2f}"
                            ),
                            args.quiet,
                        )

                    assistant_output = generate_assistant_output(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_ids,
                        max_new_tokens=generation_max_new_tokens,
                    )
                    hook_manager.set_scale(0.0)
                    raw.append(assistant_output)

                    log_message("\n" + "+" * 10 + " Round Response " + "+" * 10, args.quiet)
                    log_message(assistant_output, args.quiet)

                    new_steps = parse_steps(assistant_output)

                    if new_steps[-1]["name"] == "Final Response":
                        log["success"] += 1
                        log_message("\n" + "+" * 10 + " Ground Truth " + "+" * 10, args.quiet)
                        log_message(ground_truth, args.quiet)

                        steps.extend(new_steps)
                        results.append(
                            build_result_record(
                                task=task,
                                ground_truth=ground_truth,
                                steps=steps,
                                raw=raw,
                                steering_diagnostics=steering_diagnostics,
                                selected_layers=selected_layers,
                            )
                        )
                        save_json(args.save_path, results)
                        log_message(str(log), args.quiet)
                        break

                    if new_steps[-1]["type"] == "tool":
                        tool_name = new_steps[-1]["tool_name"]
                        if tool_name == "AskUser":
                            log_message("AskUser tool detected", args.quiet)
                            response = simulate_user_response(task, new_steps[-1]["reasoning"])
                            new_steps[-1]["output"] = response
                        elif tool_name == "Search":
                            log_message("Search tool detected", args.quiet)
                            link = "intention" in str(args.data_path)
                            response = search_serper(new_steps[-1]["reasoning"], link=link, num=args.search_top_k)
                            new_steps[-1]["output"] = response
                        elif tool_name == "Code":
                            log_message("Code tool detected", args.quiet)
                            code_content = new_steps[-1]["reasoning"].split("```python", 1)[1].split("```", 1)[0].strip()
                            code_content = f"```python\n{all_previous_code}\n{code_content}\n```"
                            log_message(" ====== Code Content ====== ", args.quiet)
                            log_message(code_content, args.quiet)
                            response = execute_code(code_content, code_file)
                            if not response.startswith("Error"):
                                new_code = parse_code_content(new_steps[-1]["reasoning"])
                                all_previous_code += new_code + "\n"
                            new_steps[-1]["output"] = response
                        else:
                            raise AssertionError(f"Unknown tool name: {tool_name}")

                        log_message("\n" + "+" * 10 + f" Tool {tool_name} Response " + "+" * 10, args.quiet)
                        log_message(response, args.quiet)

                    append_reasoning(input_messages, new_steps)
                    steps.extend(new_steps)

                except Exception as exc:
                    hook_manager.set_scale(0.0)
                    log["fail"] += 1
                    tqdm.write(str(exc))
                    if task not in existing_error_tasks:
                        error_results.append(
                            build_error_record(
                                task=task,
                                ground_truth=ground_truth,
                                steps=steps,
                                raw=raw,
                                step_time=step_time,
                                error=exc,
                                steering_diagnostics=steering_diagnostics,
                                selected_layers=selected_layers,
                            )
                        )
                        existing_error_tasks.add(task)
                        save_json(error_save_path, error_results)
                    break
    finally:
        hook_manager.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tool-prompt inference with gated activation steering.")
    parser.add_argument("--model-name-or-path", required=True, help="Local model path or HF model id.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the inference dataset.")
    parser.add_argument("--steering-vector-path", type=Path, required=True, help="Path to the .pt steering payload.")
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save successful inference results.")
    parser.add_argument("--error-save-path", type=Path, default=None, help="Path to save failed examples and error traces.")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum prompt length for generation.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of new tokens per round. "
            "Defaults to max_seq_length to better match the original Open-SMARTAgent setup."
        ),
    )
    parser.add_argument("--test-start-id", type=int, default=0, help="The start id for testing.")
    parser.add_argument("--max-test-num", type=int, default=-1, help="The max number of instances to test.")
    parser.add_argument("--method", choices=("mistral", "llama"), default="llama", help="Prompt packing method.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Device selection.")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum number of tool-interaction rounds per example.")
    parser.add_argument("--search-top-k", type=int, default=3, help="Number of search results to keep per tool call.")
    parser.add_argument("--steering-strength", type=float, default=1.0, help="Scalar multiplier applied to steering vectors.")
    parser.add_argument(
        "--steering-application",
        choices=("last_prompt", "all_tokens"),
        default="last_prompt",
        help="Where to apply steering during each forward pass.",
    )
    parser.add_argument(
        "--steering-layer-range",
        type=str,
        default=None,
        help=(
            "Inclusive range over saved steering-payload layer ids, for example `16-20`. "
            "If set, these layers are used instead of the automatic top-k layer sweep."
        ),
    )
    parser.add_argument(
        "--max-steering-layers",
        type=int,
        default=3,
        help="Maximum number of layers selected by the layer sweep.",
    )
    parser.add_argument(
        "--tool-decision-probe-path",
        type=Path,
        default=None,
        help="Optional trained step-level tool-decision probe checkpoint. If set, it overrides cosine gating.",
    )
    parser.add_argument(
        "--probe-threshold-override",
        type=float,
        default=None,
        help="Legacy center-threshold override. Still loaded into the probe checkpoint before three-way thresholds are derived.",
    )
    parser.add_argument(
        "--probe-low-threshold",
        type=float,
        default=None,
        help="If set, probabilities at or below this threshold suppress tool use (-v).",
    )
    parser.add_argument(
        "--probe-high-threshold",
        type=float,
        default=None,
        help="If set, probabilities at or above this threshold receive the strongest suppression.",
    )
    parser.add_argument(
        "--probe-neutral-margin",
        type=float,
        default=0.15,
        help="If explicit low/high thresholds are not set, build a neutral band around the probe threshold using this margin.",
    )
    parser.add_argument(
        "--negative-steering-scale",
        type=float,
        default=1.0,
        help="Base multiplier used for suppress-only steering (-v).",
    )
    parser.add_argument(
        "--first-tool-factor",
        type=float,
        default=0.35,
        help="Suppression factor used before any tool has been called in the current trajectory.",
    )
    parser.add_argument(
        "--repeat-tool-start-factor",
        type=float,
        default=1.0,
        help="Suppression factor used once the model has already called one tool.",
    )
    parser.add_argument(
        "--repeat-tool-increment",
        type=float,
        default=0.4,
        help="Additional suppression added for each extra prior tool call after the first.",
    )
    parser.add_argument(
        "--repeat-tool-max-factor",
        type=float,
        default=2.0,
        help="Maximum suppression factor allowed for repeated tool calls.",
    )
    parser.add_argument(
        "--similarity-low-threshold",
        type=float,
        default=None,
        help="If set together with --similarity-high-threshold, scores at or below this value skip suppression.",
    )
    parser.add_argument(
        "--similarity-high-threshold",
        type=float,
        default=None,
        help="If set together with --similarity-low-threshold, scores at or above this value receive the strongest suppression.",
    )
    parser.add_argument(
        "--cosine-calibration-samples",
        type=int,
        default=32,
        help="Number of pending prompts used to calibrate similarity thresholds when manual thresholds are not provided.",
    )
    parser.add_argument(
        "--cosine-low-quantile",
        type=float,
        default=0.33,
        help="Fallback calibration quantile below which suppress-only steering stays off.",
    )
    parser.add_argument(
        "--cosine-high-quantile",
        type=float,
        default=0.67,
        help="Fallback calibration quantile above which suppress-only steering reaches full strength.",
    )
    parser.add_argument(
        "--code-exec-dir",
        type=Path,
        default=Path("/data/yuqi/Steering-tooloveruse/inference/env"),
        help="Temporary directory used for executing model-generated code.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step generation and tool logs so the progress bar stays readable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    inference(parse_args())
