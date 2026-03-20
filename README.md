# Steering-tooloveruse

`Steering-tooloveruse` is a local workspace for building tool-use steering vectors and applying them during tool-prompt inference.

This project currently supports:

- building tool-vs-no-tool steering pairs
- converting pairs into the Open-SMARTAgent tool-prompt format
- extracting steering vectors from the last prompt token hidden states
- running steered inference with `Search`, `AskUser`, and `Code` tools
- evaluating saved inference result files

## Directory Layout

```text
Steering-tooloveruse/
├── data_inference/              # Inference datasets copied from Open-SMARTAgent
├── evaluate/                    # Evaluation scripts that judge saved result files
├── inference/                   # Steered inference script and tool helpers
├── model_weights/               # Local model checkpoints
├── scripts/                     # Data preparation and steering-vector builders
├── steering_data/               # Steering pair datasets
├── steering_vector/             # Saved steering vectors and summaries
└── secret.json                  # API keys for OpenAI and Serper
```

## Setup

Recommended Python environment:

```bash
/home/yche1052/miniconda3/envs/smart/bin/python
```

The project uses a local `secret.json` at the repo root. Expected fields:

```json
{
  "api_key": "...",
  "base_url": "...",
  "serper_key": "..."
}
```

These keys are used by:

- `inference/utils_askuser.py`
- `inference/utils_serper.py`
- `evaluate/inference_eval_*.py`

## End-to-End Pipeline

### 1. Build Steering Pairs

Select paired examples where one trajectory explicitly uses a tool and the matched trajectory does not.

```bash
/home/yche1052/miniconda3/envs/smart/bin/python \
  scripts/build_tool_steering_pairs.py
```

Main outputs:

- `steering_data/tool_steering_pairs_exact_sample.json`
- `steering_data/tool_steering_pairs_summary.json`

### 2. Convert Pairs to Tool-Prompt Format

Align the pair data to the same prompt style used by Open-SMARTAgent tool inference.

```bash
/home/yche1052/miniconda3/envs/smart/bin/python \
  scripts/convert_tool_steering_pairs_to_tool_prompt.py
```

Main outputs:

- `steering_data/tool_steering_pairs_tool_prompt_aligned.json`
- `steering_data/tool_steering_pairs_tool_prompt_aligned_summary.json`

### 3. Build Steering Vectors

This script computes steering vectors from the last prompt token hidden states, not from the first output token hidden states.

```bash
CUDA_VISIBLE_DEVICES=0 /home/yche1052/miniconda3/envs/smart/bin/python \
  scripts/build_steering_vector.py \
  --model-name-or-path /data/yuqi/Steering-tooloveruse/model_weights/Llama-3.1-8B-Instruct \
  --data-path /data/yuqi/Steering-tooloveruse/steering_data/tool_steering_pairs_tool_prompt_aligned.json \
  --save-path /data/yuqi/Steering-tooloveruse/steering_vector/last_prompt_steering_vector.pt \
  --summary-path /data/yuqi/Steering-tooloveruse/steering_vector/last_prompt_steering_vector_summary.json \
  --method llama \
  --device cuda
```

Main outputs:

- `steering_vector/last_prompt_steering_vector.pt`
- `steering_vector/last_prompt_steering_vector_summary.json`

### 4. Run Steered Inference

This script reproduces the Open-SMARTAgent tool-prompt inference loop, but injects a steering vector into the selected transformer layers during generation.

```bash
CUDA_VISIBLE_DEVICES=0 /home/yche1052/miniconda3/envs/smart/bin/python \
  inference/inference_tool_prompt_steered.py \
  --model-name-or-path /data/yuqi/Steering-tooloveruse/model_weights/Llama-3.1-8B-Instruct \
  --data-path /data/yuqi/Steering-tooloveruse/data_inference/domain_math_tool_prompt.json \
  --steering-vector-path /data/yuqi/Steering-tooloveruse/steering_vector/last_prompt_steering_vector.pt \
  --save-path /data/yuqi/Steering-tooloveruse/inference_results/domain_math_tool_prompt_steered.json \
  --method llama \
  --device cuda \
  --steering-strength 1.0 \
  --steering-application last_prompt
```

Important flags:

- `--steering-strength`: scales the injected steering vector
- `--steering-application last_prompt`: inject only on the last prompt position during the prefill forward pass
- `--max-test-num`: useful for smoke tests
- `--max-new-tokens`: limits per-round generation length

The saved inference format is compatible with the evaluation scripts:

- `task`
- `predict`
- `ground_truth`
- `raw`

## Evaluation

The evaluation scripts judge saved inference result files rather than re-running the model.

Available scripts:

- `evaluate/inference_eval_math.py`
- `evaluate/inference_eval_time.py`
- `evaluate/inference_eval_intention.py`

These scripts currently contain a placeholder:

```python
data_path = "PATH/TO/INFERENCE/DATA.json"
```

Before running evaluation, edit that line in the relevant script to point to your saved inference JSON.

Example:

```bash
cd /data/yuqi/Steering-tooloveruse/evaluate
/home/yche1052/miniconda3/envs/smart/bin/python inference_eval_math.py
```

The evaluation output will be written next to the inference file as:

```text
<inference_file>_judge.json
```

## Data Files

Copied inference datasets are stored under `data_inference/`.

Current `*_tool_prompt.json` sample counts:

- `domain_intention_tool_prompt.json`: 100
- `domain_math_tool_prompt.json`: 400
- `domain_time_tool_prompt.json`: 100
- `ood_gsm_tool_prompt.json`: 1319
- `ood_mint_tool_prompt.json`: 97

Total: `2016`

## Notes

### Environment Selection

If VS Code keeps selecting `/data/yuqi/.venv/bin/python`, open a new terminal after reloading the workspace. The workspace is configured to prefer:

```text
/home/yche1052/miniconda3/envs/smart/bin/python
```

### Code Tool Behavior

`inference/utils_code.py` injects a small standard-library preamble before executing model-generated code. This reduces failures caused by missing imports such as `import math`.

### Steering Stability

If steered outputs become too aggressive or start repeating tool-related patterns, try:

- lowering `--steering-strength`
- using `--steering-application last_prompt`
- running a smaller `--max-test-num` smoke test first

## Related Files

- `scripts/build_tool_steering_pairs.py`
- `scripts/convert_tool_steering_pairs_to_tool_prompt.py`
- `scripts/build_steering_vector.py`
- `inference/inference_tool_prompt_steered.py`
- `evaluate/inference_eval_math.py`
