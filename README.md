# SteeringMark

**SteeringMark** studies and mitigates **tool overuse** in tool-augmented LLM agents through
activation steering. Given a tool-use prompt where a model is offered `Search`, `Code`, and
`AskUser` tools, agents often reach for a tool even when they could answer from their own
reasoning. SteeringMark builds *tool-vs-no-tool* steering vectors from the hidden states at
step-heading tokens (e.g. `### Reasoning`, `### Search`, `### Code`, `### AskUser`), then
injects those vectors during KV-cache inference to **suppress unnecessary tool calls** while
preserving the model's task performance.

This repository is **code-only**. Input datasets, model weights, extracted steering-vector
payloads (`*.pt`), inference outputs, evaluation tables, figures, and reports are produced or
supplied locally and are intentionally not tracked (see `.gitignore`).

## Method Overview

The pipeline has four stages:

1. **Build steering data** — assemble matched full trajectories, one that uses a given tool and
   a paired one that does not, across the `math`, `time`, and `intention` domains.
2. **Extract steering vectors** — run the base model over the trajectories and average hidden
   states at `###` step-heading token positions to form per-tool steering directions
   (`all`, `search`, `code`, `askuser`, `search_askuser`), saved as `*.pt` payloads per layer.
3. **Steered inference** — during generation, once a `###` heading prefix becomes active, add
   the (negated / scaled) steering vector to the residual stream via the KV cache to suppress
   tool-specific headings. Several variants implement different suppression rules (cosine-gated,
   deviation-scaled, orthogonalized, QR-subspace-projected, strict-instruction).
4. **Evaluate** — judge saved result files for tool usage, correctness, and timing, and produce
   ablation tables and figures.

## Repository Layout

```text
SteeringMark/
├── tool_schema_utils.py          # Shared schema/heading parsing helpers (markdown & json action)
├── steering_/
│   └── build_steering_data.py    # Build paired tool / no-tool full trajectories
├── steering_vector/
│   ├── build_step_mark_steering_vectors.py      # Extract steering vectors at heading tokens
│   ├── build_step_mark_qr_subspace_payloads.py  # Build QR-subspace projection payloads
│   └── train_domain_heading_tool_classifier.py  # Train a heading-time tool-use classifier
├── inference/
│   ├── inference_tool_prompt_tag_suppressed_kvcache.py        # Main steered inference
│   ├── inference_tool_prompt_tag_suppressed_kvcache_gpu.py    # GPU variant
│   ├── inference_tool_prompt_tag_suppressed_cosine_gated_kvcache.py
│   ├── inference_tool_prompt_tag_suppressed_cosine_deviation_scaled_kvcache.py
│   ├── inference_tool_prompt_tag_suppressed_kvcache_strict_instruction.py
│   ├── inference_tool_prompt_tag_orthogonalized_kvcache.py
│   ├── inference_tool_prompt_tag_qr_subspace_projected_kvcache.py
│   ├── utils_code.py             # Sandboxed Python execution for the Code tool
│   ├── utils_serper.py           # Web search for the Search tool (Serper.dev)
│   ├── utils_askuser.py          # Simulated user replies for the AskUser tool
│   └── utils_heading_classifier.py
├── experiments/
│   ├── run_ablation.py           # Cross-format / heading-rename / matched-prompt ablations
│   ├── run_matched_prompt_baseline.py
│   └── run_llama70b_alt_extraction_baseline.py
├── evaluate/
│   ├── inference_eval_math.py    # LLM-judge evaluation per domain
│   ├── inference_eval_time.py
│   ├── inference_eval_intention.py
│   ├── merge_json_parts.py       # Merge sharded inference outputs
│   ├── plot_*.py                 # Figure generation (bar / violin / trend / demo)
│   ├── select_*.py               # Example selection utilities
│   └── figures/similarity_figure/*.py   # Steering-similarity analysis & plots
├── scripts/
│   └── run_cross_format_ablation.sh     # Convenience launcher for the cross-format ablation
└── secret.json.example           # Template for API keys (copy to secret.json)
```

## Setup

### Requirements

- Python 3.11
- `torch`, `transformers`, `numpy`, `tqdm`
- `openai` (LLM-judge evaluation and the AskUser tool)
- `matplotlib` (figure generation)
- A base model checkpoint accessible locally (e.g. `Llama-3.1-8B-Instruct`); the scripts also
  support Mistral-7B, Mistral-Nemo-12B, Mistral-Small-24B, and Llama-3-70B.

### API keys

Several scripts (`inference/utils_serper.py`, `inference/utils_askuser.py`,
`evaluate/inference_eval_*.py`) read a `secret.json` at the repository root:

```bash
cp secret.json.example secret.json
```

```json
{
  "api_key": "...",       // OpenAI-compatible key for the judge / AskUser model
  "base_url": "...",      // OpenAI-compatible base URL
  "serper_key": "..."     // Serper.dev key for web Search
}
```

`secret.json` is git-ignored — never commit real keys.

### Data & weights

Input datasets (`data_raw/`, `data_train/`, `data_inference/`, `steering_/*.json`), model
checkpoints, and extracted `*.pt` steering payloads are not part of this repository. Point the
scripts at your local copies via the CLI flags below (many defaults reference absolute local
paths and should be overridden for your environment).

## Pipeline

Paths below are illustrative — adjust `--model_name_or_path`, data paths, and output
directories to your setup.

### 1. Build steering data

```bash
python steering_/build_steering_data.py
```

Reads the raw domain trajectories and emits paired steering datasets
(`steering_data_search_20.json`, `steering_data_code_20.json`, `steering_data_askuser_20.json`).

### 2. Extract steering vectors

```bash
python steering_vector/build_step_mark_steering_vectors.py \
  --model-path /path/to/Llama-3.1-8B-Instruct \
  --output-dir steering_vector/Llama_3_8_vector_heading
```

Produces per-tool payloads such as `step_mark_all.pt`, `step_mark_search.pt`,
`step_mark_code.pt`, `step_mark_askuser.pt`, each containing per-layer steering vectors.

### 3. Run steered inference

```bash
python inference/inference_tool_prompt_tag_suppressed_kvcache.py \
  --model_name_or_path /path/to/Llama-3.1-8B-Instruct \
  --data_path data_inference/domain_math_tool_prompt.json \
  --steering_vector_dir steering_vector/Llama_3_8_vector_heading \
  --domain math \
  --steering_layers 21 \
  --steering_strength 1.0 \
  --suppress_scale 1.0 \
  --save_path inference_results/llama8b/domain_math_layer21.json
```

Key flags: `--disable-steering` (no-steer baseline), `--max_steering_layers` / `--steering_layers`
(which layers to steer), `--schema {markdown,json}`, `--ablation {none,cross_format,heading_rename,matched_prompt}`,
`--max_new_tokens`, `--max_steps`, `--device`. Run with `-h` for the full list. The
`cosine_gated`, `cosine_deviation_scaled`, `orthogonalized`, `qr_subspace_projected`, and
`strict_instruction` variants share the same interface with their respective suppression rules.

### 4. Evaluate

```bash
python evaluate/inference_eval_math.py --input inference_results/llama8b/domain_math_layer21.json
```

The `inference_eval_{math,time,intention}.py` scripts use an LLM judge to score tool usage and
correctness on saved result files. `evaluate/plot_*.py` and
`evaluate/figures/similarity_figure/*.py` regenerate the paper figures from evaluated outputs.

### Ablations

```bash
bash scripts/run_cross_format_ablation.sh          # cross-format (markdown vs. json action)
python experiments/run_ablation.py --ablation heading_rename ...
python experiments/run_matched_prompt_baseline.py ...
```

## Notes

- Steering is applied at inference time via the KV cache; the base model weights are never
  modified.
- Steering strength, target layers, and the suppression rule are the main knobs governing the
  trade-off between reduced tool overuse and preserved task accuracy.
