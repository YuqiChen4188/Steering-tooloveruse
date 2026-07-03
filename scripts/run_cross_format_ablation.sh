#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/yche1052/miniconda3/envs/nova/bin/python}
MODEL_PATH=${MODEL_PATH:-/data/yuqi/Steering-tooloveruse/model_weights/Llama-3.1-8B-Instruct}
DOMAIN=${DOMAIN:-math}
DATA_PATH=${DATA_PATH:-data_inference/domain_math_tool_prompt.json}
MARKDOWN_VECTOR_DIR=${MARKDOWN_VECTOR_DIR:-steering_vector/Llama_3_8_vector_heading}
JSON_VECTOR_DIR=${JSON_VECTOR_DIR:-steering_vector/Llama_3_8_vector_json_action}
LAYER=${LAYER:-21}
ALPHA=${ALPHA:-1.0}
MAX_TEST_NUM=${MAX_TEST_NUM:-400}
CSV_PATH=${CSV_PATH:-results/ablations/cross_format_results.csv}

"${PYTHON_BIN}" experiments/run_ablation.py \
  --ablation cross_format \
  --model-name-or-path "${MODEL_PATH}" \
  --domain "${DOMAIN}" \
  --data-path "${DATA_PATH}" \
  --markdown-vector-dir "${MARKDOWN_VECTOR_DIR}" \
  --json-vector-dir "${JSON_VECTOR_DIR}" \
  --layer "${LAYER}" \
  --alpha "${ALPHA}" \
  --max-test-num "${MAX_TEST_NUM}" \
  --csv-path "${CSV_PATH}" \
  "$@"
