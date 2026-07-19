#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-/home/qsy/anaconda3/envs/onnx/bin/python}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-${SCRIPT_DIR}/export_onnx_qwen3.py}"

MODEL_DIR="${MODEL_DIR:-/home/qsy/huggingface/model/llama3.2-1B}"
OUT_BASE="${OUT_BASE:-/home/qsy/huggingface/model/onnx_models}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-128}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-cuda}"
OPSET="${OPSET:-17}"
LOGITS_TO_KEEP="${LOGITS_TO_KEEP:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

if [[ "${LOGITS_TO_KEEP}" == "1" ]]; then
  DEFAULT_DIR_SUFFIX="-last-token"
  DEFAULT_FILE_SUFFIX="_last_token"
elif [[ "${LOGITS_TO_KEEP}" == "0" ]]; then
  DEFAULT_DIR_SUFFIX=""
  DEFAULT_FILE_SUFFIX=""
else
  DEFAULT_DIR_SUFFIX="-logits${LOGITS_TO_KEEP}"
  DEFAULT_FILE_SUFFIX="_logits${LOGITS_TO_KEEP}"
fi

OUT_DIR="${OUT_DIR:-${OUT_BASE}/llama3.2-1B-seq${SEQ_LEN}${DEFAULT_DIR_SUFFIX}}"
OUT_ONNX="${OUT_ONNX:-${OUT_DIR}/llama3_2_1b_seq${SEQ_LEN}${DEFAULT_FILE_SUFFIX}.onnx}"
OUTPUT_DIR="$(dirname "${OUT_ONNX}")"

echo "Python: ${PYTHON}"
echo "Model: ${MODEL_DIR}"
echo "Output: ${OUT_ONNX}"
echo "Shape: batch=${BATCH_SIZE}, seq_len=${SEQ_LEN}"
echo "Export: dtype=${DTYPE}, device=${DEVICE}, opset=${OPSET}, logits_to_keep=${LOGITS_TO_KEEP}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
  echo "ERROR: Export script not found: ${EXPORT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: Model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_FLAG=(--trust-remote-code)
else
  TRUST_FLAG=(--no-trust-remote-code)
fi

echo "Checking Python environment..."
PYTHONNOUSERSITE=1 "${PYTHON}" - <<'PY'
import importlib.metadata as metadata

packages = ["numpy", "torch", "transformers", "accelerate", "onnx"]
for package in packages:
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"Required package is not installed: {package}")

import numpy as np
import torch
import transformers
import accelerate
import onnx

print("runtime numpy:", np.__version__, np.__file__)
print("runtime torch:", torch.__version__)
print("runtime transformers:", transformers.__version__)
print("runtime accelerate:", accelerate.__version__)
print("runtime onnx:", onnx.__version__)
PY

echo "Exporting ONNX..."
PYTHONNOUSERSITE=1 "${PYTHON}" "${EXPORT_SCRIPT}" \
  --model-dir "${MODEL_DIR}" \
  --output "${OUT_ONNX}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --opset "${OPSET}" \
  --logits-to-keep "${LOGITS_TO_KEEP}" \
  "${TRUST_FLAG[@]}" \
  --attn-implementation eager

echo "Checking exported files..."
if [[ ! -f "${OUT_ONNX}" ]]; then
  echo "ERROR: ONNX file was not generated: ${OUT_ONNX}" >&2
  exit 1
fi

find "${OUTPUT_DIR}" -maxdepth 1 -type f -printf "%p\t%s bytes\n" | sort
echo "Done."
