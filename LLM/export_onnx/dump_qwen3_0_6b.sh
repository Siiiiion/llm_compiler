#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-/home/qsy/anaconda3/envs/onnx/bin/python}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-${SCRIPT_DIR}/export_onnx_qwen3.py}"

MODEL_DIR="${MODEL_DIR:-/home/qsy/huggingface/model/Qwen3-0.6B}"
OUT_BASE="${OUT_BASE:-/home/qsy/huggingface/model/onnx_models}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-128}"
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-cuda}"
OPSET="${OPSET:-17}"
LOGITS_TO_KEEP="${LOGITS_TO_KEEP:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
BATCH_SIZES="${BATCH_SIZES:-}"
SEQ_LENS="${SEQ_LENS:-}"
LOGITS_MODES="${LOGITS_MODES:-}"
MODEL_TAG="${MODEL_TAG:-qwen3_0_6b}"
OUTPUT_ROOT_INPUT="${OUTPUT_ROOT:-}"
MANIFEST_INPUT="${MANIFEST:-}"

GRID_MODE=0
if [[ -n "${BATCH_SIZES}" || -n "${SEQ_LENS}" || -n "${LOGITS_MODES}" || \
      -n "${OUTPUT_ROOT_INPUT}" || -n "${MANIFEST_INPUT}" ]]; then
  GRID_MODE=1
fi

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

OUT_DIR="${OUT_DIR:-${OUT_BASE}/Qwen3-0.6B-seq${SEQ_LEN}${DEFAULT_DIR_SUFFIX}}"
OUT_ONNX="${OUT_ONNX:-${OUT_DIR}/qwen3_0_6b_seq${SEQ_LEN}${DEFAULT_FILE_SUFFIX}.onnx}"
OUTPUT_DIR="$(dirname "${OUT_ONNX}")"
OUTPUT_ROOT="${OUTPUT_ROOT_INPUT:-${OUT_BASE}/Qwen3-0.6B-grid-${DTYPE}}"
MANIFEST="${MANIFEST_INPUT:-${OUTPUT_ROOT}/manifest.json}"

echo "Python: ${PYTHON}"
echo "Model: ${MODEL_DIR}"
if [[ "${GRID_MODE}" == "1" ]]; then
  echo "Output root: ${OUTPUT_ROOT}"
  echo "Manifest: ${MANIFEST}"
  printf 'Grid: batch_sizes=%s, seq_lens=%s, logits_modes=%s\n' \
    "${BATCH_SIZES:-${BATCH_SIZE}}" \
    "${SEQ_LENS:-${SEQ_LEN}}" \
    "${LOGITS_MODES:-${LOGITS_TO_KEEP}}"
else
  echo "Output: ${OUT_ONNX}"
  echo "Shape: batch=${BATCH_SIZE}, seq_len=${SEQ_LEN}"
fi
echo "Export: dtype=${DTYPE}, device=${DEVICE}, opset=${OPSET}"

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

if [[ "${GRID_MODE}" == "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
else
  mkdir -p "${OUTPUT_DIR}"
fi

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

import accelerate
import numpy as np
import onnx
import torch
import transformers

print("runtime numpy:", np.__version__, np.__file__)
print("runtime torch:", torch.__version__)
print("runtime transformers:", transformers.__version__)
print("runtime accelerate:", accelerate.__version__)
print("runtime onnx:", onnx.__version__)
PY

echo "Exporting ONNX..."
EXPORT_ARGS=(
  --model-dir "${MODEL_DIR}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --opset "${OPSET}"
  "${TRUST_FLAG[@]}"
  --attn-implementation eager
)

if [[ "${GRID_MODE}" == "1" ]]; then
  EXPORT_ARGS+=(
    --output-root "${OUTPUT_ROOT}"
    --manifest "${MANIFEST}"
    --model-tag "${MODEL_TAG}"
    --batch-sizes "${BATCH_SIZES:-${BATCH_SIZE}}"
    --seq-lens "${SEQ_LENS:-${SEQ_LEN}}"
    --logits-modes "${LOGITS_MODES:-${LOGITS_TO_KEEP}}"
  )
else
  EXPORT_ARGS+=(
    --output "${OUT_ONNX}"
    --batch-size "${BATCH_SIZE}"
    --seq-len "${SEQ_LEN}"
    --logits-to-keep "${LOGITS_TO_KEEP}"
  )
fi

PYTHONNOUSERSITE=1 "${PYTHON}" "${EXPORT_SCRIPT}" "${EXPORT_ARGS[@]}"

echo "Checking exported files..."
if [[ "${GRID_MODE}" == "1" ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: Manifest was not generated: ${MANIFEST}" >&2
    exit 1
  fi
  find "${OUTPUT_ROOT}" -maxdepth 2 -type f -printf "%p\t%s bytes\n" | sort
else
  if [[ ! -f "${OUT_ONNX}" ]]; then
    echo "ERROR: ONNX file was not generated: ${OUT_ONNX}" >&2
    exit 1
  fi
  find "${OUTPUT_DIR}" -maxdepth 1 -type f -printf "%p\t%s bytes\n" | sort
fi
echo "Done."
