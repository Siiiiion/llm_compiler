# Fixed-shape Causal-LM ONNX Export

This directory exports a local HuggingFace causal language model as a fixed-shape ONNX
graph for the TVM/TLM workload generation and whole-model benchmark workflow.

Run commands from:

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM/export_onnx
```

## Export Options

The shell wrappers accept environment variables, so the model and output location can be
changed without editing the scripts.

| Variable | Meaning | Qwen default | Llama default |
| --- | --- | --- | --- |
| `PYTHON` | Python used for export | `/home/qsy/anaconda3/envs/onnx/bin/python` | same |
| `MODEL_DIR` | Local HuggingFace model directory | `/home/qsy/huggingface/model/Qwen3-0.6B` | `/home/qsy/huggingface/model/llama3.2-1B` |
| `OUT_BASE` | Base directory for generated model directories | `/home/qsy/huggingface/model/onnx_models` | same |
| `OUT_DIR` | Override the generated model directory | derived automatically | derived automatically |
| `OUT_ONNX` | Override the final ONNX path directly | derived automatically | derived automatically |
| `BATCH_SIZE` | Fixed batch size | `1` | `1` |
| `SEQ_LEN` | Fixed input sequence length | `128` | `128` |
| `DTYPE` | `float32`, `float16`, or `bfloat16` | `float32` | `float16` |
| `DEVICE` | Export device: `cpu` or `cuda` | `cuda` | `cuda` |
| `OPSET` | ONNX opset | `17` | `17` |
| `LOGITS_TO_KEEP` | `0`: all token logits; `1`: final-token logits | `0` | `0` |
| `TRUST_REMOTE_CODE` | `1` or `0` | `1` | `0` |
| `BATCH_SIZES` | Comma-separated Qwen grid, such as `1,2,4` | unset | not used by wrapper |
| `SEQ_LENS` | Comma-separated Qwen grid, such as `64,128,256` | unset | not used by wrapper |
| `LOGITS_MODES` | Comma-separated Qwen `logits_to_keep` grid | unset | not used by wrapper |
| `OUTPUT_ROOT` | Root directory for grid outputs | derived automatically | not used by wrapper |
| `MANIFEST` | Grid manifest path | `<OUTPUT_ROOT>/manifest.json` | not used by wrapper |

`OUT_ONNX` has the highest priority. Its parent directory is created automatically.

## Qwen3-0.6B

Export the current FP32 seq128 full-logits graph:

```bash
./dump_qwen3_0_6b.sh
```

Export seq256 to the default model directory:

```bash
SEQ_LEN=256 ./dump_qwen3_0_6b.sh
```

Export serving-style prefill logits for only the final prompt token:

```bash
SEQ_LEN=128 \
LOGITS_TO_KEEP=1 \
./dump_qwen3_0_6b.sh
```

The default output is then:

```text
/home/qsy/huggingface/model/onnx_models/Qwen3-0.6B-seq128-last-token/qwen3_0_6b_seq128_last_token.onnx
```

Export another Qwen-compatible model to an exact requested path:

```bash
MODEL_DIR=/home/qsy/huggingface/model/my-qwen-model \
OUT_ONNX=/home/qsy/huggingface/model/onnx_models/my-qwen-fp16-seq512/model.onnx \
BATCH_SIZE=1 \
SEQ_LEN=512 \
DTYPE=float16 \
DEVICE=cuda \
LOGITS_TO_KEEP=1 \
TRUST_REMOTE_CODE=1 \
./dump_qwen3_0_6b.sh
```

Export a complete fixed-shape grid. The model is loaded once, every combination receives a
separate ONNX directory, and the exporter writes one manifest:

```bash
BATCH_SIZES=1,2,4 \
SEQ_LENS=64,128,256 \
LOGITS_MODES=0,1 \
DTYPE=float32 \
OUTPUT_ROOT=/home/qsy/huggingface/model/onnx_models/Qwen3-0.6B-grid-fp32 \
./dump_qwen3_0_6b.sh
```

This command exports `3 x 3 x 2 = 18` fixed-shape graphs and writes:

```text
/home/qsy/huggingface/model/onnx_models/Qwen3-0.6B-grid-fp32/manifest.json
```

For a smaller sequence-only sweep, omit the other grid variables:

```bash
BATCH_SIZE=1 \
SEQ_LENS=64,128,256,512 \
LOGITS_TO_KEEP=1 \
./dump_qwen3_0_6b.sh
```

## Llama-3.2-1B

Export the existing FP16 seq128 full-logits graph:

```bash
./dump_llama3_2_1b.sh
```

Export an FP32 last-token graph to an exact location:

```bash
MODEL_DIR=/home/qsy/huggingface/model/llama3.2-1B \
OUT_ONNX=/home/qsy/huggingface/model/onnx_models/llama3.2-1B-seq128-fp32-last-token/llama3_2_1b_seq128_fp32_last_token.onnx \
SEQ_LEN=128 \
DTYPE=float32 \
DEVICE=cpu \
LOGITS_TO_KEEP=1 \
./dump_llama3_2_1b.sh
```

FP32 export requires roughly twice the weight storage of FP16 and normally needs more host
memory and export time.

## Direct Python Command

The underlying exporter is model-independent for HuggingFace causal LMs that accept
`input_ids`, `attention_mask`, `use_cache`, and `logits_to_keep`:

```bash
PYTHONNOUSERSITE=1 \
/home/qsy/anaconda3/envs/onnx/bin/python export_onnx_qwen3.py \
  --model-dir=/path/to/huggingface-model \
  --output=/path/to/output/model.onnx \
  --batch-size=1 \
  --seq-len=128 \
  --dtype=float32 \
  --device=cuda \
  --opset=17 \
  --logits-to-keep=1 \
  --trust-remote-code \
  --attn-implementation=eager
```

Use `--no-trust-remote-code` for models that use the installed Transformers implementation.

The equivalent direct grid interface is:

```bash
PYTHONNOUSERSITE=1 \
/home/qsy/anaconda3/envs/onnx/bin/python export_onnx_qwen3.py \
  --model-dir=/home/qsy/huggingface/model/Qwen3-0.6B \
  --output-root=/home/qsy/huggingface/model/onnx_models/Qwen3-0.6B-grid-fp32 \
  --manifest=/home/qsy/huggingface/model/onnx_models/Qwen3-0.6B-grid-fp32/manifest.json \
  --model-tag=qwen3_0_6b \
  --batch-sizes=1,2,4 \
  --seq-lens=64,128,256 \
  --logits-modes=0,1 \
  --dtype=float32 \
  --device=cuda \
  --opset=17
```

## Verify the ONNX

The ONNX graph may store large initializers as external files beside the `.onnx` file. Do not
move only the `.onnx` file; keep the entire output directory together.

```bash
ONNX_MODEL=/path/to/output/model.onnx
ONNX_PY=/home/qsy/anaconda3/envs/onnx/bin/python

PYTHONNOUSERSITE=1 "$ONNX_PY" -c "import onnx; from onnx import TensorProto; m=onnx.load('$ONNX_MODEL',load_external_data=False); print([(x.name,[d.dim_value for d in x.type.tensor_type.shape.dim],TensorProto.DataType.Name(x.type.tensor_type.elem_type)) for x in m.graph.output])"
```

Expected last-token output examples:

```text
Qwen3-0.6B:    logits [1, 1, 151936]
Llama-3.2-1B: logits [1, 1, 128256]
```

## Add Workloads to the RTX 4090 Registry

After exporting each fixed shape, dump it directly into the shared 4090 registry. The script
adds the model-specific task pickle and rebuilds `all_tasks.pkl` from every `*.task.pkl` in
the directory.

For a grid export, import the complete manifest in one process. Relay/task artifacts are
written for every graph, while `all_tasks.pkl` is rebuilt only once at the end:

```bash
cd /home/qsy/workspace/complier/llm_compiler

MANIFEST=/home/qsy/huggingface/model/onnx_models/Qwen3-0.6B-grid-fp32/manifest.json
REG=/home/qsy/workspace/dataset/network_info/4090
ONNX_PY=/home/qsy/anaconda3/envs/onnx/bin/python
TVM_ONNX_PYTHONPATH=/data3/qsy/complier/tlm/python:/home/qsy/anaconda3/envs/onnx/lib/python3.10/site-packages:/home/qsy/anaconda3/envs/tlm/lib/python3.10/site-packages/decorator-5.2.1-py3.10.egg:/home/qsy/anaconda3/envs/tlm/lib/python3.10/site-packages/cloudpickle-3.1.1-py3.10.egg

PYTHONNOUSERSITE=1 \
PYTHONPATH="$TVM_ONNX_PYTHONPATH" \
"$ONNX_PY" gen/dump_network_info.py \
  --target=nvidia/geforce-rtx-4090 \
  --hardware-name=4090 \
  --network-info-dir="$REG" \
  --only-workloads \
  --skip-relay-artifact \
  --onnx-manifest="$MANIFEST"
```

Manifest import automatically repairs the known Transformers attention-mask pattern where a
FP16/FP32 `Where` zero branch is paired with the same dtype's minimum value exported as FP64.
For a single ONNX import, the equivalent explicit option is `--fix-mask-where-dtype`; the old
`--fix-fp16-mask-where` spelling remains available as a compatibility alias.

Task extraction processes full model parameters and is CPU/memory intensive. On this machine,
the first Qwen3-0.6B FP32 seq64 graph took about five minutes, so a large grid should be run in
a persistent terminal session.

`--skip-relay-artifact` is recommended for a shared workload registry: a single Qwen3-0.6B
FP32 Relay artifact is about 3 GB because it embeds parameter bytes. The task pickle is only
about 200 KB. Omit this option only for selected shapes that must later be loaded directly by
`benchmark_relay.py`.

The manifest flow is resumable. With `--skip-relay-artifact`, an existing task pickle is
detected before loading the corresponding ONNX, so rerunning the same command skips completed
shapes. Use `--overwrite-task-info` only when those completed task files must be regenerated.

The original single-ONNX interface remains available:

```bash
cd /home/qsy/workspace/complier/llm_compiler

ONNX_MODEL=/path/to/output/model.onnx
MODEL_NAME=qwen3_0_6b_seq128
SEQ_LEN=128
REG=/home/qsy/workspace/dataset/network_info/4090
ONNX_PY=/home/qsy/anaconda3/envs/onnx/bin/python
TVM_ONNX_PYTHONPATH=/data3/qsy/complier/tlm/python:/home/qsy/anaconda3/envs/onnx/lib/python3.10/site-packages:/home/qsy/anaconda3/envs/tlm/lib/python3.10/site-packages/decorator-5.2.1-py3.10.egg:/home/qsy/anaconda3/envs/tlm/lib/python3.10/site-packages/cloudpickle-3.1.1-py3.10.egg

PYTHONNOUSERSITE=1 \
PYTHONPATH="$TVM_ONNX_PYTHONPATH" \
"$ONNX_PY" gen/dump_network_info.py \
  --target=nvidia/geforce-rtx-4090 \
  --hardware-name=4090 \
  --network-info-dir="$REG" \
  --only-workloads \
  --onnx-model="$ONNX_MODEL" \
  --model-name="$MODEL_NAME" \
  --input-shapes="{\"input_ids\":[1,${SEQ_LEN}],\"attention_mask\":[1,${SEQ_LEN}]}" \
  --input-dtypes='{"input_ids":"int64","attention_mask":"int64"}'
```

Add `--overwrite-relay --overwrite-task-info` only when intentionally replacing an existing
artifact with the same model name and input shape.

## Experiment Rules

- Export a separate fixed-shape ONNX for every batch/sequence combination. Do not relabel a
  seq128 graph as seq256 by changing only `--input-shapes` during TVM import.
- Use distinct output directories for full logits and last-token logits so their external
  weight files and compiled libraries cannot be mixed.
- FP16 and FP32 produce different workload keys. Build separate task registries, generated
  candidates, measured histories, and compiled libraries for each dtype.
- For fair model comparisons, keep dtype, batch size, sequence length, `LOGITS_TO_KEEP`, TVM
  target, candidate count, and benchmark parameters identical.
