# Attention Schedule vs FlashAttention

This directory evaluates the actual capability boundary of the current framework:

1. build one fixed-shape causal GQA Relay graph;
2. extract the graph into the QK, softmax, and PV AutoScheduler tasks supported by TVM 0.12;
3. give the exact same initial sketches and candidate budget to the Pretrain and LAPO models;
4. measure all generated schedules and compile the complete Attention graph with each history;
5. benchmark official FlashAttention on the exact same Q/K/V arrays;
6. compare end-to-end latency and output correctness.

The TVM side is a complete multi-kernel Attention implementation. It is not an online-softmax
FlashAttention algorithm and task latencies are never added manually.

## 1. Prepare One Experiment

Start with Qwen3-0.6B's shape at sequence length 128:

```bash
cd /home/qsy/workspace/complier/llm_compiler/LLM

EXP=/home/qsy/workspace/gen_data/attention_b1_s128_fp16
TLM_PY=/home/qsy/anaconda3/envs/tlm/bin/python

"$TLM_PY" attention/create_attention_task.py \
  --output-dir="$EXP" \
  --target=nvidia/geforce-rtx-4090 \
  --batch=1 \
  --seq-len=128 \
  --num-heads=16 \
  --num-kv-heads=8 \
  --head-dim=128 \
  --dtype=float16 \
  --sketches-per-task=48
```

This writes `metadata.json`, `attention.relay.pkl`, `attention_sketch.json`, `inputs.npz`,
and a private `network_info/` registry. Do not reuse the existing FP32 Qwen registry.
The saved sketch records normalize `SP` factors and end at the first `PPT`, matching the
prompt format used by `make_dataset.py`; no completed schedule suffix is sent to the model.

## 2. Generate and Measure Equal-Budget Schedules

Run one GPU exclusively. Both variants use the same sketch file, seed, sampling settings,
and `keep_cnt`:

```bash
CUDA_VISIBLE_DEVICES=0 "$TLM_PY" attention/run_generation.py \
  --experiment-dir="$EXP" \
  --pretrain-model=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2 \
  --lapo-model=/home/qsy/huggingface/model/Qwen3-0.6B-4090-lapo-v1 \
  --cuda-visible=0 \
  --keep-cnt=64 \
  --generation-seed=42 \
  --stage=all
```

Use `--stage=generate` and `--stage=measure` to split a long run. Add `--dry-run` to print
the exact subprocess commands without executing them.
The Pretrain checkpoint must match `model_name_or_path` in the LAPO checkpoint's
`dpo_manifest.json`; using another Stage-1 checkpoint is not a clean LAPO ablation.

## 3. Benchmark the Complete TVM Attention Graph

```bash
CUDA_VISIBLE_DEVICES=0 "$TLM_PY" attention/benchmark_tvm_attention.py \
  --experiment-dir="$EXP" \
  --variant=pretrain="$EXP/pretrain_measured.json" \
  --variant=lapo="$EXP/lapo_measured.json" \
  --warmup=20 \
  --repeat=50 \
  --min-repeat-ms=500
```

The benchmark refuses incomplete histories. Every extracted task must have at least one valid
record before a Pretrain/LAPO number is accepted. It rebuilds each library by default so a
new measurement history cannot silently reuse a stale `.so`; pass `--reuse-libraries` only
when the histories and TVM build are unchanged. The current TVM Default schedule is omitted
because its fused QK/scale/mask CUDA schedule fails `VerifyMemory`; it is not used as a result.

## 4. Benchmark FlashAttention

Use a Python environment containing a CUDA build of PyTorch and `flash-attn`. It can be
different from the TVM environment because the two scripts exchange NPZ/JSON files:

```bash
CUDA_VISIBLE_DEVICES=0 /home/qsy/anaconda3/envs/nano/bin/python \
  attention/benchmark_flash_attention.py \
  --experiment-dir="$EXP" \
  --device=cuda:0 \
  --warmup=20 \
  --number=1 \
  --repeat=50 \
  --min-repeat-ms=500
```

The script benchmarks manual eager causal GQA as a correctness reference and
`flash_attn_func` as the comparison target. FlashAttention requires FP16 or BF16; this entry
currently prepares FP16 because it is supported by both this TVM version and FlashAttention.
Keep `min_repeat_ms` aligned with the TVM benchmark so both sides use sustained GPU execution
instead of comparing TVM steady-state timing with one synchronized short kernel.

## 5. Produce the Final Table

The comparison is environment-independent and can run with a normal NumPy Python:

```bash
python attention/compare_results.py \
  --tvm-report="$EXP/tvm_report.json" \
  --flash-report="$EXP/flash_report.json" \
  --output="$EXP/comparison.json"
```

Interpretation:

- `lapo_speedup_vs_pretrain > 1`: LAPO improved the generated Attention implementation.
- `lapo_speedup_vs_flash > 1`: LAPO's complete TVM Attention graph was faster than
  FlashAttention for this fixed shape.
- A speed result is invalid when its correctness entry is false.

After sequence length 128 works end to end, repeat preparation in separate directories for
`256, 512, 1024, 2048, 4096`. Never mix tasks or measured histories across shapes/dtypes.
