#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""torchrun 包装脚本：创新点 2 的 DPO / LAPO / SFT 训练入口。

与 ``run_train_qwen3_clm.py`` 并列。通过环境变量切换训练模式：

关键环境变量：
    USE_DPO        ——  默认 true。 置为 false / 0 则退化为 SFT 基线（仅在 chosen 上 CLM）；
                       等价于 ``TRAINING_MODE=sft``。
    TRAINING_MODE  ——  显式优先级更高。可选：sft / dpo / lapo。
                       设置后会覆盖 USE_DPO 的推断。
    LAPO           ——  便捷开关；true 时强制 TRAINING_MODE=lapo。
    BETA           ——  DPO 温度系数（默认 0.1）。
    PREFERENCE_DATASET ——  必填。build_preference_pairs.py 的输出目录。
    POLICY_MODEL_PATH / REF_MODEL_PATH / TOKENIZER_NAME ——  路径覆盖。
    OUTPUT_DIR ——  输出目录；未设置时按训练模式自动命名。

示例：
    # 跑真正的 DPO（默认）
    PREFERENCE_DATASET=~/workspace/gen_data/4090_prefs \
    POLICY_MODEL_PATH=~/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
    python3 run_train_qwen3_dpo.py

    # 只跑 SFT 基线（不训 DPO，用于消融）
    USE_DPO=false python3 run_train_qwen3_dpo.py

    # 跑 LAPO（Latency-Weighted DPO）
    LAPO=true python3 run_train_qwen3_dpo.py
"""

from __future__ import annotations

import os
import shlex
import subprocess


home_dir = os.path.expanduser("~")
workspace_dir = os.path.join(home_dir, "workspace")
model_root_dir = os.path.join(home_dir, "huggingface", "model")
script_basename = os.path.basename(os.path.abspath(__file__))
log_file = os.environ.get("LOG_FILE", f"{script_basename}.log")
session_name = os.environ.get("SESSION_NAME", script_basename.replace(".", "_"))
torchrun_bin = os.environ.get(
    "TORCHRUN_BIN",
    os.path.join(home_dir, "anaconda3", "envs", "tlm", "bin", "torchrun"),
)
master_port = int(os.environ.get("MASTER_PORT", "29541"))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_cuda_visible_devices(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def _is_valid_local_model_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _is_valid_local_tokenizer_dir(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.isfile(os.path.join(path, "tokenizer_config.json"))
        or os.path.isfile(os.path.join(path, "tokenizer.json"))
    )


def _looks_like_local_path(path: str) -> bool:
    if not path:
        return False
    return path.startswith("/") or path.startswith("./") or path.startswith("../") or path.startswith("~")


def _append_optional_arg(parts, name, value):
    if value not in (None, ""):
        parts.append(f"{name}={shlex.quote(str(value))}")


cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
visible_devices = _parse_cuda_visible_devices(cuda_visible_devices)
if not visible_devices:
    raise ValueError("CUDA_VISIBLE_DEVICES is empty. Please set at least one GPU id.")

nproc_per_node = int(os.environ.get("NPROC_PER_NODE", str(len(visible_devices))))
if nproc_per_node <= 0:
    raise ValueError(f"NPROC_PER_NODE must be > 0, got {nproc_per_node}")
if nproc_per_node > len(visible_devices):
    print(
        f"[run_train_qwen3_dpo] NPROC_PER_NODE={nproc_per_node} exceeds visible GPUs="
        f"{len(visible_devices)}; fallback to {len(visible_devices)}"
    )
    nproc_per_node = len(visible_devices)

smoke_test = _env_flag("SMOKE_TEST", False)
if smoke_test:
    nproc_per_node = 1
    cuda_visible_devices = visible_devices[0]


# ---- Resolve training_mode ------------------------------------------------
# Precedence: TRAINING_MODE > LAPO > USE_DPO
TRAINING_MODES = {"sft", "dpo", "lapo"}
training_mode = os.environ.get("TRAINING_MODE", "").strip().lower()
if training_mode and training_mode not in TRAINING_MODES:
    raise ValueError(f"TRAINING_MODE must be one of {TRAINING_MODES}, got {training_mode!r}")

if not training_mode:
    if _env_flag("LAPO", False):
        training_mode = "lapo"
    elif _env_flag("USE_DPO", True):
        training_mode = "dpo"
    else:
        training_mode = "sft"

print(f"[run_train_qwen3_dpo] training_mode = {training_mode}")


# ---- Resolve paths --------------------------------------------------------
base_model_dir = os.path.join(model_root_dir, "Qwen3-0.6B")
struct_stage1_dir = os.path.join(model_root_dir, "Qwen3-0.6B-4090-struct-stage1")
struct_stage2_dir = os.path.join(model_root_dir, "Qwen3-0.6B-4090-struct-stage2")
ext_stage1_dir = os.path.join(model_root_dir, "Qwen3-0.6B-tvm-ext-4090-struct-stage1")

policy_candidates = [struct_stage2_dir, struct_stage1_dir, ext_stage1_dir, base_model_dir]

policy_model_path = os.environ.get("POLICY_MODEL_PATH") or os.environ.get("MODEL_NAME_OR_PATH")
if not policy_model_path:
    for p in policy_candidates:
        if _is_valid_local_model_dir(p):
            policy_model_path = p
            break
if not policy_model_path:
    raise FileNotFoundError(
        "POLICY_MODEL_PATH not set and no valid fallback found in "
        f"{policy_candidates}. Please provide POLICY_MODEL_PATH."
    )
if _looks_like_local_path(policy_model_path):
    policy_model_path = os.path.expanduser(policy_model_path)
    if not _is_valid_local_model_dir(policy_model_path):
        raise FileNotFoundError(
            f"Invalid POLICY_MODEL_PATH: {policy_model_path}. "
            "Expected a local directory containing config.json."
        )

ref_model_path = os.environ.get("REF_MODEL_PATH", policy_model_path)
if _looks_like_local_path(ref_model_path):
    ref_model_path = os.path.expanduser(ref_model_path)
    if not _is_valid_local_model_dir(ref_model_path):
        raise FileNotFoundError(
            f"Invalid REF_MODEL_PATH: {ref_model_path}. "
            "Expected a local directory containing config.json."
        )

tokenizer_name = os.environ.get("TOKENIZER_NAME", policy_model_path)
if _looks_like_local_path(tokenizer_name):
    tokenizer_name = os.path.expanduser(tokenizer_name)
    if not _is_valid_local_tokenizer_dir(tokenizer_name):
        for fallback in (policy_model_path, base_model_dir):
            if _is_valid_local_tokenizer_dir(fallback):
                print(
                    f"[run_train_qwen3_dpo] TOKENIZER_NAME={tokenizer_name} invalid; "
                    f"fallback to {fallback}"
                )
                tokenizer_name = fallback
                break
        else:
            raise FileNotFoundError(
                f"Invalid TOKENIZER_NAME: {tokenizer_name}. No valid tokenizer dir found."
            )

preference_dataset = os.environ.get("PREFERENCE_DATASET")
if not preference_dataset:
    default_prefs = os.path.join(workspace_dir, "gen_data", "4090_prefs")
    if os.path.isdir(default_prefs):
        preference_dataset = default_prefs
    else:
        raise FileNotFoundError(
            "PREFERENCE_DATASET is required. Run build_preference_pairs.py first or "
            f"set PREFERENCE_DATASET. Default {default_prefs} not found."
        )
preference_dataset = os.path.expanduser(preference_dataset)
if not os.path.isfile(os.path.join(preference_dataset, "train_pairs.jsonl")):
    raise FileNotFoundError(
        f"{preference_dataset}/train_pairs.jsonl not found. "
        "Build preference pairs before launching DPO."
    )


output_dir_default = os.path.join(
    model_root_dir, f"Qwen3-0.6B-4090-{training_mode}"
)
output_dir = os.environ.get("OUTPUT_DIR", output_dir_default)
if smoke_test:
    output_dir = os.environ.get("OUTPUT_DIR", output_dir + "-smoke")


# ---- Training hyperparams -------------------------------------------------
per_device_train_batch_size = os.environ.get(
    "PER_DEVICE_TRAIN_BATCH_SIZE", "1" if smoke_test else "4"
)
per_device_eval_batch_size = os.environ.get(
    "PER_DEVICE_EVAL_BATCH_SIZE", "1" if smoke_test else "4"
)
gradient_accumulation_steps = os.environ.get("GRADIENT_ACCUMULATION_STEPS", "2")
learning_rate = os.environ.get("LEARNING_RATE", "5e-7")
num_train_epochs = os.environ.get("NUM_TRAIN_EPOCHS", "1")
logging_steps = os.environ.get("LOGGING_STEPS", "5" if smoke_test else "50")
eval_steps = os.environ.get("EVAL_STEPS", "20" if smoke_test else "500")
save_steps = os.environ.get("SAVE_STEPS", "40" if smoke_test else "1000")
save_total_limit = os.environ.get("SAVE_TOTAL_LIMIT", "1" if smoke_test else "3")
dataloader_num_workers = os.environ.get(
    "DATALOADER_NUM_WORKERS", "0" if smoke_test else "4"
)
dataloader_persistent_workers = _env_flag("DATALOADER_PERSISTENT_WORKERS", not smoke_test)
warmup_ratio = os.environ.get("WARMUP_RATIO", "0.03")
warmup_steps = os.environ.get("WARMUP_STEPS")
max_steps = os.environ.get("MAX_STEPS", "30" if smoke_test else None)
max_train_samples = os.environ.get("MAX_TRAIN_SAMPLES", "512" if smoke_test else None)
max_eval_samples = os.environ.get("MAX_EVAL_SAMPLES", "128" if smoke_test else None)
metric_for_best_model = os.environ.get("METRIC_FOR_BEST_MODEL", "eval_loss")
greater_is_better = _env_flag("GREATER_IS_BETTER", False)
load_best_model_at_end = _env_flag("LOAD_BEST_MODEL_AT_END", not smoke_test)
overwrite_output_dir = _env_flag("OVERWRITE_OUTPUT_DIR", smoke_test)
gradient_checkpointing = _env_flag("GRADIENT_CHECKPOINTING", False)
group_by_length = _env_flag("GROUP_BY_LENGTH", False)
remove_unused_columns = False  # DPO collator depends on extra columns

beta = os.environ.get("BETA", "0.1")
loss_type = os.environ.get("LOSS_TYPE", "sigmoid")
label_smoothing = os.environ.get("LABEL_SMOOTHING", "0.0")
lapo_weight_clip = os.environ.get("LAPO_WEIGHT_CLIP", "10.0")
lapo_normalize = _env_flag("LAPO_NORMALIZE", True)
max_prompt_length = os.environ.get("MAX_PROMPT_LENGTH", "512")
max_length = os.environ.get("MAX_LENGTH", "1024")
sample_seed = os.environ.get("SAMPLE_SEED", "42")


if os.path.exists(log_file) and _env_flag("DELETE_LOG_IF_EXISTS", False):
    os.remove(log_file)


cmd_parts = [
    shlex.quote(torchrun_bin),
    f"--nproc_per_node={nproc_per_node}",
    f"--master_port={master_port}",
    "train_qwen3_dpo.py",
    "--do_train",
]
if os.path.isfile(os.path.join(preference_dataset, "validation_pairs.jsonl")):
    cmd_parts.append("--do_eval")

cmd_parts.extend(
    [
        f"--model_name_or_path={shlex.quote(policy_model_path)}",
        f"--ref_model_name_or_path={shlex.quote(ref_model_path)}",
        f"--tokenizer_name={shlex.quote(tokenizer_name)}",
        f"--output_dir={shlex.quote(output_dir)}",
        f"--preference_dataset={shlex.quote(preference_dataset)}",
        f"--training_mode={training_mode}",
        f"--beta={beta}",
        f"--loss_type={loss_type}",
        f"--label_smoothing={label_smoothing}",
        f"--lapo_weight_clip={lapo_weight_clip}",
        f"--lapo_normalize={lapo_normalize}",
        f"--max_prompt_length={max_prompt_length}",
        f"--max_length={max_length}",
        f"--sample_seed={sample_seed}",
        f"--per_device_train_batch_size={per_device_train_batch_size}",
        f"--per_device_eval_batch_size={per_device_eval_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--logging_steps={logging_steps}",
        f"--num_train_epochs={num_train_epochs}",
        f"--learning_rate={learning_rate}",
        f"--eval_strategy=steps",
        f"--eval_steps={eval_steps}",
        f"--save_strategy=steps",
        f"--save_steps={save_steps}",
        f"--save_total_limit={save_total_limit}",
        f"--dataloader_num_workers={dataloader_num_workers}",
        f"--dataloader_persistent_workers={dataloader_persistent_workers}",
        "--bf16=True",
        f"--gradient_checkpointing={gradient_checkpointing}",
        f"--group_by_length={group_by_length}",
        f"--remove_unused_columns={remove_unused_columns}",
        "--report_to=none",
    ]
)

if nproc_per_node > 1:
    cmd_parts.append("--ddp_find_unused_parameters=False")
if warmup_steps:
    cmd_parts.append(f"--warmup_steps={warmup_steps}")
else:
    cmd_parts.append(f"--warmup_ratio={warmup_ratio}")
if load_best_model_at_end and os.path.isfile(
    os.path.join(preference_dataset, "validation_pairs.jsonl")
):
    cmd_parts.extend(
        [
            "--load_best_model_at_end=True",
            f"--metric_for_best_model={metric_for_best_model}",
            f"--greater_is_better={greater_is_better}",
        ]
    )
if overwrite_output_dir:
    cmd_parts.append("--overwrite_output_dir")

_append_optional_arg(cmd_parts, "--max_steps", max_steps)
_append_optional_arg(cmd_parts, "--max_train_samples", max_train_samples)
_append_optional_arg(cmd_parts, "--max_eval_samples", max_eval_samples)


train_cmd = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} " + " ".join(cmd_parts)
print("[run_train_qwen3_dpo] launching command:")
print(train_cmd)

cmd = """tmux new -s %s -d '{
{
set -x
echo "#################################################################"
date
echo "training_mode=%s policy=%s ref=%s prefs=%s"

export PYTHONUNBUFFERED=1
%s

date
} |& tee -a %s
}'
""" % (
    session_name,
    training_mode,
    policy_model_path,
    ref_model_path,
    preference_dataset,
    train_cmd,
    log_file,
)

subprocess.Popen(cmd, shell=True)
