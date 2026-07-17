import os
import shlex
import subprocess


home_dir = os.path.expanduser("~")
workspace_dir = os.path.join(home_dir, "workspace")
model_root_dir = os.path.join(home_dir, "huggingface", "model")
script_basename = os.path.basename(os.path.abspath(__file__))
log_file = os.environ.get("LOG_FILE", f"{script_basename}.log")
session_name = os.environ.get("SESSION_NAME", script_basename.replace(".", "_"))
torchrun_bin = os.environ.get("TORCHRUN_BIN", os.path.join(home_dir, "anaconda3", "envs", "tlm", "bin", "torchrun"))
master_port = int(os.environ.get("MASTER_PORT", "29531"))


def _parse_cuda_visible_devices(raw):
    return [x.strip() for x in raw.split(",") if x.strip()]


def _is_valid_local_model_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _is_valid_local_tokenizer_dir(path):
    return os.path.isdir(path) and (
        os.path.isfile(os.path.join(path, "tokenizer_config.json"))
        or os.path.isfile(os.path.join(path, "tokenizer.json"))
    )


def _looks_like_local_path(path):
    if not path:
        return False
    return path.startswith("/") or path.startswith("./") or path.startswith("../")


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


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
        f"[run_train_qwen3_clm] NPROC_PER_NODE={nproc_per_node} exceeds visible GPUs={len(visible_devices)}; "
        f"fallback to {len(visible_devices)}"
    )
    nproc_per_node = len(visible_devices)

smoke_test = _env_flag("SMOKE_TEST", False)
if smoke_test:
    nproc_per_node = 1
    cuda_visible_devices = visible_devices[0]

# stage1: generic warmup on for_gen dataset
# stage2: decision-focused training with PPT suffix mask (recommended)
train_stage = os.environ.get("TRAIN_STAGE", "stage1")

base_model_dir = os.path.join(model_root_dir, "Qwen3-0.6B")
legacy_finetuned_model_dir = os.path.join(model_root_dir, "Qwen3-0.6B-fintuned")
struct_stage1_dir = os.path.join(model_root_dir, "Qwen3-0.6B-4090-struct-stage1")
struct_stage2_dir = os.path.join(model_root_dir, "Qwen3-0.6B-4090-struct-stage2")
default_dataset_dir = os.path.join(workspace_dir, "gen_data", "4090_gen_qwen")

if train_stage == "stage1":
    model_name_or_path = os.environ.get("MODEL_NAME_OR_PATH", base_model_dir)
    tokenizer_name = os.environ.get("TOKENIZER_NAME", base_model_dir)
    dataset_name = os.environ.get("DATASET_NAME", default_dataset_dir)
    output_dir = os.environ.get("OUTPUT_DIR", struct_stage1_dir)
    learning_rate = os.environ.get("LEARNING_RATE", "1e-5")
    num_train_epochs = os.environ.get("NUM_TRAIN_EPOCHS", "1")
else:
    # stage2 default: continue from clean stage1 warmup.
    model_name_or_path = os.environ.get("MODEL_NAME_OR_PATH", struct_stage1_dir)
    tokenizer_name = os.environ.get("TOKENIZER_NAME", base_model_dir)
    dataset_name = os.environ.get("DATASET_NAME", default_dataset_dir)
    output_dir = os.environ.get("OUTPUT_DIR", struct_stage2_dir)
    learning_rate = os.environ.get("LEARNING_RATE", "5e-6")
    num_train_epochs = os.environ.get("NUM_TRAIN_EPOCHS", "1")

if smoke_test:
    output_dir = os.environ.get("OUTPUT_DIR", output_dir + "-smoke")

# Validate model path for local-dir usage and provide robust fallback for stage2.
if _looks_like_local_path(model_name_or_path) and not _is_valid_local_model_dir(model_name_or_path):
    if train_stage == "stage2":
        for fallback in (struct_stage1_dir, legacy_finetuned_model_dir, base_model_dir):
            if _is_valid_local_model_dir(fallback):
                print(
                    f"[run_train_qwen3_clm] MODEL_NAME_OR_PATH={model_name_or_path} is invalid; "
                    f"fallback to {fallback}"
                )
                model_name_or_path = fallback
                break
        else:
            raise FileNotFoundError(
                f"Invalid MODEL_NAME_OR_PATH: {model_name_or_path}. "
                f"No valid fallback found in {struct_stage1_dir}, {legacy_finetuned_model_dir} or {base_model_dir}."
            )
    else:
        raise FileNotFoundError(
            f"Invalid MODEL_NAME_OR_PATH: {model_name_or_path}. "
            "Expected a local directory containing config.json or a valid HF repo id."
        )

# Validate tokenizer path for local-dir usage and provide robust fallback.
if _looks_like_local_path(tokenizer_name) and not _is_valid_local_tokenizer_dir(tokenizer_name):
    tokenizer_fallbacks = [
        model_name_or_path,
        legacy_finetuned_model_dir,
        base_model_dir,
    ]
    chosen = None
    for fallback in tokenizer_fallbacks:
        if _is_valid_local_tokenizer_dir(fallback):
            chosen = fallback
            break

    if chosen is None:
        raise FileNotFoundError(
            f"Invalid TOKENIZER_NAME: {tokenizer_name}. "
            "No valid tokenizer directory found in model/tokenizer fallbacks."
        )

    print(
        f"[run_train_qwen3_clm] TOKENIZER_NAME={tokenizer_name} is invalid; "
        f"fallback to {chosen}"
    )
    tokenizer_name = chosen

if os.path.exists(log_file) and _env_flag("DELETE_LOG_IF_EXISTS", False):
    os.remove(log_file)

per_device_train_batch_size = os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "2" if smoke_test else "8")
per_device_eval_batch_size = os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2" if smoke_test else "8")
gradient_accumulation_steps = os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1")
logging_steps = os.environ.get("LOGGING_STEPS", "5" if smoke_test else "100")
eval_steps = os.environ.get("EVAL_STEPS", "10" if smoke_test else "2000")
save_steps = os.environ.get("SAVE_STEPS", "20" if smoke_test else "2000")
save_total_limit = os.environ.get("SAVE_TOTAL_LIMIT", "1" if smoke_test else "3")
dataloader_num_workers = os.environ.get("DATALOADER_NUM_WORKERS", "0" if smoke_test else "8")
dataloader_persistent_workers = _env_flag("DATALOADER_PERSISTENT_WORKERS", not smoke_test)
warmup_ratio = os.environ.get("WARMUP_RATIO", "0.03")
warmup_steps = os.environ.get("WARMUP_STEPS")
max_steps = os.environ.get("MAX_STEPS", "20" if smoke_test else None)
max_train_samples = os.environ.get("MAX_TRAIN_SAMPLES", "2048" if smoke_test else None)
max_eval_samples = os.environ.get("MAX_EVAL_SAMPLES", "256" if smoke_test else None)
metric_for_best_model = os.environ.get("METRIC_FOR_BEST_MODEL", "eval_accuracy")
greater_is_better = _env_flag("GREATER_IS_BETTER", True)
min_suffix_tokens = os.environ.get("MIN_SUFFIX_TOKENS", "16")
sample_seed = os.environ.get("SAMPLE_SEED", "42")
load_best_model_at_end = _env_flag("LOAD_BEST_MODEL_AT_END", not smoke_test)
overwrite_output_dir = _env_flag("OVERWRITE_OUTPUT_DIR", smoke_test)
# Large batch + dynamic padding on 0.6B bf16 makes gradient_checkpointing unnecessary.
# Keep it as an opt-in escape hatch for when you push batch size even higher.
gradient_checkpointing = _env_flag("GRADIENT_CHECKPOINTING", False)
group_by_length = _env_flag("GROUP_BY_LENGTH", False)
remove_unused_columns = _env_flag("REMOVE_UNUSED_COLUMNS", True)

cmd_parts = [
    shlex.quote(torchrun_bin),
    f"--nproc_per_node={nproc_per_node}",
    f"--master_port={master_port}",
    "train_qwen3_clm.py",
    "--do_train",
    "--do_eval",
    f"--model_name_or_path={shlex.quote(model_name_or_path)}",
    f"--tokenizer_name={shlex.quote(tokenizer_name)}",
    f"--output_dir={shlex.quote(output_dir)}",
    f"--dataset_name={shlex.quote(dataset_name)}",
    f"--per_device_train_batch_size={per_device_train_batch_size}",
    f"--per_device_eval_batch_size={per_device_eval_batch_size}",
    f"--gradient_accumulation_steps={gradient_accumulation_steps}",
    f"--logging_steps={logging_steps}",
    f"--num_train_epochs={num_train_epochs}",
    f"--remove_unused_columns={remove_unused_columns}",
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
    "--mask_prefix_before_ppt=True",
    "--drop_samples_without_ppt=True",
    "--subsample_before_mask=True",
    f"--min_suffix_tokens={min_suffix_tokens}",
    f"--sample_seed={sample_seed}",
    "--report_to=none",
]
if nproc_per_node > 1:
    cmd_parts.append("--ddp_find_unused_parameters=False")
if warmup_steps:
    cmd_parts.append(f"--warmup_steps={warmup_steps}")
else:
    cmd_parts.append(f"--warmup_ratio={warmup_ratio}")
if load_best_model_at_end:
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
cmd = """tmux new -s %s -d '{
{
set -x
echo "#################################################################"
date

export PYTHONUNBUFFERED=1
%s

date
} |& tee -a %s
}'
""" % (
    session_name,
    train_cmd,
    log_file,
)

subprocess.Popen(cmd, shell=True)