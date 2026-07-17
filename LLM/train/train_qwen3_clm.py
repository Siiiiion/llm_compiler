import logging
import math
import os
import sys
import inspect
from dataclasses import dataclass, field
from typing import List, Optional

import accelerate
import datasets
import numpy as np
import torch
from datasets import load_from_disk

# accelerate may access numpy._core on some versions; numpy 1.x only exposes numpy.core.
if not hasattr(np, "_core") and hasattr(np, "core"):
    np._core = np.core

if "keep_torch_compile" not in inspect.signature(accelerate.Accelerator.unwrap_model).parameters:
    _orig_unwrap_model = accelerate.Accelerator.unwrap_model

    def _unwrap_model_compat(self, model, keep_fp32_wrapper=True, keep_torch_compile=None):
        return _orig_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    accelerate.Accelerator.unwrap_model = _unwrap_model_compat

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    from transformers.utils import is_torch_tpu_available
except Exception:
    # Newer versions may move or remove TPU utility in non-TPU setups.
    def is_torch_tpu_available():
        return False

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "Base model path or HF repo id."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer path if different from model_name_or_path."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache dir for downloaded models."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Model revision(branch/tag/commit)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Use HF auth token if needed."},
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    low_cpu_mem_usage: bool = field(default=True)
    trust_remote_code: bool = field(default=True)
    early_stopping_patience: Optional[int] = field(default=None)


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="/home/qsy/workspace/gen_data/4090_gen_qwen",
        metadata={"help": "Path to tokenized dataset saved by datasets.save_to_disk."},
    )
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    sample_seed: int = field(
        default=42,
        metadata={"help": "Random seed used when subsampling train/eval splits."},
    )
    subsample_before_mask: bool = field(
        default=True,
        metadata={"help": "Subsample and shuffle split before PPT masking for faster smoke tests."},
    )
    mask_prefix_before_ppt: bool = field(
        default=True,
        metadata={"help": "Mask labels before (and including) PPT marker so loss focuses on decision suffix."},
    )
    ppt_marker: str = field(
        default="PPT",
        metadata={"help": "Marker token that separates prompt prefix and decision suffix."},
    )
    drop_samples_without_ppt: bool = field(
        default=True,
        metadata={"help": "Drop samples that do not contain PPT marker in input_ids."},
    )
    min_suffix_tokens: int = field(
        default=16,
        metadata={"help": "Minimum number of supervised (non -100) suffix tokens kept per sample."},
    )
    fix_mistral_regex: bool = field(
        default=True,
        metadata={"help": "Try enabling fix_mistral_regex when tokenizer supports it."},
    )


def _build_ppt_marker_id_candidates(tokenizer, marker: str) -> List[List[int]]:
    candidates = []
    for text in (marker, f" {marker}", f"{marker} "):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and ids not in candidates:
            candidates.append(ids)
    candidates.sort(key=len, reverse=True)
    return candidates


def _find_subseq_end(input_ids: List[int], patterns: List[List[int]]) -> Optional[int]:
    for i in range(len(input_ids)):
        for pat in patterns:
            end = i + len(pat)
            if end <= len(input_ids) and input_ids[i:end] == pat:
                return end - 1
    return None


def _load_tokenizer(model_args, data_args, tokenizer_path):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if data_args.fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    except TypeError:
        tokenizer_kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    return tokenizer


def _maybe_shuffle_and_select(dataset, split_name, max_samples, seed):
    if max_samples is None:
        return dataset

    before = len(dataset)
    target_size = min(before, max_samples)
    if target_size <= 0:
        return dataset.select([])

    if target_size < before:
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(target_size))

    logger.info(
        "Subsampled split=%s before_mask=%d after_subsample=%d sample_seed=%d",
        split_name,
        before,
        len(dataset),
        seed,
    )
    return dataset


def _apply_ppt_suffix_mask(dataset, split_name, marker_patterns, drop_without_ppt, min_suffix_tokens):
    before = len(dataset)
    has_precomputed = "ppt_end" in dataset.column_names
    if has_precomputed:
        logger.info(
            "PPT mask split=%s using precomputed 'ppt_end' column (fast path)",
            split_name,
        )
    else:
        logger.info(
            "PPT mask split=%s falling back to runtime subseq search (legacy dataset)",
            split_name,
        )

    def _mask_one(example):
        labels = list(example["labels"])
        input_ids = example["input_ids"]

        if has_precomputed:
            marker_end = int(example.get("ppt_end", -1))
            if marker_end < 0:
                marker_end = None
        else:
            marker_end = _find_subseq_end(input_ids, marker_patterns)

        if marker_end is None:
            if drop_without_ppt:
                example["keep_sample"] = 0
                example["suffix_tokens"] = 0
            else:
                labels = [-100] * len(labels)
                example["keep_sample"] = 1
                example["suffix_tokens"] = 0
        else:
            cutoff = min(marker_end + 1, len(labels))
            for idx in range(cutoff):
                labels[idx] = -100
            suffix_tokens = len(labels) - cutoff
            example["keep_sample"] = 1 if suffix_tokens >= min_suffix_tokens else 0
            example["suffix_tokens"] = suffix_tokens

        example["labels"] = labels
        return example

    dataset = dataset.map(
        _mask_one,
        desc=f"Applying PPT suffix mask on {split_name}",
        load_from_cache_file=True,
    )
    dataset = dataset.filter(
        lambda x: x["keep_sample"] == 1,
        desc=f"Filtering short/invalid PPT samples on {split_name}",
        load_from_cache_file=True,
    )
    after = len(dataset)

    for col in ("keep_sample", "suffix_tokens", "ppt_end"):
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    logger.info(
        "PPT mask split=%s before=%d after=%d dropped=%d drop_ratio=%.4f",
        split_name,
        before,
        after,
        before - after,
        0.0 if before == 0 else (before - after) / before,
    )
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Backward-compatible alias for older CLI usage.
        cli_args = []
        for arg in sys.argv[1:]:
            if arg == "--evaluation_strategy":
                cli_args.append("--eval_strategy")
            elif arg.startswith("--evaluation_strategy="):
                cli_args.append(arg.replace("--evaluation_strategy=", "--eval_strategy=", 1))
            else:
                cli_args.append(arg)
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=cli_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, fp16: {training_args.fp16}, bf16: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters: {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)

    tokenized_datasets = load_from_disk(data_args.dataset_name, keep_in_memory=False)

    tokenizer_path = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = _load_tokenizer(model_args, data_args, tokenizer_path)

    # Qwen tokenizer may not define a pad token; use eos token to keep batching stable.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    marker_patterns = None
    if data_args.mask_prefix_before_ppt:
        marker_patterns = _build_ppt_marker_id_candidates(tokenizer, data_args.ppt_marker)
        if not marker_patterns:
            raise ValueError(f"Cannot tokenize ppt_marker={data_args.ppt_marker!r} into non-empty ids")
        logger.info("Using PPT marker id patterns: %s", marker_patterns)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
    )

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if training_args.gradient_checkpointing:
        # Disable KV cache when gradient checkpointing is enabled to avoid extra memory usage.
        model.config.use_cache = False

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def _strip_unused_length_column(ds):
        # If group_by_length is on, Trainer reads training_args.length_column_name.
        if training_args.group_by_length:
            return ds
        if "length" in ds.column_names:
            ds = ds.remove_columns("length")
        return ds

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train split in dataset")
        train_dataset = tokenized_datasets["train"]
        required_columns = {"input_ids", "attention_mask", "labels"}
        missing_columns = required_columns - set(train_dataset.column_names)
        if missing_columns:
            raise ValueError(f"Train split is missing required columns: {sorted(missing_columns)}")
        if data_args.max_train_samples is not None and data_args.subsample_before_mask:
            train_dataset = _maybe_shuffle_and_select(
                train_dataset,
                split_name="train",
                max_samples=data_args.max_train_samples,
                seed=data_args.sample_seed,
            )
        if data_args.mask_prefix_before_ppt:
            train_dataset = _apply_ppt_suffix_mask(
                train_dataset,
                split_name="train",
                marker_patterns=marker_patterns,
                drop_without_ppt=data_args.drop_samples_without_ppt,
                min_suffix_tokens=data_args.min_suffix_tokens,
            )
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
        train_dataset = _strip_unused_length_column(train_dataset)
        logger.info(f"Loaded train split with {len(train_dataset)} samples")
    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation split in dataset")
        eval_dataset = tokenized_datasets["validation"]
        required_columns = {"input_ids", "attention_mask", "labels"}
        missing_columns = required_columns - set(eval_dataset.column_names)
        if missing_columns:
            raise ValueError(f"Validation split is missing required columns: {sorted(missing_columns)}")
        if data_args.max_eval_samples is not None and data_args.subsample_before_mask:
            eval_dataset = _maybe_shuffle_and_select(
                eval_dataset,
                split_name="validation",
                max_samples=data_args.max_eval_samples,
                seed=data_args.sample_seed + 1,
            )
        if data_args.mask_prefix_before_ppt:
            eval_dataset = _apply_ppt_suffix_mask(
                eval_dataset,
                split_name="validation",
                marker_patterns=marker_patterns,
                drop_without_ppt=data_args.drop_samples_without_ppt,
                min_suffix_tokens=data_args.min_suffix_tokens,
            )
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
        eval_dataset = _strip_unused_length_column(eval_dataset)
        logger.info(f"Loaded validation split with {len(eval_dataset)} samples")

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = np.asarray(labels)[:, 1:]
            preds = np.asarray(preds)[:, :-1]
            mask = labels != -100
            total = int(mask.sum())
            if total == 0:
                return {"accuracy": 0.0}
            correct = int(((preds == labels) & mask).sum())
            return {"accuracy": correct / total}

    else:
        eval_dataset = None
        compute_metrics = None
        preprocess_logits_for_metrics = None

    # Dynamic padding saves a lot of FLOPs when sequences are shorter than max_length.
    # Samples from the new dataset are stored unpadded; legacy datasets are already
    # pre-padded so collator.pad() is effectively a no-op for them.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    callbacks = []
    if model_args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(model_args.early_stopping_patience))

    # Optional structural evaluation via eval_struct.StructuralEvalCallback.
    # Only activated when STRUCT_EVAL_SKETCH_PATH is provided; keeps default
    # training path independent of TVM startup cost.
    struct_sketch = os.environ.get("STRUCT_EVAL_SKETCH_PATH")
    struct_target = os.environ.get("STRUCT_EVAL_TARGET")
    if training_args.do_eval and struct_sketch and struct_target:
        try:
            from eval_struct import StructuralEvalCallback  # noqa: WPS433
            callbacks.append(
                StructuralEvalCallback(
                    sketch_path=struct_sketch,
                    target=struct_target,
                    max_workloads=int(os.environ.get("STRUCT_EVAL_MAX_WORKLOADS", "32")),
                    max_states_per_workload=int(os.environ.get("STRUCT_EVAL_MAX_STATES", "2")),
                    max_new_tokens=int(os.environ.get("STRUCT_EVAL_MAX_NEW_TOKENS", "512")),
                    batch_size=int(os.environ.get("STRUCT_EVAL_BATCH_SIZE", "8")),
                    do_build=os.environ.get("STRUCT_EVAL_DO_BUILD", "0").lower() in {"1", "true", "yes"},
                    do_sample=os.environ.get("STRUCT_EVAL_DO_SAMPLE", "0").lower() in {"1", "true", "yes"},
                    seed=int(os.environ.get("STRUCT_EVAL_SEED", "42")),
                )
            )
            logger.info("StructuralEvalCallback enabled: sketch=%s target=%s", struct_sketch, struct_target)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to register StructuralEvalCallback: %s", exc)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=callbacks or None,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
