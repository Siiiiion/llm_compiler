#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3 TVM schedule 生成器的 DPO / LAPO 训练脚本（创新点 2）。

该脚本与 ``train_qwen3_clm.py`` 并列，**不修改 stage1 / stage2 的任何代码**。

关键特性：
    - ``--training_mode`` 是核心可控参数，取值为 ``{sft, dpo, lapo}``：
        * ``sft``  —— 基线：只在 chosen 上做 CLM，视作 stage2 的替代基线；
        * ``dpo``  —— 标准 DPO (Rafailov et al., 2024) 的 ``sigmoid`` 损失；
        * ``lapo`` —— Latency-Weighted DPO：按 ``latency_weight`` 逐样本加权。
    - 从 ``build_preference_pairs.py`` 产出的 JSONL 直接加载；无需修改旧数据管线。
    - 使用 HuggingFace ``Trainer`` 自己实现 DPO 损失，不依赖 ``trl``。
    - DDP / bf16 / dynamic padding 全部沿用 CLM 训练的配置约定。
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import accelerate
import numpy as np
import torch
import torch.nn.functional as F

# numpy._core 兼容 shim（保持与 train_qwen3_clm.py 一致）
if not hasattr(np, "_core") and hasattr(np, "core"):
    np._core = np.core

if "keep_torch_compile" not in inspect.signature(accelerate.Accelerator.unwrap_model).parameters:
    _orig_unwrap_model = accelerate.Accelerator.unwrap_model

    def _unwrap_model_compat(self, model, keep_fp32_wrapper=True, keep_torch_compile=None):
        return _orig_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    accelerate.Accelerator.unwrap_model = _unwrap_model_compat

import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)


TRAINING_MODES = ("sft", "dpo", "lapo")


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "policy 模型路径（通常是 stage1 / stage2 的 checkpoint）"},
    )
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "参考模型路径；为空则复用 model_name_or_path（推荐）"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer 路径；为空则复用 model_name_or_path"},
    )
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    low_cpu_mem_usage: bool = field(default=True)
    trust_remote_code: bool = field(default=True)


@dataclass
class DPODataArguments:
    preference_dataset: str = field(
        metadata={"help": "build_preference_pairs.py 输出目录（含 train_pairs.jsonl）"},
    )
    train_file_name: str = field(default="train_pairs.jsonl")
    validation_file_name: str = field(default="validation_pairs.jsonl")
    max_prompt_length: int = field(default=512)
    max_length: int = field(default=1024)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    sample_seed: int = field(default=42)
    append_eos_to_suffix: bool = field(
        default=True,
        metadata={"help": "在 chosen / rejected 末尾追加 eos，让模型学到终止位置"},
    )


@dataclass
class DPOTrainingExtraArguments:
    training_mode: str = field(
        default="dpo",
        metadata={
            "help": f"训练模式：{TRAINING_MODES}；sft 表示只在 chosen 上做 CLM"
        },
    )
    beta: float = field(default=0.1, metadata={"help": "DPO 温度系数 β"})
    label_smoothing: float = field(default=0.0, metadata={"help": "IPO-like 标签平滑"})
    loss_type: str = field(
        default="sigmoid",
        metadata={"choices": ["sigmoid", "hinge"]},
    )
    lapo_weight_clip: float = field(
        default=10.0,
        metadata={"help": "LAPO 模式下对 latency_weight 的上限裁剪"},
    )
    lapo_normalize: bool = field(
        default=True,
        metadata={"help": "LAPO 模式下将 batch 内权重归一化到均值 1，稳定学习率"},
    )
    log_reward_margin: bool = field(default=True)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _tokenize_pair(
    example: Dict[str, Any],
    tokenizer,
    max_prompt_length: int,
    max_length: int,
    append_eos: bool,
) -> Dict[str, Any]:
    """将单个偏好对样本编码成 (chosen_input_ids, rejected_input_ids) 等字段。"""

    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(example["chosen"], add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(example["rejected"], add_special_tokens=False)["input_ids"]

    if append_eos and tokenizer.eos_token_id is not None:
        eos = tokenizer.eos_token_id
        if not chosen_ids or chosen_ids[-1] != eos:
            chosen_ids = list(chosen_ids) + [eos]
        if not rejected_ids or rejected_ids[-1] != eos:
            rejected_ids = list(rejected_ids) + [eos]

    if len(prompt_ids) > max_prompt_length:
        # 从左侧截断 prompt：保留靠近 PPT 的末尾，因为那是决策上下文的关键
        prompt_ids = prompt_ids[-max_prompt_length:]

    def _assemble(suffix_ids: List[int]) -> Tuple[List[int], List[int]]:
        if len(prompt_ids) + len(suffix_ids) > max_length:
            keep = max_length - len(prompt_ids)
            if keep <= 0:
                # prompt 已经占满；放弃该样本（调用方需处理）
                return [], []
            suffix_ids = suffix_ids[:keep]
        input_ids = list(prompt_ids) + list(suffix_ids)
        labels = [-100] * len(prompt_ids) + list(suffix_ids)
        return input_ids, labels

    chosen_input_ids, chosen_labels = _assemble(chosen_ids)
    rejected_input_ids, rejected_labels = _assemble(rejected_ids)

    if not chosen_input_ids or not rejected_input_ids:
        return {
            "valid": 0,
            "chosen_input_ids": [tokenizer.pad_token_id or 0],
            "chosen_labels": [-100],
            "rejected_input_ids": [tokenizer.pad_token_id or 0],
            "rejected_labels": [-100],
            "prompt_length": 0,
            "latency_weight": 1.0,
            "latency_gap": 1.0,
        }

    chosen_suffix_len = sum(1 for x in chosen_labels if x != -100)
    rejected_suffix_len = sum(1 for x in rejected_labels if x != -100)
    if chosen_suffix_len == 0 or rejected_suffix_len == 0:
        return {
            "valid": 0,
            "chosen_input_ids": [tokenizer.pad_token_id or 0],
            "chosen_labels": [-100],
            "rejected_input_ids": [tokenizer.pad_token_id or 0],
            "rejected_labels": [-100],
            "prompt_length": 0,
            "latency_weight": 1.0,
            "latency_gap": 1.0,
        }

    return {
        "valid": 1,
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
        "prompt_length": len(prompt_ids),
        "latency_weight": float(example.get("latency_weight", 1.0)),
        "latency_gap": float(example.get("latency_gap", 1.0)),
    }


def _load_preference_dataset(
    data_args: DPODataArguments,
    tokenizer,
) -> Tuple[Any, Optional[Any]]:
    train_path = os.path.join(data_args.preference_dataset, data_args.train_file_name)
    validation_path = os.path.join(
        data_args.preference_dataset, data_args.validation_file_name
    )
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Train preference JSONL not found: {train_path}")

    data_files = {"train": train_path}
    if os.path.isfile(validation_path):
        data_files["validation"] = validation_path

    raw = load_dataset("json", data_files=data_files, keep_in_memory=True)

    columns = raw["train"].column_names

    def _map_fn(example):
        return _tokenize_pair(
            example,
            tokenizer=tokenizer,
            max_prompt_length=data_args.max_prompt_length,
            max_length=data_args.max_length,
            append_eos=data_args.append_eos_to_suffix,
        )

    tokenized = raw.map(_map_fn, remove_columns=columns, desc="Tokenizing preference pairs")
    tokenized = tokenized.filter(lambda x: x["valid"] == 1)
    tokenized = tokenized.remove_columns(["valid"])

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"] if "validation" in tokenized else None
    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


class DPOCollator:
    """把 chosen / rejected 两条序列同时 pad 到 batch 最大长度。"""

    def __init__(self, tokenizer, pad_to_multiple_of: int = 8, sft_only: bool = False):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.sft_only = sft_only
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _pad(self, seqs: List[List[int]], pad_val: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        if self.pad_to_multiple_of > 1:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                       // self.pad_to_multiple_of
                       * self.pad_to_multiple_of)
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            if len(s) > 0:
                out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_ids = [f["chosen_input_ids"] for f in features]
        chosen_labels = [f["chosen_labels"] for f in features]
        chosen_ids_tensor = self._pad(chosen_ids, self.pad_id)
        chosen_labels_tensor = self._pad(chosen_labels, -100)
        chosen_attn = (chosen_ids_tensor != self.pad_id).long()

        latency_weight = torch.tensor(
            [float(f.get("latency_weight", 1.0)) for f in features], dtype=torch.float32
        )
        latency_gap = torch.tensor(
            [float(f.get("latency_gap", 1.0)) for f in features], dtype=torch.float32
        )
        prompt_length = torch.tensor(
            [int(f.get("prompt_length", 0)) for f in features], dtype=torch.long
        )

        if self.sft_only:
            return {
                "chosen_input_ids": chosen_ids_tensor,
                "chosen_attention_mask": chosen_attn,
                "chosen_labels": chosen_labels_tensor,
                "latency_weight": latency_weight,
                "latency_gap": latency_gap,
                "prompt_length": prompt_length,
            }

        rejected_ids = [f["rejected_input_ids"] for f in features]
        rejected_labels = [f["rejected_labels"] for f in features]
        rejected_ids_tensor = self._pad(rejected_ids, self.pad_id)
        rejected_labels_tensor = self._pad(rejected_labels, -100)
        rejected_attn = (rejected_ids_tensor != self.pad_id).long()

        return {
            "chosen_input_ids": chosen_ids_tensor,
            "chosen_attention_mask": chosen_attn,
            "chosen_labels": chosen_labels_tensor,
            "rejected_input_ids": rejected_ids_tensor,
            "rejected_attention_mask": rejected_attn,
            "rejected_labels": rejected_labels_tensor,
            "latency_weight": latency_weight,
            "latency_gap": latency_gap,
            "prompt_length": prompt_length,
        }


# ---------------------------------------------------------------------------
# DPO loss utilities
# ---------------------------------------------------------------------------


def _compute_sequence_logps(
    logits: torch.Tensor, labels: torch.Tensor, label_pad_id: int = -100
) -> torch.Tensor:
    """返回每条序列的 suffix token log-prob 总和。形状 (B,)。"""
    # logits: (B, T, V); labels: (B, T)
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].clone()
    mask = (shifted_labels != label_pad_id)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(shifted_logits.float(), dim=-1)
    per_token = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token = per_token * mask.float()
    return per_token.sum(dim=-1)


def _dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    loss_type: str,
    label_smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pi_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratio - ref_logratio  # (B,)
    if loss_type == "sigmoid":
        loss = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "hinge":
        loss = torch.relu(1 - beta * logits)
    else:
        raise ValueError(f"Unknown loss_type={loss_type}")

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------


class DPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        ref_model: Optional[torch.nn.Module] = None,
        training_mode: str = "dpo",
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
        lapo_weight_clip: float = 10.0,
        lapo_normalize: bool = True,
        log_reward_margin: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.training_mode = training_mode
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.lapo_weight_clip = lapo_weight_clip
        self.lapo_normalize = lapo_normalize
        self.log_reward_margin = log_reward_margin
        self._last_metrics: Dict[str, float] = {}

        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            ds_enabled = getattr(self, "is_deepspeed_enabled", False)
            if ds_enabled:
                # deepspeed 路径暂不支持 prepare_deepspeed；直接按 device 迁移
                self.ref_model = self.ref_model.to(self.args.device)
            else:
                self.ref_model = self._prepare_ref_model(self.ref_model)

    def _prepare_ref_model(self, ref_model: torch.nn.Module) -> torch.nn.Module:
        device = self.args.device
        dtype = None
        if self.args.bf16:
            dtype = torch.bfloat16
        elif self.args.fp16:
            dtype = torch.float16
        if dtype is not None:
            ref_model = ref_model.to(dtype=dtype)
        return ref_model.to(device)

    # -------- forward helpers --------

    def _forward_logps(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        input_ids = batch[f"{key}_input_ids"]
        attention_mask = batch[f"{key}_attention_mask"]
        labels = batch[f"{key}_labels"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return _compute_sequence_logps(outputs.logits, labels)

    # -------- main loss --------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = self.training_mode
        if mode == "sft":
            outputs = model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
                labels=inputs["chosen_labels"],
                use_cache=False,
            )
            loss = outputs.loss
            self._last_metrics = {
                "loss/sft": float(loss.detach().cpu().item()),
            }
            return (loss, outputs) if return_outputs else loss

        if mode not in ("dpo", "lapo"):
            raise ValueError(f"Unknown training_mode={mode}")

        policy_chosen_logps = self._forward_logps(model, inputs, "chosen")
        policy_rejected_logps = self._forward_logps(model, inputs, "rejected")

        if self.ref_model is None:
            raise RuntimeError("DPO / LAPO requires a reference model")

        with torch.no_grad():
            ref_chosen_logps = self._forward_logps(self.ref_model, inputs, "chosen")
            ref_rejected_logps = self._forward_logps(self.ref_model, inputs, "rejected")

        per_sample_loss, chosen_rewards, rejected_rewards = _dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta,
            loss_type=self.loss_type,
            label_smoothing=self.label_smoothing,
        )

        if mode == "lapo":
            w = inputs["latency_weight"].to(per_sample_loss.dtype).to(per_sample_loss.device)
            if self.lapo_weight_clip and self.lapo_weight_clip > 0:
                w = w.clamp(min=0.0, max=float(self.lapo_weight_clip))
            if self.lapo_normalize:
                w_mean = w.mean().clamp(min=1e-6)
                w = w / w_mean
            loss = (per_sample_loss * w).mean()
        else:
            loss = per_sample_loss.mean()

        metrics = {
            "loss/dpo": float(loss.detach().cpu().item()),
            "reward/chosen_mean": float(chosen_rewards.mean().cpu().item()),
            "reward/rejected_mean": float(rejected_rewards.mean().cpu().item()),
            "reward/accuracy": float((chosen_rewards > rejected_rewards).float().mean().cpu().item()),
        }
        if self.log_reward_margin:
            metrics["reward/margin_mean"] = float(
                (chosen_rewards - rejected_rewards).mean().cpu().item()
            )
        self._last_metrics = metrics

        return (loss, None) if return_outputs else loss

    # Attach detailed metrics onto Trainer log stream.
    def log(self, logs: Dict[str, float], *args, **kwargs):
        if self._last_metrics:
            for k, v in self._last_metrics.items():
                logs.setdefault(k, v)
            self._last_metrics = {}
        return super().log(logs, *args, **kwargs)

    # Evaluation fallback: reuse compute_loss and accuracy-like margin metric.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        loss = loss.detach()
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, None, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_tokenizer(model_args: ModelArguments, path: str):
    kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": model_args.trust_remote_code,
    }
    return AutoTokenizer.from_pretrained(path, **kwargs)


def _load_causal_model(model_args: ModelArguments, path: str):
    config = AutoConfig.from_pretrained(
        path,
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
        path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        trust_remote_code=model_args.trust_remote_code,
    )
    return model


def main():
    parser = HfArgumentParser(
        (ModelArguments, DPODataArguments, DPOTrainingExtraArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, dpo_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        cli_args = []
        for arg in sys.argv[1:]:
            if arg == "--evaluation_strategy":
                cli_args.append("--eval_strategy")
            elif arg.startswith("--evaluation_strategy="):
                cli_args.append(arg.replace("--evaluation_strategy=", "--eval_strategy=", 1))
            else:
                cli_args.append(arg)
        model_args, data_args, dpo_args, training_args = parser.parse_args_into_dataclasses(
            args=cli_args
        )

    if dpo_args.training_mode not in TRAINING_MODES:
        raise ValueError(
            f"--training_mode must be one of {TRAINING_MODES}, got {dpo_args.training_mode}"
        )

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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"bf16: {training_args.bf16}, training_mode: {dpo_args.training_mode}"
    )

    set_seed(training_args.seed)

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    tokenizer_path = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = _load_tokenizer(model_args, tokenizer_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset = _load_preference_dataset(data_args, tokenizer)

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=data_args.sample_seed).select(
            range(min(len(train_dataset), data_args.max_train_samples))
        )
    if eval_dataset is not None and data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.shuffle(seed=data_args.sample_seed + 1).select(
            range(min(len(eval_dataset), data_args.max_eval_samples))
        )

    logger.info("train_dataset size = %d", len(train_dataset))
    if eval_dataset is not None:
        logger.info("eval_dataset size = %d", len(eval_dataset))

    model = _load_causal_model(model_args, model_args.model_name_or_path)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    ref_model = None
    if dpo_args.training_mode in ("dpo", "lapo"):
        ref_path = model_args.ref_model_name_or_path or model_args.model_name_or_path
        logger.info("Loading reference model from %s", ref_path)
        ref_model = _load_causal_model(model_args, ref_path)
        if ref_model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            ref_model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DPOCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        sft_only=(dpo_args.training_mode == "sft"),
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        ref_model=ref_model,
        training_mode=dpo_args.training_mode,
        beta=dpo_args.beta,
        loss_type=dpo_args.loss_type,
        label_smoothing=dpo_args.label_smoothing,
        lapo_weight_clip=dpo_args.lapo_weight_clip,
        lapo_normalize=dpo_args.lapo_normalize,
        log_reward_margin=dpo_args.log_reward_margin,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["training_mode"] = dpo_args.training_mode
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        manifest = {
            "training_mode": dpo_args.training_mode,
            "beta": dpo_args.beta,
            "loss_type": dpo_args.loss_type,
            "label_smoothing": dpo_args.label_smoothing,
            "lapo_weight_clip": dpo_args.lapo_weight_clip,
            "lapo_normalize": dpo_args.lapo_normalize,
            "preference_dataset": data_args.preference_dataset,
            "max_prompt_length": data_args.max_prompt_length,
            "max_length": data_args.max_length,
            "model_name_or_path": model_args.model_name_or_path,
            "ref_model_name_or_path": model_args.ref_model_name_or_path or model_args.model_name_or_path,
        }
        with open(os.path.join(training_args.output_dir, "dpo_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        if eval_dataset is not None:
            metrics["eval_samples"] = len(eval_dataset)
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except (OverflowError, KeyError):
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
