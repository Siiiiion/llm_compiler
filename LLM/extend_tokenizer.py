#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性给 Qwen3 tokenizer 扩充 TVM auto_scheduler 常用 token，并用子词均值
warm-start 新 embedding。产物是一个新的 `<output_dir>`，里面包含：

    config.json / model.safetensors / generation_config.json
    tokenizer_config.json / tokenizer.json / special_tokens_map.json / added_tokens.json
    extend_tokenizer_manifest.json  <-- 本脚本生成的摘要

典型用法（推荐一次性跑完）：

    python extend_tokenizer.py \
        --base_model /home/qsy/huggingface/model/Qwen3-0.6B \
        --output_dir /home/qsy/huggingface/model/Qwen3-0.6B-tvm-ext \
        --promote_marker PPT

之后将训练 / 数据集构建里所有的 `Qwen3-0.6B` 全部替换成
`Qwen3-0.6B-tvm-ext`，并重新走一遍 `make_dataset.py`，这样：

1. 生成序列更短（常见 TVM 步骤关键字 / 整数常量从 2–3 个 BPE 合为 1 个 token）
2. `PPT` 会成为单一 special token（加速 `ppt_end` 定位，也防止被拆 / 误拆）
3. 新 token 的 embedding 用对应子词均值初始化，loss 从一个合理点起步

注意事项：
- 扩词表后**必须重建数据集**：旧数据集里的 token id 是旧 tokenizer 下的，继续训练
  会导致 PPT 掩码错位、embedding lookup 位置不匹配。
- 产物 tie_word_embeddings 保持与原模型一致。
- 脚本默认把模型以原始 dtype（通常 bf16）读出 & 写回，不做精度转换。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("extend_tokenizer")


# ---------------------------------------------------------------------------
# 默认新增 token 列表（可通过 --extra_tokens_file 追加；通过 --no_* 跳过分组）
# ---------------------------------------------------------------------------

# 对应 tvm/auto_scheduler/transform_step.cc 的 step 关键字。这里保留所有已知常见的
# 后缀关键字；实际词表里没有的会 fallback 为单 token，add_tokens 会自动跳过重复。
DEFAULT_STEP_TOKENS: List[str] = [
    # 核心 step keyword
    "SP",    # SplitStep
    "FU",    # FuseStep
    "RE",    # ReorderStep
    "CA",    # ComputeAtStep
    "CI",    # ComputeInlineStep
    "CR",    # ComputeRootStep
    "CHR",   # CacheReadStep
    "CHW",   # CacheWriteStep
    "RF",    # RfactorStep
    "AN",    # AnnotationStep
    "PA",    # PragmaStep
    "FSP",   # FollowSplitStep
    "FFSP",  # FollowFusedSplitStep
    "SA",    # StorageAlignStep
    # 决策 / 终结 marker
    "PPT",   # prompt/decision 分隔
    "MEM",   # storage / memory scope 标记
    "SPC",   # special-stage / storage_align 里常见的前缀
    "ROOT",  # root stage marker
]

# 常用整数常量：调度里常见的 block/thread/tile 尺寸、循环上限等。2 的幂 + 一些常见
# 非 2 幂（12、24、48、96 等）。这些数字在 BPE 里经常被切成 2–3 个 token。
DEFAULT_INT_TOKENS: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "12", "14", "16", "20", "24", "28", "32",
    "40", "48", "56", "64", "72", "80", "96", "112", "128",
    "160", "192", "224", "256", "320", "384", "448", "512",
    "640", "768", "896", "1024", "1280", "1536", "2048", "3072", "4096",
    "-1",
]

DEFAULT_BOOL_TOKENS: List[str] = ["True", "False"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_extra_tokens_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def _dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _subword_ids(old_tokenizer, token: str) -> List[int]:
    ids = old_tokenizer(token, add_special_tokens=False)["input_ids"]
    # also try " token" (leading space) — BPE 对字首空格敏感，有时更接近真实上下文
    ids_space = old_tokenizer(f" {token}", add_special_tokens=False)["input_ids"]
    # 取更短的那一个作为初始化依据；长度相同则取无前导空格的
    if 0 < len(ids_space) < len(ids):
        return ids_space
    return ids


def _init_from_subwords(
    old_tokenizer,
    embedding_weight: torch.Tensor,
    token: str,
    orig_vocab_size: int,
) -> Optional[torch.Tensor]:
    sub_ids = _subword_ids(old_tokenizer, token)
    sub_ids = [i for i in sub_ids if 0 <= i < orig_vocab_size]
    if not sub_ids:
        return None
    idx = torch.tensor(sub_ids, dtype=torch.long, device=embedding_weight.device)
    return embedding_weight[idx].mean(dim=0).to(embedding_weight.dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extend Qwen3 tokenizer with TVM tokens and warm-start embeddings.",
    )
    parser.add_argument("--base_model", required=True,
                        help="Original Qwen3 model/tokenizer directory.")
    parser.add_argument("--output_dir", required=True,
                        help="Destination directory for the extended model + tokenizer.")
    parser.add_argument("--extra_tokens_file", default=None,
                        help="Optional file with one extra token per line (lines starting with '#' are ignored).")
    parser.add_argument("--promote_marker", default=None,
                        help="If set, add the marker (eg. 'PPT') as a SPECIAL token "
                             "so it is always emitted as a single id.")
    parser.add_argument("--no_steps", action="store_true",
                        help="Skip step-keyword tokens.")
    parser.add_argument("--no_ints", action="store_true",
                        help="Skip integer-constant tokens.")
    parser.add_argument("--no_bool", action="store_true",
                        help="Skip True/False tokens.")
    parser.add_argument("--pad_to_multiple_of", type=int, default=8,
                        help="After resize, pad embedding rows to this multiple (default=8 for TensorCore).")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                        help="Pass trust_remote_code=True to HF loaders (Qwen3 requires it).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would happen; do not save.")
    return parser.parse_args()


def collect_candidate_tokens(args) -> List[str]:
    extra: List[str] = []
    if not args.no_steps:
        extra.extend(DEFAULT_STEP_TOKENS)
    if not args.no_ints:
        extra.extend(DEFAULT_INT_TOKENS)
    if not args.no_bool:
        extra.extend(DEFAULT_BOOL_TOKENS)
    if args.extra_tokens_file:
        extra.extend(_load_extra_tokens_file(args.extra_tokens_file))
    return _dedup_preserve_order(extra)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    args = parse_args()

    logger.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    # snapshot the ORIGINAL tokenizer before we mutate it; used for subword lookups
    old_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)

    logger.info("Loading model from %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
    )
    model.eval()

    orig_vocab_size = len(tokenizer)
    orig_embed_rows = model.get_input_embeddings().weight.shape[0]
    logger.info("Original tokenizer vocab size = %d", orig_vocab_size)
    logger.info("Original input embedding rows = %d", orig_embed_rows)
    logger.info("tie_word_embeddings = %s", getattr(model.config, "tie_word_embeddings", None))

    candidates = collect_candidate_tokens(args)
    logger.info("Candidate extra tokens (pre-filter) = %d", len(candidates))

    # Promote marker goes in first so its id ends up stable/small within the added range.
    promoted: Optional[str] = args.promote_marker
    if promoted:
        if promoted in candidates:
            candidates.remove(promoted)

    # Filter out tokens that already exist as a single id in the original tokenizer —
    # re-adding them would be a no-op at best or an id collision at worst.
    filtered: List[str] = []
    already_single: List[str] = []
    for tok in candidates:
        ids = old_tokenizer(tok, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            already_single.append(tok)
            continue
        filtered.append(tok)

    logger.info("Skipping %d tokens that are already a single id: %s",
                len(already_single), already_single[:20])
    logger.info("Will request add_tokens for %d new tokens", len(filtered))

    if args.dry_run:
        logger.info("Dry run: promoted_marker=%s, new_tokens=%s", promoted, filtered)
        return

    n_special = 0
    if promoted:
        n_special = tokenizer.add_tokens([promoted], special_tokens=True)
        logger.info("Added %d special token(s): %r", n_special, promoted)

    n_added = tokenizer.add_tokens(filtered, special_tokens=False)
    logger.info("Added %d normal tokens", n_added)

    new_vocab_size = len(tokenizer)
    logger.info("New tokenizer vocab size = %d (delta=%d)",
                new_vocab_size, new_vocab_size - orig_vocab_size)

    # Resize embeddings, pad to multiple-of for TensorCore friendliness.
    model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=args.pad_to_multiple_of)
    new_embed_rows = model.get_input_embeddings().weight.shape[0]
    logger.info("New input embedding rows = %d", new_embed_rows)

    # Warm-start every newly added id using subword means under the OLD tokenizer.
    input_embed = model.get_input_embeddings().weight.data
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    output_embed = None
    if not tied:
        out_mod = model.get_output_embeddings()
        if out_mod is not None and hasattr(out_mod, "weight"):
            output_embed = out_mod.weight.data

    all_new_tokens: List[str] = ([promoted] if promoted else []) + filtered
    n_init = 0
    with torch.no_grad():
        for tok_str in all_new_tokens:
            new_id = tokenizer.convert_tokens_to_ids(tok_str)
            if new_id is None or new_id < orig_vocab_size or new_id >= new_embed_rows:
                continue
            vec = _init_from_subwords(old_tokenizer, input_embed, tok_str, orig_vocab_size)
            if vec is None:
                continue
            input_embed[new_id].copy_(vec)
            if output_embed is not None:
                vec_out = _init_from_subwords(old_tokenizer, output_embed, tok_str, orig_vocab_size)
                if vec_out is not None:
                    output_embed[new_id].copy_(vec_out)
            n_init += 1
    logger.info("Warm-started %d / %d new embedding rows", n_init, len(all_new_tokens))

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Saving extended tokenizer + model to %s", args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    manifest = {
        "base_model": args.base_model,
        "output_dir": args.output_dir,
        "promoted_marker": promoted,
        "added_special_tokens": [promoted] if promoted else [],
        "added_normal_tokens": filtered,
        "already_single_token_skipped": already_single,
        "orig_vocab_size": orig_vocab_size,
        "new_vocab_size": new_vocab_size,
        "new_embedding_rows": new_embed_rows,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "tie_word_embeddings": tied,
        "warm_started_rows": n_init,
    }
    manifest_path = os.path.join(args.output_dir, "extend_tokenizer_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest written to %s", manifest_path)

    # Sanity check
    for tok in all_new_tokens[:5]:
        ids_new = tokenizer(tok, add_special_tokens=False)["input_ids"]
        ids_old = old_tokenizer(tok, add_special_tokens=False)["input_ids"]
        logger.info("  %-8s  old_ids=%s  new_ids=%s", tok, ids_old, ids_new)


if __name__ == "__main__":
    main()
