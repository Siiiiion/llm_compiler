#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural evaluation for the LLM-as-SketchPolicy pipeline.

此脚本用一个小规模 workload 样本，量化「模型生成的 decision suffix 能否被
TVM auto_scheduler / LocalBuilder 接受」，输出如下指标：

- parse_valid_rate    : 被 SketchPolicy.gen_states 成功解析成 state 的比例
- build_valid_rate    : 经 LocalBuilder 构建 (error_no == 0) 的比例（需要 --do_build）
- fallback_rate       : 全部解析失败、只能回退 sketch 的 workload 比例
- distinct_rate       : 生成 state 去重后数量 / 生成 state 总数
- avg_parse_per_sketch: 每个 workload 成功 parse 的 state 平均数
- samples_per_sec     : 端到端评估吞吐

使用方式:
1) 独立 CLI（训练完跑一轮）:
     python eval_struct.py \
         --model_name_or_path /path/to/ckpt \
         --sketch_path /path/to/sketch.json \
         --target "cuda -model=4090" \
         --max_workloads 64 \
         --max_new_tokens 512 \
         --output_json ./eval_struct.json

2) 作为 `StructuralEvalCallback` 被 `train_qwen3_clm.py` 装入 Trainer，
   每次 on_evaluate 时把 eval_parse_valid_rate 等写进 metrics 字典。
   启用方法：设置环境变量 `STRUCT_EVAL_SKETCH_PATH` 与 `STRUCT_EVAL_TARGET`。
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("eval_struct")


# ---------------------------------------------------------------------------
# TVM-side helpers (lazy-imported so the Trainer process only pays the cost if
# structural eval is actually enabled).
# ---------------------------------------------------------------------------


def _require_tvm():
    import tvm  # noqa: F401
    from tvm import auto_scheduler  # noqa: F401

    return tvm, auto_scheduler


def _setup_target_and_tasks(target_str: str):
    tvm, _ = _require_tvm()

    # Make local modules importable even if eval_struct.py is invoked from a
    # different CWD (eg by Trainer in another working dir).
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.append(this_dir)

    from common import load_and_register_tasks, register_data_path  # noqa: WPS433

    register_data_path(target_str)
    target = tvm.target.Target(target_str)
    tasks = load_and_register_tasks()
    return target, tasks


def _read_sketch_workloads(sketch_path: str, max_workloads: int, seed: int):
    _, auto_scheduler = _require_tvm()

    inputs, _ = auto_scheduler.RecordReader(sketch_path).read_lines()
    inp_dic: Dict[str, object] = {}
    sketch_dic: Dict[str, List[object]] = {}

    for inp in inputs:
        inp_str = inp.to_json()
        if inp_str in inp_dic:
            inp = auto_scheduler.measure.recover_measure_input(inp_dic[inp_str])
        else:
            inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
            inp_dic[inp_str] = inp
        sketch_dic.setdefault(inp.task.workload_key, []).append(inp)

    items = list(sketch_dic.items())
    if seed is not None:
        random.Random(seed).shuffle(items)
    if max_workloads and max_workloads > 0:
        items = items[:max_workloads]
    return items


# ---------------------------------------------------------------------------
# Core evaluation.
# ---------------------------------------------------------------------------


def _decode_decision_tokens(tokenizer, token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = text.strip()
    if not text:
        return []
    return text.split()


def _resolve_max_new_tokens(requested, min_tokens, scale):
    requested = max(int(requested), 1)
    scaled = max(1, math.ceil(requested * scale))
    return max(requested, scaled, min_tokens)


def _generate_decisions_for_states(
    task,
    states,
    input_obj,
    tokenizer,
    model,
    device,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    top_k: int,
    temperature: float,
    batch_size: int,
    trim_last_input_token: bool,
):
    """Mirror of `gen_state.gen_func`, returns list-of-list-of-strings (tokens)."""
    from make_dataset import input_to_tokens  # local import to avoid hard dep at CLI import time

    if len(states) == 0:
        return []

    tokens = input_to_tokens(task, states, input_obj)
    if len(tokens) == 0:
        return []

    tokenizer.padding_side = "left"
    batch = tokenizer(tokens, padding=True, max_length=None)
    input_ids_all = batch["input_ids"]
    attn_all = batch["attention_mask"]

    gen_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs.update({"top_p": top_p, "top_k": top_k, "temperature": temperature})
    if tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    responses: List[List[int]] = []
    with torch.no_grad():
        for start in range(0, len(input_ids_all), batch_size):
            ids = input_ids_all[start : start + batch_size]
            am = attn_all[start : start + batch_size]
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            am_t = torch.tensor(am, dtype=torch.long, device=device)
            if trim_last_input_token and ids_t.shape[-1] > 1:
                ids_t = ids_t[:, :-1]
                am_t = am_t[:, :-1]
            # Clamp the generation budget to something sensible (Qwen3 advertises
            # 131072 which makes the default look harmless but in pathological
            # callers causes OOM / runaway generation).
            max_model_len = min(int(getattr(tokenizer, "model_max_length", 2048) or 2048), 8192)
            budget = max_model_len - ids_t.shape[-1]
            if budget <= 0:
                responses.extend([[] for _ in range(ids_t.shape[0])])
                continue
            local_kwargs = dict(gen_kwargs)
            local_kwargs["max_new_tokens"] = min(local_kwargs["max_new_tokens"], budget)
            out = model.generate(input_ids=ids_t, attention_mask=am_t, **local_kwargs)
            out = out[:, ids_t.shape[-1] :]
            responses.extend(out.tolist())

    return [_decode_decision_tokens(tokenizer, r) for r in responses]


def evaluate_structural_validity(
    model,
    tokenizer,
    sketch_workloads: List[Tuple[str, List[object]]],
    device,
    max_states_per_workload: int = 2,
    max_new_tokens: int = 512,
    min_gen_tokens: int = 128,
    gen_token_scale: float = 2.0,
    do_sample: bool = False,
    top_p: float = 0.95,
    top_k: int = 50,
    temperature: float = 0.7,
    batch_size: int = 8,
    do_build: bool = False,
    build_timeout: int = 15,
    trim_last_input_token: bool = False,
) -> Dict[str, float]:
    """Run structural evaluation over a list of (workload_key, [MeasureInput]) tuples.

    Returns a metrics dict with prefix-less keys (ie. `parse_valid_rate`), the
    caller prepends `eval_` if needed.
    """
    _, auto_scheduler = _require_tvm()

    total_inputs = 0
    total_parsed = 0
    total_generated = 0
    workload_fallback = 0
    distinct_inputs = 0

    builder = None
    built_ok = 0
    built_total = 0
    if do_build:
        builder = auto_scheduler.measure.LocalBuilder(timeout=build_timeout)

    t0 = time.time()

    for workload_key, inputs in sketch_workloads:
        if not inputs:
            continue
        selected = inputs[:max_states_per_workload] if max_states_per_workload > 0 else inputs
        total_inputs += len(selected)

        task = selected[0].task
        policy = auto_scheduler.SketchPolicy(task)

        resolved_max_new_tokens = _resolve_max_new_tokens(
            max_new_tokens,
            min_tokens=min_gen_tokens,
            scale=gen_token_scale,
        )

        def _gen_func(_task, states, _max_new_tokens):
            return _generate_decisions_for_states(
                _task,
                states,
                selected[0],
                tokenizer,
                model,
                device,
                max_new_tokens=resolved_max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                batch_size=batch_size,
                trim_last_input_token=trim_last_input_token,
            )

        try:
            parsed_states = policy.gen_states([inp.state for inp in selected], _gen_func)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("gen_states failed for workload=%s: %s", workload_key[:64], exc)
            parsed_states = []

        total_generated += len(selected)  # one attempt per input
        total_parsed += len(parsed_states)

        if len(parsed_states) == 0:
            workload_fallback += 1
            continue

        seen_inputs = set()
        unique_measure_inputs = []
        for st in parsed_states:
            inp = auto_scheduler.MeasureInput(task, st)
            s = inp.to_json()
            if s in seen_inputs:
                continue
            seen_inputs.add(s)
            unique_measure_inputs.append(inp)
        distinct_inputs += len(unique_measure_inputs)

        if do_build and unique_measure_inputs:
            try:
                build_results = builder.build(unique_measure_inputs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("LocalBuilder.build failed for workload=%s: %s", workload_key[:64], exc)
                build_results = []
            built_total += len(unique_measure_inputs)
            built_ok += sum(1 for r in build_results if getattr(r, "error_no", -1) == 0)

    elapsed = max(1e-6, time.time() - t0)
    num_workloads = len(sketch_workloads)

    result: Dict[str, float] = {
        "parse_valid_rate": total_parsed / total_generated if total_generated else 0.0,
        "fallback_rate": workload_fallback / num_workloads if num_workloads else 0.0,
        "distinct_rate": distinct_inputs / total_parsed if total_parsed else 0.0,
        "avg_parse_per_sketch": total_parsed / num_workloads if num_workloads else 0.0,
        "num_workloads": float(num_workloads),
        "num_generated": float(total_generated),
        "samples_per_sec": total_generated / elapsed,
        "elapsed_sec": elapsed,
    }
    if do_build:
        result["build_valid_rate"] = built_ok / built_total if built_total else 0.0
        result["num_built"] = float(built_total)
    return result


# ---------------------------------------------------------------------------
# TrainerCallback
# ---------------------------------------------------------------------------


class StructuralEvalCallback:
    """Lazy-loaded TrainerCallback (avoids importing transformers at module import)."""

    def __new__(cls, *args, **kwargs):
        from transformers import TrainerCallback as _TrainerCallback

        class _Impl(_TrainerCallback):
            def __init__(
                self,
                sketch_path: str,
                target: str,
                max_workloads: int = 32,
                max_states_per_workload: int = 2,
                max_new_tokens: int = 512,
                batch_size: int = 8,
                do_build: bool = False,
                do_sample: bool = False,
                seed: int = 42,
                metric_prefix: str = "eval_struct_",
                run_every_eval: bool = True,
                only_on_world_zero: bool = True,
            ):
                super().__init__()
                self.sketch_path = sketch_path
                self.target_str = target
                self.max_workloads = max_workloads
                self.max_states_per_workload = max_states_per_workload
                self.max_new_tokens = max_new_tokens
                self.batch_size = batch_size
                self.do_build = do_build
                self.do_sample = do_sample
                self.seed = seed
                self.metric_prefix = metric_prefix
                self.run_every_eval = run_every_eval
                self.only_on_world_zero = only_on_world_zero
                self._cached_workloads: Optional[List[Tuple[str, List[object]]]] = None
                self._tvm_ready = False

            def _ensure_tvm(self):
                if self._tvm_ready:
                    return
                _setup_target_and_tasks(self.target_str)
                self._cached_workloads = _read_sketch_workloads(
                    self.sketch_path, self.max_workloads, self.seed
                )
                self._tvm_ready = True
                logger.info(
                    "StructuralEvalCallback ready: %d workloads loaded from %s",
                    len(self._cached_workloads),
                    self.sketch_path,
                )

            def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
                if self.only_on_world_zero and getattr(args, "process_index", 0) != 0:
                    return
                if not self.run_every_eval:
                    return
                metrics = kwargs.get("metrics")
                model = kwargs.get("model")
                tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
                if metrics is None or model is None or tokenizer is None:
                    logger.warning("StructuralEvalCallback invoked without metrics/model/tokenizer")
                    return

                try:
                    self._ensure_tvm()
                except Exception as exc:
                    logger.exception("StructuralEvalCallback TVM setup failed: %s", exc)
                    return

                was_training = model.training
                orig_use_cache = getattr(model.config, "use_cache", None)
                try:
                    model.eval()
                    model.config.use_cache = True
                    device = next(model.parameters()).device
                    result = evaluate_structural_validity(
                        model=model,
                        tokenizer=tokenizer,
                        sketch_workloads=self._cached_workloads or [],
                        device=device,
                        max_states_per_workload=self.max_states_per_workload,
                        max_new_tokens=self.max_new_tokens,
                        batch_size=self.batch_size,
                        do_build=self.do_build,
                        do_sample=self.do_sample,
                    )
                except Exception as exc:
                    logger.exception("Structural eval failed: %s", exc)
                    return
                finally:
                    if orig_use_cache is not None:
                        model.config.use_cache = orig_use_cache
                    if was_training:
                        model.train()

                for key, value in result.items():
                    metrics[f"{self.metric_prefix}{key}"] = float(value)
                logger.info("StructuralEvalCallback metrics: %s", result)

        return _Impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class CliArgs:
    model_name_or_path: str = field(metadata={"help": "Fine-tuned checkpoint path"})
    sketch_path: str = field(metadata={"help": "Sketch records (same format as gen_state.py)"})
    target: str = field(metadata={"help": "TVM target string, e.g. 'cuda -model=4090'"})

    tokenizer_name: Optional[str] = field(default=None)
    output_json: str = field(default="eval_struct.json")

    max_workloads: int = field(default=64)
    max_states_per_workload: int = field(default=2)
    max_new_tokens: int = field(default=512)
    batch_size: int = field(default=8)
    do_build: bool = field(default=False)
    do_sample: bool = field(default=False)
    top_p: float = field(default=0.95)
    top_k: int = field(default=50)
    temperature: float = field(default=0.7)
    seed: int = field(default=42)
    trust_remote_code: bool = field(default=True)
    trim_last_input_token: bool = field(default=False)
    device: Optional[str] = field(default=None)


def _load_model_tokenizer(args: CliArgs):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    dtype_kwargs = {"trust_remote_code": args.trust_remote_code}
    try:
        dtype_kwargs["dtype"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **dtype_kwargs)
    except TypeError:
        dtype_kwargs.pop("dtype", None)
        dtype_kwargs["torch_dtype"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **dtype_kwargs)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tok, device


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    from transformers import HfArgumentParser  # lazy import keeps `import eval_struct` cheap

    parser = HfArgumentParser(CliArgs)
    (args,) = parser.parse_args_into_dataclasses()

    _setup_target_and_tasks(args.target)
    sketch_workloads = _read_sketch_workloads(args.sketch_path, args.max_workloads, args.seed)
    logger.info("Loaded %d workloads from %s", len(sketch_workloads), args.sketch_path)

    model, tokenizer, device = _load_model_tokenizer(args)
    logger.info("Model/tokenizer loaded on %s", device)

    result = evaluate_structural_validity(
        model=model,
        tokenizer=tokenizer,
        sketch_workloads=sketch_workloads,
        device=device,
        max_states_per_workload=args.max_states_per_workload,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        do_build=args.do_build,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        trim_last_input_token=args.trim_last_input_token,
    )

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "sketch_path": args.sketch_path,
        "target": args.target,
        "max_workloads": args.max_workloads,
        "do_build": args.do_build,
        "do_sample": args.do_sample,
        "metrics": result,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output_json)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as fout:
        json.dump(payload, fout, indent=2)

    logger.info("Structural eval metrics: %s", result)
    logger.info("Report written to %s", args.output_json)


if __name__ == "__main__":
    main()
