#!/usr/bin/env python3
"""Small, isolated old-source/filtered-source training and generation comparison."""

from LLM.core.layout import LLM_ROOT, project_pythonpath


import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from LLM.core.io import atomic_json
from LLM.core.schedule_record_utils import digest, normalize_record
from LLM.core.runtime import now


def prepare(directory, rebuild, original_raw, tokenizer_path, count=256):
    from datasets import Dataset, DatasetDict
    from LLM.generation.gen_state import load_tokenizer
    from LLM.core.tokenization_protocol import build_ppt_marker_id_candidates, find_ppt_marker_end

    tokenizer = load_tokenizer(tokenizer_path, True, True)
    patterns = build_ppt_marker_id_candidates(tokenizer)
    rng = random.Random(42)
    # Identical workloads and row counts isolate source filtering from workload selection.
    reservoir, counts = {}, {}
    for line in (rebuild / "raw/train_raw.json").open():
        row = json.loads(line)
        key = row["workload_key"]
        counts[key] = counts.get(key, 0) + 1
        if rng.randrange(counts[key]) == 0:
            reservoir[key] = row
    chosen = sorted(reservoir)
    random.Random(42).shuffle(chosen)
    chosen = set(chosen[:count])
    filtered = [reservoir[key] for key in sorted(chosen)]
    original, old_counts = {}, {}
    for line in (original_raw / "train_raw.json").open():
        row = json.loads(line)
        key = row.get("workload_key")
        if key not in chosen:
            continue
        old_counts[key] = old_counts.get(key, 0) + 1
        if rng.randrange(old_counts[key]) == 0:
            original[key] = row
    if set(original) != chosen:
        raise ValueError("Original and filtered workload coverage differs")

    def encode(rows):
        encoded = []
        for row in rows:
            ids = tokenizer(row["text"], add_special_tokens=False, truncation=False)["input_ids"]
            if len(ids) > 2048:
                raise ValueError("Controlled sample exceeds complete sequence budget")
            ppt_end = find_ppt_marker_end(ids, patterns)
            if ppt_end < 0:
                raise ValueError("Missing PPT boundary")
            encoded.append(
                {
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": list(ids),
                    "length": len(ids),
                    "original_length": len(ids),
                    "was_truncated": False,
                    "sequence_complete": True,
                    "ppt_boundary_version": "ppt_boundary_v2",
                    "ppt_end": ppt_end,
                }
            )
        return Dataset.from_list(encoded)

    # Evaluation loss uses the same small validation rows in both arms.
    validation = []
    for line in (rebuild / "raw/validation_raw.json").open():
        validation.append(json.loads(line))
        if len(validation) == 32:
            break
    for name, rows in (
        ("original", [original[key] for key in sorted(chosen)]),
        ("filtered", filtered),
    ):
        DatasetDict(train=encode(rows), validation=encode(validation)).save_to_disk(
            str(directory / name / "dataset")
        )
        atomic_json(
            directory / name / "selection.json",
            {
                "workload_keys": sorted(chosen),
                "rows": len(rows),
                "text_sha256": digest([row["text"] for row in rows]),
                "complete_tokens": sum(
                    len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
                    for row in rows
                ),
                "supervised_suffix_tokens": sum(
                    len(ids) - find_ppt_marker_end(ids, patterns) - 1
                    for ids in (
                        tokenizer(row["text"], add_special_tokens=False)["input_ids"]
                        for row in rows
                    )
                ),
            },
        )


def eval_workloads(source_run):
    from LLM.audit.audit_tvm_backend import register_tasks
    from tvm import auto_scheduler
    from tvm.auto_scheduler.measure import recover_measure_input

    source = json.loads((source_run / "manifest.json").read_text())
    audit = json.loads((Path(source["inputs"]["audit_dir"]) / "manifest.json").read_text())
    register_tasks(audit["registry"]["path"])
    selected = {}
    for line in (source_run / "samples.jsonl").open():
        sample = json.loads(line)
        if sample["split"] != "validation" or "median" not in sample["roles"]:
            continue
        category = sample["operator_family"]
        if category == "convolution":
            category += "_long" if sample["length"] > 1024 else "_short"
        if (
            category not in {"convolution_long", "matmul_dense", "batch_matmul", "softmax"}
            or category in selected
        ):
            continue
        normalized, ppt_index = normalize_record(sample["record"])
        normalized["i"][0][0] = json.dumps(normalized["i"][0][0])
        normalized["i"][1][1] = normalized["i"][1][1][: ppt_index + 1]
        inp, _ = auto_scheduler.measure_record.load_record_from_string(json.dumps(normalized))
        inp = recover_measure_input(inp, rebuild_state=True)
        selected[category] = (sample["workload_key"], [inp])
    return selected


def run(args):
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    module_dir = str(LLM_ROOT)
    os.environ["PYTHONPATH"] = project_pythonpath(os.environ.get("PYTHONPATH", ""))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    manifest = {
        "created_at": now(),
        "base_model": args.base_model,
        "tokenizer": args.tokenizer,
        "steps": args.steps,
        "training_rows_per_arm": 256,
        "micro_batch": 2,
        "gradient_accumulation": 4,
        "seed": 42,
        "evaluation_split": "validation",
        "holdout_used": False,
        "max_new_tokens_per_stage_cap": 256,
        "required_candidates": 1,
        "comparison_scope": "small_pilot_not_full_training_or_causal_performance_claim",
    }
    atomic_json(directory / "manifest.json", manifest)
    try:
        atomic_json(directory / "status.json", {"state": "RUNNING", "stage": "PREPARE"})
        prepare(directory, Path(args.rebuild_dir), Path(args.original_raw), args.tokenizer)
        selected = eval_workloads(Path(args.source_run))
        atomic_json(
            directory / "evaluation_selection.json",
            {name: value[0] for name, value in selected.items()},
        )
        metrics = {}

        def evaluate(name, checkpoint):
            import gc
            import torch
            from LLM.generation.gen_state import load_model, load_tokenizer
            from LLM.validation.eval_struct import evaluate_structural_validity

            atomic_json(directory / "status.json", {"state": "RUNNING", "stage": f"EVAL_{name}"})
            tokenizer = load_tokenizer(args.tokenizer, True, True)
            model = load_model(checkpoint, True, "cuda:0")
            model.eval()
            details = []
            result = evaluate_structural_validity(
                model,
                tokenizer,
                list(selected.values()),
                "cuda:0",
                max_states_per_workload=1,
                max_new_tokens=256,
                do_build=True,
                build_timeout=30,
                do_sample=False,
                seed=42,
                per_workload_reports=details,
            )
            metrics[name] = {"metrics": result, "workloads": details}
            atomic_json(directory / "evaluation.json", metrics)
            del model
            gc.collect()
            torch.cuda.empty_cache()

        evaluate("base", args.base_model)
        for name in ("original", "filtered"):
            atomic_json(directory / "status.json", {"state": "RUNNING", "stage": f"TRAIN_{name}"})
            command = [
                sys.executable,
                str(Path(module_dir) / "train/train_qwen3_clm.py"),
                "--model_name_or_path",
                args.base_model,
                "--tokenizer_name",
                args.tokenizer,
                "--dataset_name",
                str(directory / name / "dataset"),
                "--output_dir",
                str(directory / name / "model"),
                "--do_train",
                "--max_steps",
                str(args.steps),
                "--per_device_train_batch_size",
                "2",
                "--gradient_accumulation_steps",
                "4",
                "--learning_rate",
                "5e-6",
                "--lr_scheduler_type",
                "constant",
                "--bf16",
                "True",
                "--torch_dtype",
                "bfloat16",
                "--gradient_checkpointing",
                "True",
                "--logging_steps",
                "4",
                "--save_strategy",
                "no",
                "--eval_strategy",
                "no",
                "--report_to",
                "none",
                "--seed",
                "42",
                "--dataloader_num_workers",
                "0",
                "--mask_prefix_before_ppt",
                "True",
                "--remove_unused_columns",
                "False",
            ]
            atomic_json(directory / name / "command.json", {"argv": command})
            with (directory / name / "train.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            evaluate(name, str(directory / name / "model"))
        atomic_json(
            directory / "status.json",
            {
                "state": "SUCCEEDED",
                "stage": "COMPLETE",
                "finished_at": now(),
                "elapsed_seconds": time.monotonic() - started,
                "exit_code": 0,
                "model_generation_quality_evaluated": True,
                "runtime_correctness_checked": False,
                "latency_measured": False,
            },
        )
    except BaseException as error:
        atomic_json(
            directory / "status.json",
            {
                "state": "FAILED",
                "error_message": str(error),
                "finished_at": now(),
                "exit_code": 1,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rebuild-dir", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument(
        "--original-raw", default="/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext"
    )
    parser.add_argument(
        "--base-model",
        default="/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2",
    )
    parser.add_argument(
        "--tokenizer", default="/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2"
    )
    parser.add_argument("--steps", type=int, default=32)
    run(parser.parse_args())
