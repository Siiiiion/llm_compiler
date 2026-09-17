#!/usr/bin/env python3
"""Run equal-budget Pretrain/LAPO generation and optional measurement."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR = SCRIPT_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and measure schedules from two checkpoints using one sketch set."
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--pretrain-model", required=True)
    parser.add_argument("--lapo-model", required=True)
    parser.add_argument("--python", default="/home/qsy/anaconda3/envs/tlm/bin/python")
    parser.add_argument("--stage", choices=["generate", "measure", "all"], default="all")
    parser.add_argument("--cuda-visible", default="0")
    parser.add_argument("--keep-cnt", type=int, default=64)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command, env, dry_run):
    print("$", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env, cwd=LLM_DIR)


def generation_command(args, metadata, model_path, output_path):
    paths = metadata["paths"]
    stats_path = str(Path(output_path).with_suffix(".stats.json"))
    return [
        args.python,
        str(LLM_DIR / "gen_state.py"),
        f"--target={metadata['target']}",
        f"--model_name_or_path={model_path}",
        f"--sketch_path={paths['sketch']}",
        f"--save_path={output_path}",
        "--allow_repeat=True",
        f"--keep_cnt={args.keep_cnt}",
        "--is_build=True",
        "--fallback_to_sketch_when_invalid=False",
        f"--max_retries={args.max_retries}",
        "--require_full_keep_cnt=True",
        "--do_sample=True",
        "--disable_eos_stop=False",
        "--sample_top_k=0",
        "--sample_top_p=1.0",
        "--sample_temperature=0.6",
        f"--generation_batch_size={args.generation_batch_size}",
        f"--generation_seed={args.generation_seed}",
        f"--network_info_dir={paths['network_info_dir']}",
        f"--to_measure_program_dir={paths['to_measure_program_dir']}",
        f"--stats_path={stats_path}",
    ]


def measurement_command(args, metadata, candidate_path, measured_path):
    paths = metadata["paths"]
    return [
        args.python,
        str(LLM_DIR / "measure_programs.py"),
        f"--target={metadata['target']}",
        f"--network-info-dir={paths['network_info_dir']}",
        f"--to-measure-path={candidate_path}",
        f"--measured-path={measured_path}",
        "--batch-size=64",
        f"--run-timeout={args.run_timeout}",
        "--no-resume",
        "--no-drop-hold-out",
        "--print-error-details",
    ]


def main():
    args = parse_args()
    if args.keep_cnt <= 0:
        raise ValueError("--keep-cnt must be positive")
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    metadata_path = experiment_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    if metadata.get("format") != "llm_compiler.attention_experiment":
        raise ValueError(f"Unsupported experiment metadata: {metadata_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible
    variants = {
        "pretrain": Path(args.pretrain_model).expanduser().resolve(),
        "lapo": Path(args.lapo_model).expanduser().resolve(),
    }
    outputs = {}
    for label, model_path in variants.items():
        if not model_path.is_dir():
            raise FileNotFoundError(f"{label} model directory not found: {model_path}")
        candidate_path = experiment_dir / f"{label}_candidates.json"
        measured_path = experiment_dir / f"{label}_measured.json"
        outputs[label] = (model_path, candidate_path, measured_path)

    if args.stage in {"generate", "all"}:
        for label, (model_path, candidate_path, _) in outputs.items():
            run(
                generation_command(args, metadata, str(model_path), str(candidate_path)),
                env,
                args.dry_run,
            )
    if args.stage in {"measure", "all"}:
        for _, candidate_path, measured_path in outputs.values():
            if not candidate_path.is_file() and not args.dry_run:
                raise FileNotFoundError(f"Missing generated candidates: {candidate_path}")
            run(
                measurement_command(args, metadata, str(candidate_path), str(measured_path)),
                env,
                args.dry_run,
            )


if __name__ == "__main__":
    main()
