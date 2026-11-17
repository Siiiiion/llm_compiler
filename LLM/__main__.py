"""Discover and run LLM pipeline commands without importing model dependencies."""

import argparse
import os
import runpy
import sys

from pathlib import Path

# Also support python /path/to/LLM/__main__.py from another working directory.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LLM.core.layout import project_pythonpath

COMMANDS = {
    "dataset": ("LLM.data.make_dataset", "Build raw schedule text and tokenized datasets"),
    "rebuild-dataset": ("LLM.data.rebuild_clm_dataset", "Encode complete CLM sequences"),
    "audit": ("LLM.audit.audit_schedule_dataset", "Audit original and normalized TVM schedules"),
    "prepare-data": ("LLM.data.prepare_audited_clm", "Export raw data with audit provenance"),
    "rebuild-audited": (
        "LLM.data.rebuild_audited_clm",
        "Export, encode, verify and publish audited data",
    ),
    "verify-data": ("LLM.data.verify_audited_clm", "Verify rebuilt data and audit provenance"),
    "extend-tokenizer": ("LLM.data.extend_tokenizer", "Extend schedule vocabulary and embeddings"),
    "generate": ("LLM.generation.gen_state", "Generate schedule candidates"),
    "schedule": (
        "LLM.generation.task_scheduler",
        "Select workloads for the next measurement budget",
    ),
    "measure": ("LLM.benchmark.measure_programs", "Measure schedule candidates"),
    "split-records": ("LLM.data.split_measure_records", "Split measurement records by workload"),
    "eval-struct": (
        "LLM.validation.eval_struct",
        "Evaluate generated state parsing and compilation",
    ),
    "validate-roundtrip": (
        "LLM.validation.validate_generation_roundtrip",
        "Replay audited sequences through the parser",
    ),
    "validate-constrained": (
        "LLM.validation.validate_constrained_generation",
        "Validate constrained generation",
    ),
    "validate-candidates": (
        "LLM.validation.validate_multicandidate_generation",
        "Validate a fixed candidate budget",
    ),
    "validate-sketches": (
        "LLM.validation.validate_multisketch_generation",
        "Validate multiple sketches per workload",
    ),
    "benchmark": ("LLM.benchmark.benchmark_relay", "Check and benchmark a Relay model"),
    "benchmark-batch": (
        "LLM.benchmark.benchmark_relay_batch",
        "Benchmark a suite of Relay artifacts",
    ),
    "analyze-history": (
        "LLM.benchmark.analyze_missing_history",
        "Analyze missing schedule histories",
    ),
    "classify-speedup": (
        "LLM.benchmark.classify_network_speedup",
        "Classify network performance results",
    ),
    "train-clm": ("LLM.train.train_qwen3_clm", "Train a causal schedule model"),
    "build-pairs": ("LLM.train.build_preference_pairs", "Build measured-latency preference pairs"),
    "train-dpo": ("LLM.train.train_qwen3_dpo", "Train SFT, DPO or LAPO"),
    "baseline": ("LLM.benchmark.baseline", "Tune and benchmark an Ansor baseline"),
    "tune": ("LLM.benchmark.tune_relay", "Tune a Relay model"),
    "diagnose-generation": ("LLM.experiments.diagnose_model_generation", "Inspect model decisions"),
    "diagnose-roundtrip": (
        "LLM.experiments.diagnose_generation_roundtrip",
        "Diagnose state replay",
    ),
    "diagnose-interventions": (
        "LLM.experiments.diagnose_generation_interventions",
        "Compare generation interventions",
    ),
    "measure-representatives": (
        "LLM.experiments.measure_plan6_representatives",
        "Measure representative experiment states",
    ),
    "controlled-experiment": (
        "LLM.experiments.run_plan6_controlled_experiment",
        "Run the historical controlled pilot",
    ),
    "plot-example": ("LLM.experiments.evaluate", "Plot illustrative historical latency values"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m LLM",
        description="LLM-guided TVM scheduling. Use <command> --help for command options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name:23} {description}" for name, (_, description) in COMMANDS.items()
        ),
    )
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return
    command = argv.pop(0)
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    target = COMMANDS[command][0]
    os.environ["PYTHONPATH"] = project_pythonpath(os.environ.get("PYTHONPATH", ""))
    previous_argv = sys.argv
    try:
        sys.argv = [target, *argv]
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
