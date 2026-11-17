#!/usr/bin/env python3
"""Trace model decisions against the actual CUDA parser in an isolated diagnostic run."""

from LLM.core.layout import code_dependencies, source_path


import argparse
import gc
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path

from LLM.core.io import atomic_json, file_sha256
from LLM.core.schedule_record_utils import digest, normalize_record


def split_expectations(steps, marker):
    """Mirror the fixed SPC fields checked by CUDA InitFillTileSize/InitThreadBind."""
    last_prompt = max((i for i, step in enumerate(steps) if step[0] == "PPT"), default=-1)
    selected = steps[:last_prompt] if marker == "PPT" else steps[last_prompt + 1 :]
    return [
        {
            "stage": step[1],
            "iter": step[2],
            "extent": step[3] or 0,
            "factors": len(step[4]),
            "inner_to_outer": step[5],
        }
        for step in selected
        if step[0] == "SP"
    ]


def inspect_split_output(tokens, expected):
    """Explain the first lexical failure without changing what the TVM parser receives."""
    cursor = 0
    for decision, spec in enumerate(expected):
        fields = [
            ("opcode", "SPC"),
            ("stage", spec["stage"]),
            ("iter", spec["iter"]),
            ("extent", spec["extent"]),
        ]
        fields += [(f"factor_{i}", None) for i in range(spec["factors"])]
        fields.append(("inner_to_outer", spec["inner_to_outer"]))
        for field, required in fields:
            if cursor >= len(tokens):
                return {
                    "valid": False,
                    "kind": "incomplete",
                    "decision": decision,
                    "field": field,
                    "field_offset": cursor,
                    "expected": required,
                }
            actual = tokens[cursor]
            try:
                value = actual if field == "opcode" else int(actual)
            except ValueError:
                return {
                    "valid": False,
                    "kind": "not_integer",
                    "decision": decision,
                    "field": field,
                    "field_offset": cursor,
                    "actual": actual,
                }
            if required is not None and value != required:
                return {
                    "valid": False,
                    "kind": "fixed_field_mismatch",
                    "decision": decision,
                    "field": field,
                    "field_offset": cursor,
                    "actual": actual,
                    "expected": required,
                }
            cursor += 1
    return {"valid": True, "required_fields": cursor, "trailing_fields": len(tokens) - cursor}


class TracedModel:
    """Delegate generation unchanged, preserving exact prompt and raw continuation IDs."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        import torch

        started = time.monotonic()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        for prompt, mask, response in zip(
            input_ids.tolist(), attention_mask.tolist(), output.tolist()
        ):
            continuation = response[len(prompt) :]
            self.calls.append(
                {
                    "prompt_ids": prompt,
                    "attention_mask": mask,
                    "prompt": self.tokenizer.decode(prompt, skip_special_tokens=False),
                    "continuation_ids": continuation,
                    "raw_continuation": self.tokenizer.decode(
                        continuation, skip_special_tokens=False
                    ),
                    "budget": kwargs["max_new_tokens"],
                    "tokens": len(continuation),
                    "eos_seen": self.tokenizer.eos_token_id in continuation,
                    "budget_exhausted": len(continuation) >= kwargs["max_new_tokens"],
                    "generation_seconds": time.monotonic() - started,
                }
            )
        return output


def make_sketch(sample):
    from tvm import auto_scheduler
    from tvm.auto_scheduler.measure import recover_measure_input

    record, ppt = normalize_record(sample["record"])
    record["i"][0][0] = json.dumps(record["i"][0][0])
    record["i"][1][1] = record["i"][1][1][: ppt + 1]
    inp, _ = auto_scheduler.measure_record.load_record_from_string(json.dumps(record))
    return recover_measure_input(inp, rebuild_state=True)


def trace_case(model, tokenizer, sample, cap, directory, replay=False):
    from tvm import auto_scheduler
    from LLM.validation.eval_struct import _generate_decisions_for_states
    from LLM.generation.generation_protocol import isolated_generation_rng, resolve_max_new_tokens
    from LLM.data.make_dataset import input_to_tokens

    sketch = make_sketch(sample)
    proxy = TracedModel(model, tokenizer)
    callbacks = []
    started = time.monotonic()

    def callback(task, states, request):
        prompts = input_to_tokens(task, states, sketch)
        start = len(proxy.calls)
        budget = resolve_max_new_tokens(request)
        if cap:
            budget = min(budget, cap)
        if replay:
            outputs = []
            for prompt in prompts:
                assert sample["row"]["text"].startswith(prompt)
                outputs.append(sample["row"]["text"][len(prompt) :].split())
        else:
            outputs = _generate_decisions_for_states(
                task,
                states,
                sketch,
                tokenizer,
                proxy,
                "cuda:0",
                budget,
                False,
                0.95,
                50,
                0.7,
                1,
                False,
            )
        for index, (state, prompt, tokens) in enumerate(zip(states, prompts, outputs)):
            serialized = json.loads(auto_scheduler.MeasureInput(task, state).to_json())
            steps = serialized[1][1]
            marker = next((s[0] for s in reversed(steps) if s[0] in {"PPT", "TBS", "PRS"}), None)
            expected = split_expectations(steps, marker) if marker in {"PPT", "TBS"} else []
            report = {
                "stage": marker,
                "requested_fields": int(request),
                "budget": budget,
                "state_steps": steps,
                "prompt": prompt,
                "output_fields": tokens,
                "expected_splits": expected,
                "lexical_check": inspect_split_output(tokens, expected) if expected else None,
            }
            if not replay:
                report["generation"] = proxy.calls[start + index]
            callbacks.append(report)
        atomic_json(directory / "active_callbacks.json", callbacks)
        return outputs

    result = {
        "sample_id": sample["sample_id"],
        "workload_key": sample["workload_key"],
        "cap": cap,
        "replay": replay,
        "callbacks": callbacks,
    }
    try:
        with isolated_generation_rng(42):
            states = auto_scheduler.SketchPolicy(sketch.task, seed=42, verbose=0).gen_states(
                [sketch.state], callback
            )
        result["parsed"] = len(states)
        result["built"] = 0
        if states:
            import tempfile

            candidates = [auto_scheduler.MeasureInput(sketch.task, state) for state in states]
            with tempfile.TemporaryDirectory(prefix="generation-diagnostic-") as temporary:
                previous = tempfile.tempdir
                try:
                    tempfile.tempdir = temporary
                    builds = auto_scheduler.LocalBuilder(timeout=60, n_parallel=1).build(
                        candidates, verbose=0
                    )
                    result["build_results"] = [
                        {
                            "error_no": int(r.error_no),
                            "error": str(r.error_msg),
                            "module_bytes": (
                                Path(r.filename).stat().st_size
                                if r.filename and Path(r.filename).is_file()
                                else 0
                            ),
                        }
                        for r in builds
                    ]
                    result["built"] = sum(
                        r["error_no"] == 0 and r["module_bytes"] > 0
                        for r in result["build_results"]
                    )
                finally:
                    tempfile.tempdir = previous
    except Exception:
        result["error"] = traceback.format_exc()
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def run(args):
    import torch
    from LLM.audit.audit_tvm_backend import compiler_fingerprint, register_tasks
    from LLM.generation.gen_state import load_model, load_tokenizer

    torch.set_num_threads(4)
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prior = Path(args.training_run).resolve()
    source = Path(args.source_run).resolve()
    base_manifest = json.loads((source / "manifest.json").read_text())
    if file_sha256(source / "samples.jsonl") != base_manifest["samples_sha256"]:
        raise ValueError("Source sample identity changed")
    selection = json.loads((prior / "evaluation_selection.json").read_text())
    samples = {}
    for line in (source / "samples.jsonl").open():
        sample = json.loads(line)
        if (
            sample["split"] == "validation"
            and "median" in sample["roles"]
            and sample["workload_key"] in selection.values()
        ):
            family = next(f for f, key in selection.items() if key == sample["workload_key"])
            samples[family] = sample
    assert set(samples) == set(selection)
    audit = json.loads((Path(base_manifest["inputs"]["audit_dir"]) / "manifest.json").read_text())
    register_tasks(audit["registry"]["path"])
    model_paths = {
        "base": args.base_model,
        "original": str(prior / "original/model"),
        "filtered": str(prior / "filtered/model"),
    }
    code_paths = [Path(__file__).resolve()]
    code_paths += [
        source_path(n)
        for n in [
            "generation_protocol.py",
            "eval_struct.py",
            "gen_state.py",
            "tokenization_protocol.py",
            "make_dataset.py",
            "schedule_record_utils.py",
            "audit_tvm_backend.py",
        ]
    ]
    code_paths = code_dependencies(code_paths)
    code = {str(p.resolve()): file_sha256(p) for p in code_paths}
    frozen = directory / "frozen_code"
    frozen.mkdir()
    for path in code_paths:
        shutil.copy2(path, frozen / path.name)
    inputs = {}
    for model in model_paths.values():
        for path in Path(model).glob("*"):
            if path.suffix in {".safetensors", ".json"}:
                inputs[str(path)] = file_sha256(path)
    manifest = {
        "created_at": time.time(),
        "code": code,
        "model_files": inputs,
        "models": model_paths,
        "tokenizer": args.base_model,
        "caps": args.caps,
        "source_manifest_id": base_manifest["manifest_id"],
        "samples": {f: s["sample_id"] for f, s in samples.items()},
        "compiler": compiler_fingerprint(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv"], text=True
        ),
        "fallback_enabled": False,
        "seed": 42,
        "holdout_used": False,
    }
    manifest["manifest_id"] = digest(manifest)
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(directory / "samples.json", samples)
    tokenizer = load_tokenizer(args.base_model, True, True)
    results = []
    try:
        for family, sample in samples.items():
            atomic_json(directory / "status.json", {"state": "RUNNING", "case": f"replay/{family}"})
            result = trace_case(None, tokenizer, sample, 0, directory, replay=True)
            result.update(arm="replay", family=family)
            results.append(result)
            atomic_json(directory / "results.json", results)
        for arm, checkpoint in model_paths.items():
            model = load_model(checkpoint, True, "cuda:0")
            model.eval()
            model.config.use_cache = True
            for cap in args.caps:
                for family, sample in samples.items():
                    case = f"{arm}/{cap or 'production'}/{family}"
                    atomic_json(
                        directory / "status.json",
                        {"state": "RUNNING", "case": case, "completed": len(results)},
                    )
                    result = trace_case(model, tokenizer, sample, cap, directory)
                    result.update(arm=arm, family=family)
                    results.append(result)
                    atomic_json(directory / "results.json", results)
                    print(
                        f"{case}: parsed={result.get('parsed')} built={result.get('built')} stages={[r['stage'] for r in result['callbacks']]}",
                        flush=True,
                    )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for path, expected in code.items():
            assert file_sha256(path) == expected, path
        for path, expected in inputs.items():
            assert file_sha256(path) == expected, path
        atomic_json(
            directory / "status.json",
            {
                "state": "SUCCEEDED",
                "cases": len(results),
                "elapsed_seconds": time.monotonic() - started,
                "errors": sum("error" in r for r in results),
                "finished_at": time.time(),
            },
        )
    except BaseException:
        atomic_json(directory / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--training-run", default="LLM/audit_runs/4090_plan6_controlled_20261109_02"
    )
    parser.add_argument(
        "--source-run", default="LLM/audit_runs/4090_plan5_roundtrip_full_20261102_01"
    )
    parser.add_argument(
        "--base-model",
        default="/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2",
    )
    parser.add_argument(
        "--caps", nargs="+", type=int, default=[256, 0], help="0 uses production per-stage budgets"
    )
    run(parser.parse_args())
