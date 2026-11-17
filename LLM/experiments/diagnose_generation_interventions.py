#!/usr/bin/env python3
"""Use explicit diagnostic interventions to test observed parser failure mechanisms."""

from LLM.core.layout import code_dependencies, source_path


import argparse
import copy
import gc
import json
import shutil
import time
import traceback
from pathlib import Path

from LLM.experiments.diagnose_model_generation import (
    TracedModel,
    inspect_split_output,
    make_sketch,
    split_expectations,
)
from LLM.core.io import atomic_json, file_sha256
from LLM.core.schedule_record_utils import digest


def replace_split_headers(tokens, expected):
    """Keep factors, replacing only deterministic fixed fields in existing SPC slots."""
    tokens = list(tokens)
    cursor, changes = 0, []
    for spec in expected:
        if cursor + spec["factors"] + 5 > len(tokens) or tokens[cursor] != "SPC":
            raise ValueError("Intervention requires complete SPC slots in the same order")
        fields = {
            1: spec["stage"],
            2: spec["iter"],
            3: spec["extent"],
            4 + spec["factors"]: spec["inner_to_outer"],
        }
        for offset, value in fields.items():
            index = cursor + offset
            if tokens[index] != str(value):
                changes.append({"index": index, "from": tokens[index], "to": str(value)})
                tokens[index] = str(value)
        cursor += spec["factors"] + 5
    return tokens, changes


def raise_thread_factor(tokens, expected, family):
    """A single manually specified factor edit; this is not a production repair policy."""
    target = (2, 5) if family == "matmul_dense" else (6, 15)
    tokens = list(tokens)
    cursor = 0
    for spec in expected:
        if (spec["stage"], spec["iter"]) == target:
            index = cursor + 5  # second split factor controls the thread tile in this sketch
            before = tokens[index]
            tokens[index] = "32"
            return tokens, [{"index": index, "from": before, "to": "32"}]
        cursor += spec["factors"] + 5
    raise ValueError("Expected manual intervention split was not present")


def initial_geometry(sample, trace):
    """Recover actual root iterator extents from the model's first-stage factors."""
    from tvm import auto_scheduler
    from tvm.auto_scheduler.measure import recover_measure_input

    record = copy.deepcopy(sample["record"])
    steps = record["i"][1][1]
    ppt = next(i for i, step in enumerate(steps) if step[0] == "PPT")
    steps = steps[: ppt + 1]
    tokens, cursor = trace["callbacks"][0]["output_fields"], 0
    for step in steps:
        if step[0] == "SP":
            n = len(step[4])
            step[4] = list(map(int, tokens[cursor + 4 : cursor + 4 + n]))
            cursor += n + 5
    record["i"][1][1] = steps
    if not isinstance(record["i"][0][0], str):
        record["i"][0][0] = json.dumps(record["i"][0][0])
    inp, _ = auto_scheduler.measure_record.load_record_from_string(json.dumps(record))
    inp = recover_measure_input(inp, rebuild_state=True)
    state = inp.task.compute_dag.infer_bound_from_state(inp.state)
    roots = []
    for stage_id, stage in enumerate(state.stages):
        if int(stage.compute_at) == 0 and int(stage.op_type) != 0:
            roots.append(
                {
                    "stage": stage_id,
                    "name": str(stage.op.name),
                    "iters": [
                        {
                            "name": str(it.name),
                            "extent": str(it.range.extent),
                            "annotation": int(it.annotation),
                            "kind": int(it.iter_kind),
                        }
                        for it in stage.iters
                    ],
                }
            )
    return {
        "roots": roots,
        "warp_size": int(inp.task.hardware_params.warp_size),
        "max_vthread_extent": int(inp.task.hardware_params.max_vthread_extent),
    }


def run_case(model, tokenizer, sample, trace, intervention, directory):
    from tvm import auto_scheduler
    from LLM.validation.eval_struct import _generate_decisions_for_states
    from LLM.generation.generation_protocol import isolated_generation_rng, resolve_max_new_tokens
    from LLM.data.make_dataset import input_to_tokens
    from LLM.validation.generation_roundtrip_worker import build_candidate

    sketch = make_sketch(sample)
    proxy, callbacks = TracedModel(model, tokenizer), []

    def callback(task, states, request):
        prompts = input_to_tokens(task, states, sketch)
        outputs = []
        for state, prompt in zip(states, prompts):
            steps = json.loads(auto_scheduler.MeasureInput(task, state).to_json())[1][1]
            marker = next(s[0] for s in reversed(steps) if s[0] in {"PPT", "TBS", "PRS"})
            cached = next((c for c in trace["callbacks"] if c["prompt"] == prompt), None)
            if cached:
                tokens = list(cached["output_fields"])
                provenance = "cached_model_output_same_prompt"
            else:
                tokens = _generate_decisions_for_states(
                    task,
                    [state],
                    sketch,
                    tokenizer,
                    proxy,
                    "cuda:0",
                    resolve_max_new_tokens(request),
                    False,
                    0.95,
                    50,
                    0.7,
                    1,
                    False,
                )[0]
                provenance = "fresh_model_output_changed_prompt"
            expected = split_expectations(steps, marker) if marker in {"PPT", "TBS"} else []
            before, changes = list(tokens), []
            if intervention in {"fixed_headers", "headers_and_prs_oracle"} and marker == "TBS":
                tokens, changes = replace_split_headers(tokens, expected)
            if (
                intervention
                in {"thread_factor", "thread_factor_and_headers", "thread_headers_prs_oracle"}
                and marker == "PPT"
            ):
                tokens, changes = raise_thread_factor(tokens, expected, trace["family"])
            if (
                intervention in {"thread_factor_and_headers", "thread_headers_prs_oracle"}
                and marker == "TBS"
            ):
                tokens, changes = replace_split_headers(tokens, expected)
            if (
                intervention in {"headers_and_prs_oracle", "thread_headers_prs_oracle"}
                and marker == "PRS"
            ):
                # Explicit oracle control. Stage IDs/iter 0 are from this actual prompt;
                # zero is a valid unroll choice. Never count this as model performance.
                prompt_fields = prompt.split()
                stage_ids = list(map(int, prompt_fields[prompt_fields.index("PRS") + 1 :]))
                tokens = [
                    field
                    for stage in stage_ids
                    for field in ["PR", str(stage), "0", "auto_unroll_max_step$0"]
                ]
                changes.append({"kind": "oracle_unroll", "stages": stage_ids, "config": 0})
            callbacks.append(
                {
                    "stage": marker,
                    "prompt": prompt,
                    "before": before,
                    "after": tokens,
                    "provenance": provenance,
                    "manual_changes": changes,
                    "lexical_check": inspect_split_output(tokens, expected) if expected else None,
                }
            )
            atomic_json(directory / "active_callbacks.json", callbacks)
            outputs.append(tokens)
        return outputs

    started = time.monotonic()
    result = {
        "arm": trace["arm"],
        "family": trace["family"],
        "intervention": intervention,
        "sample_id": sample["sample_id"],
        "callbacks": callbacks,
        "unmodified_model_success": False,
        "latency_measured": False,
    }
    try:
        with isolated_generation_rng(42):
            states = auto_scheduler.SketchPolicy(sketch.task, seed=42, verbose=0).gen_states(
                [sketch.state], callback
            )
        result["parsed"] = len(states)
        result["builds"] = [
            build_candidate(
                auto_scheduler.MeasureInput(sketch.task, state).serialize(), sample["sample_id"]
            )
            for state in states
        ]
        result["built"] = sum(r["status"] == "PASSED" for r in result["builds"])
    except Exception:
        result["error"] = traceback.format_exc()
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def run(args):
    import torch
    from LLM.audit.audit_tvm_backend import register_tasks, compiler_fingerprint
    from LLM.generation.gen_state import load_model, load_tokenizer

    torch.set_num_threads(4)
    source = Path(args.source_run).resolve()
    directory = Path(args.run_dir).resolve()
    directory.mkdir(exist_ok=False, parents=True)
    traces = json.loads((source / "results.json").read_text())
    samples = json.loads((source / "samples.json").read_text())
    prior_manifest = json.loads((source / "manifest.json").read_text())
    registry_manifest = json.loads(
        Path("LLM/audit_runs/4090_plan5_full_fast_20261022_01/manifest.json").read_text()
    )
    register_tasks(registry_manifest["registry"]["path"])
    code = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for name in [
        "diagnose_model_generation.py",
        "generation_protocol.py",
        "eval_struct.py",
        "gen_state.py",
        "generation_roundtrip_worker.py",
    ]:
        path = source_path(name).resolve()
        code[str(path)] = file_sha256(path)
    code = {str(path): file_sha256(path) for path in code_dependencies(code)}
    frozen = directory / "frozen_code"
    frozen.mkdir()
    for name in code:
        shutil.copy2(name, frozen / Path(name).name)
    manifest = {
        "created_at": time.time(),
        "source_run": str(source),
        "source_results_sha256": file_sha256(source / "results.json"),
        "code": code,
        "compiler": compiler_fingerprint(),
        "manual_interventions_only": True,
        "formal_model_quality_evaluation": False,
    }
    manifest["manifest_id"] = digest(manifest)
    atomic_json(directory / "manifest.json", manifest)
    geometry = []
    for trace in traces:
        if trace["arm"] == "replay" or trace["cap"] != 256:
            continue
        geometry.append(
            {
                "arm": trace["arm"],
                "family": trace["family"],
                **initial_geometry(samples[trace["family"]], trace),
            }
        )
    atomic_json(directory / "geometry.json", geometry)
    tokenizer = load_tokenizer(prior_manifest["tokenizer"], True, True)
    cases = [
        ("base", "convolution_long", "fixed_headers"),
        ("original", "matmul_dense", "thread_factor"),
        ("filtered", "matmul_dense", "thread_factor"),
        ("original", "convolution_long", "thread_factor"),
        ("filtered", "convolution_long", "thread_factor"),
    ]
    if args.conv_oracle_controls:
        cases = [
            ("base", "convolution_long", "headers_and_prs_oracle"),
            ("original", "convolution_long", "thread_headers_prs_oracle"),
            ("filtered", "convolution_long", "thread_factor_and_headers"),
            ("filtered", "convolution_long", "thread_headers_prs_oracle"),
        ]
    results = []
    try:
        for arm in ["base", "original", "filtered"]:
            model = load_model(prior_manifest["models"][arm], True, "cuda:0")
            model.eval()
            model.config.use_cache = True
            for current_arm, family, intervention in cases:
                if current_arm != arm:
                    continue
                atomic_json(
                    directory / "status.json",
                    {"state": "RUNNING", "case": f"{arm}/{family}/{intervention}"},
                )
                trace = next(
                    t
                    for t in traces
                    if t["arm"] == arm and t["family"] == family and t["cap"] == 256
                )
                result = run_case(model, tokenizer, samples[family], trace, intervention, directory)
                results.append(result)
                atomic_json(directory / "results.json", results)
                print(
                    arm,
                    family,
                    intervention,
                    "parsed",
                    result.get("parsed"),
                    "built",
                    result.get("built"),
                    flush=True,
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for path, expected in code.items():
            assert file_sha256(path) == expected
        atomic_json(
            directory / "status.json",
            {
                "state": "SUCCEEDED",
                "cases": len(results),
                "errors": sum("error" in r for r in results),
                "finished_at": time.time(),
            },
        )
    except BaseException:
        atomic_json(directory / "status.json", {"state": "FAILED", "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run", default="LLM/audit_runs/4090_plan6_generation_diagnosis_20261109_01"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--conv-oracle-controls", action="store_true")
    run(parser.parse_args())
