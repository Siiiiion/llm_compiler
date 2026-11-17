"""Replay known schedule suffixes through the production generation callback and TVM parser."""

import copy
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

from LLM.core.io import atomic_json
from LLM.core.schedule_record_utils import digest, normalize_record

TOKENIZER = None
TEMP_ROOT = None


def initialize(registry, tokenizer_path, temp_root):
    global TOKENIZER, TEMP_ROOT
    from LLM.audit.audit_tvm_backend import register_tasks
    from LLM.generation.gen_state import load_tokenizer

    register_tasks(registry)
    TOKENIZER = load_tokenizer(tokenizer_path, True, True)
    TOKENIZER.padding_side = "left"
    TEMP_ROOT = temp_root


def replay_suffix(tokenizer, text, prompt, full_ids):
    """Use the exact prompt encoder and suffix decoder used by gen_state.gen_func."""
    from LLM.generation.gen_state import decode_decision_tokens
    from LLM.core.tokenization_protocol import encode_generation_prompts, encode_prompt_suffix

    if not text.startswith(prompt):
        raise ValueError("Generation prompt is not an exact prefix of the audited training text")
    suffix = text[len(prompt) :]
    prompt_ids, combined_ids, _ = encode_prompt_suffix(tokenizer, prompt, suffix)
    actual_prompt = encode_generation_prompts(tokenizer, [prompt])["input_ids"][0]
    if actual_prompt != prompt_ids or combined_ids != full_ids:
        raise ValueError("Training tokens and the production generation condition differ")
    tokens = decode_decision_tokens(tokenizer, combined_ids[len(prompt_ids) :])
    if tokens != suffix.split():
        raise ValueError("Production suffix decoding changed the parser tokens")
    return tokens, len(prompt_ids)


def verify_full_text(tokenizer, ids, text):
    # PPT is a deliberate special token in the fixed tokenizer. Generation only
    # decodes a continuation, whereas this check decodes the complete prompt too.
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != text:
        raise ValueError("Full text is not preserved by token encoding/decoding")


class StageProgress:
    """Keep the last entered stage on disk even when the worker is killed."""

    def __init__(self, report, path=None):
        self.report = report
        self.path = Path(path) if path else None
        self.started = self.previous = time.monotonic()
        self.timings = []

    def enter(self, stage, **details):
        current = time.monotonic()
        if self.timings:
            self.timings[-1]["seconds"] = current - self.previous
        self.previous = current
        self.report["stage"] = stage
        self.timings.append({"stage": stage, "elapsed_seconds": current - self.started, **details})
        self.report["stage_timings"] = self.timings
        if self.path:
            atomic_json(
                self.path,
                {
                    "sample_id": self.report["sample_id"],
                    "pid": os.getpid(),
                    "updated_at_unix": time.time(),
                    "stage": stage,
                    "elapsed_seconds": current - self.started,
                    "stage_timings": self.timings,
                },
            )


def build_candidate(serialized, sample_id, progress_path=None):
    """Build/export with its own worker budget after reconstruction has finished."""
    from tvm.auto_scheduler.measure import _local_build_worker
    from tvm.contrib import tar

    report = {"sample_id": sample_id, "status": "FAILED", "stage": "build_export"}
    progress = StageProgress(report, progress_path)
    progress.enter("build_export")
    with tempfile.TemporaryDirectory(prefix="roundtrip-", dir=TEMP_ROOT) as directory:
        previous, old_temp = os.environ.get("TMPDIR"), tempfile.tempdir
        try:
            os.environ["TMPDIR"], tempfile.tempdir = directory, directory
            filename, _, error_no, message, duration = _local_build_worker(serialized, tar.tar, 0)
            report.update(build_error_no=int(error_no), build_seconds=float(duration))
            if int(error_no):
                raise ValueError(str(message)[-12000:])
            exported = Path(filename) if filename else None
            if exported is None or not exported.is_file() or exported.stat().st_size == 0:
                raise ValueError("Builder returned success without an exported module")
            report["exported_module_bytes"] = exported.stat().st_size
            report["status"] = "PASSED"
            progress.enter("complete")
        except Exception:
            report["error_message"] = traceback.format_exc()[-16000:]
        finally:
            tempfile.tempdir = old_temp
            if previous is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous
    return report


def check_sample(sample, progress_path=None, defer_build=False):
    import tvm
    from tvm import auto_scheduler
    from tvm.auto_scheduler.measure import _local_build_worker, recover_measure_input
    from tvm.contrib import tar

    from LLM.generation.gen_state import decode_decision_tokens
    from LLM.data.make_dataset import input_to_tokens
    from LLM.core.tokenization_protocol import build_ppt_marker_id_candidates, find_ppt_marker_end

    started = time.monotonic()
    report = {
        key: sample[key]
        for key in (
            "sample_id",
            "split",
            "row_index",
            "workload_key",
            "source_file",
            "source_input_hash",
            "length",
            "length_bucket",
            "operator_family",
            "roles",
        )
    }
    report.update(status="FAILED", stage="formatting", callbacks=[])
    progress = StageProgress(report, progress_path)
    try:
        progress.enter("normalize_record")
        row, record = sample["row"], sample["record"]
        normalized, ppt_index = normalize_record(record)
        if ppt_index is None:
            raise ValueError("Source schedule has no PPT")
        normalized["i"][0][0] = json.dumps(normalized["i"][0][0])
        progress.enter("restore_reference_state")
        expected, _ = auto_scheduler.measure_record.load_record_from_string(json.dumps(normalized))
        expected = recover_measure_input(expected, rebuild_state=True)
        progress.enter("format_reference_state")
        if input_to_tokens(expected.task, [expected.state], expected)[0] != row["text"]:
            raise ValueError("Audited normalized state does not format to the saved raw text")
        sketch_record = copy.deepcopy(normalized)
        sketch_record["i"][1][1] = sketch_record["i"][1][1][: ppt_index + 1]
        progress.enter("restore_sketch_state")
        sketch, _ = auto_scheduler.measure_record.load_record_from_string(json.dumps(sketch_record))
        sketch = recover_measure_input(sketch, rebuild_state=True)

        progress.enter("token_encoding")
        ids = TOKENIZER(row["text"], add_special_tokens=False, truncation=False)["input_ids"]
        if ids != sample["input_ids"] or len(ids) != sample["length"]:
            raise ValueError("Raw text encoding differs from the published Arrow sample")
        verify_full_text(TOKENIZER, ids, row["text"])
        progress.enter("ppt_boundary")
        ppt_end = find_ppt_marker_end(ids, build_ppt_marker_id_candidates(TOKENIZER))
        if ppt_end != sample["ppt_end"] or ppt_end < 0:
            raise ValueError("Published PPT token boundary differs")
        if decode_decision_tokens(TOKENIZER, ids[ppt_end + 1 :]) != row["suffix_text"].split():
            raise ValueError("Published PPT boundary does not decode to the complete suffix")

        def callback(task, states, requested_tokens):
            progress.enter("generation_prompt", callback=len(report["callbacks"]) + 1)
            outputs = []
            for prompt in input_to_tokens(task, states, sketch):
                tokens, prompt_length = replay_suffix(TOKENIZER, row["text"], prompt, ids)
                marker = prompt.split()[-1]
                report["callbacks"].append(
                    {
                        "prompt_tokens": prompt_length,
                        "prompt_tail": prompt.split()[-8:],
                        "suffix_fields": len(tokens),
                        "requested_tokens": int(requested_tokens),
                    }
                )
                if len(report["callbacks"]) == 1 and marker != "PPT":
                    raise ValueError("First production callback does not end at PPT")
                outputs.append(tokens)
            progress.enter("parser_reconstruction", callback=len(report["callbacks"]))
            return outputs

        progress.enter("policy_initialization")
        policy = auto_scheduler.SketchPolicy(sketch.task, seed=0, verbose=0)
        progress.enter("parser_reconstruction")
        states = policy.gen_states([sketch.state], callback)
        if not report["callbacks"] or len(states) != 1:
            raise ValueError(f"Known correct suffix produced {len(states)} states; expected one")
        state = states[0]
        candidate = auto_scheduler.MeasureInput(sketch.task, state)
        progress.enter("state_equivalence")
        candidate_normalized, _ = normalize_record({"i": json.loads(candidate.to_json())})
        steps_equal = candidate_normalized["i"][1] == normalized["i"][1]
        state_equal = str(state) == str(expected.state)
        text_equal = input_to_tokens(sketch.task, [state], sketch)[0] == row["text"]
        report.update(
            normalized_steps_equal=steps_equal,
            effective_state_equal=state_equal,
            formatted_text_equal=text_equal,
        )
        if not (steps_equal and state_equal and text_equal):
            raise ValueError("Reconstructed schedule differs from the audited normalized state")
        progress.enter("apply_generated_state")
        actual_schedule = sketch.task.compute_dag.apply_steps_from_state(
            state, layout_rewrite=sketch.task.layout_rewrite_option
        )
        progress.enter("apply_reference_state")
        expected_schedule = expected.task.compute_dag.apply_steps_from_state(
            expected.state, layout_rewrite=expected.task.layout_rewrite_option
        )
        progress.enter("lower_generated_tir")
        actual_ir = tvm.lower(*actual_schedule)
        progress.enter("lower_reference_tir")
        expected_ir = tvm.lower(*expected_schedule)
        progress.enter("compare_lowered_tir")
        if not tvm.ir.structural_equal(actual_ir, expected_ir, map_free_vars=True):
            raise ValueError("Reconstructed and audited lowered TIR differ")
        report["lowered_tir_equal"] = True
        if defer_build:
            report.update(status="PASSED", serialized_candidate=candidate.serialize())
            progress.enter("reconstruction_complete")
            report["duration_seconds"] = time.monotonic() - started
            return report
        progress.enter("build_export")
        with tempfile.TemporaryDirectory(prefix="roundtrip-", dir=TEMP_ROOT) as directory:
            previous = os.environ.get("TMPDIR")
            old_temp = tempfile.tempdir
            try:
                os.environ["TMPDIR"], tempfile.tempdir = directory, directory
                filename, _, error_no, message, duration = _local_build_worker(
                    candidate.serialize(), tar.tar, 0
                )
                report.update(build_error_no=int(error_no), build_seconds=float(duration))
                if int(error_no):
                    raise ValueError(str(message)[-12000:])
                exported = Path(filename) if filename else None
                if exported is None or not exported.is_file() or exported.stat().st_size == 0:
                    raise ValueError("Builder returned success without an exported module")
                report["exported_module_bytes"] = exported.stat().st_size
            finally:
                tempfile.tempdir = old_temp
                if previous is None:
                    os.environ.pop("TMPDIR", None)
                else:
                    os.environ["TMPDIR"] = previous
        report.update(status="PASSED", stage="complete")
        progress.enter("complete")
    except Exception:
        report["error_message"] = traceback.format_exc()[-16000:]
    report["duration_seconds"] = time.monotonic() - started
    report["result_id"] = digest(report)
    return report
