"""Distinct-state accounting under fixed attempts and candidate check failures."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from LLM.core.schedule_record_utils import input_hash
from LLM.validation.validate_multicandidate_generation import (
    aggregate,
    read_worker_check,
    summarize_workload,
)


def attempt(mode, candidates=(), parsed=1, rejected=False):
    return {
        "mode": mode,
        "parsed": parsed,
        "candidates": [{"identity": identity, "check": check} for identity, check in candidates],
        "callbacks": [
            {"constraint_traces": [{"failure": "no legal completion"}] if rejected else []}
        ],
        "telemetry": {"generated_tokens": 12, "token_budget": 256},
        "generation_seconds": 0.25,
    }


GOOD = {"resource_valid": True, "built": True, "correct": True}


def test_duplicates_consume_budget_without_satisfying_candidate_target():
    rows = [attempt("state_v1", [("a", GOOD)]) for _ in range(8)]
    result = summarize_workload(rows, "state_v1", 4)
    assert result["attempts"] == 8
    assert result["unique_correct"] == 1
    assert result["duplicate_states"] == 7
    assert result["duplicate_rate_of_parsed"] == 7 / 8
    assert result["shortfall"] == 3
    assert result["attempts_to_k"] is None
    assert result["generated_tokens"] == 96
    assert result["token_budget"] == 2048


def test_failure_gates_and_modes_do_not_share_candidate_counts():
    rows = [
        attempt("none", [("a", GOOD)]),
        attempt("state_v1", [("a", GOOD)]),
        attempt("state_v1", [("b", {"resource_valid": False})]),
        attempt("state_v1", [("c", {"resource_valid": True, "built": False})]),
        attempt("state_v1", [("d", {"resource_valid": True, "built": True})]),
        attempt("state_v1", [], parsed=0, rejected=True),
        attempt("state_v1", [], parsed=0),
    ]
    result = summarize_workload(rows, "state_v1", 2)
    assert result["attempts"] == 6
    assert result["parsed_attempts"] == 4
    assert result["unique_states"] == 4
    assert result["unique_built"] == 2
    assert result["unique_correct"] == 1
    assert result["failures"] == {
        "constraint_rejected": 1,
        "no_final_state": 1,
        "resource_rejected": 1,
        "build_rejected": 1,
        "correctness_rejected": 1,
    }


def test_k_reached_keeps_later_attempts_and_macro_uses_all_workloads():
    rows = [attempt("none", [(str(i), GOOD)]) for i in range(8)]
    success = summarize_workload(rows, "none", 4)
    failure = summarize_workload([attempt("none", parsed=0) for _ in range(8)], "none", 4)
    result = aggregate([{"modes": {"none": success}}, {"modes": {"none": failure}}], ["none"])
    assert success["attempts_to_k"] == 4
    assert success["attempts"] == 8
    assert success["unique_correct"] == 8
    assert result["none"]["unique_correct_macro"] == 4
    assert result["none"]["k_reached_rate"] == 0.5
    assert result["none"]["attempts"] == 16


def test_candidate_identity_keeps_decision_factors_and_ignores_measurements():
    record = {
        "i": [['["test",[16]]', "cuda"], [[], [["SP", 0, 0, 16, [2], 1]]]],
        "r": [[1], 0],
        "v": "v0.6",
    }
    changed_factor = copy.deepcopy(record)
    changed_factor["i"][1][1][0][4] = [4]
    changed_result = copy.deepcopy(record)
    changed_result["r"] = [[2], 0]
    assert input_hash(record) != input_hash(changed_factor)
    assert input_hash(record) == input_hash(changed_result)


def test_worker_teardown_crash_preserves_reported_runtime_rejection(tmp_path):
    path = tmp_path / "candidate.check.json"
    rejected = {
        "resource_valid": True,
        "built": True,
        "correct": False,
        "error": "CUDA_ERROR_MISALIGNED_ADDRESS",
    }
    path.write_text(json.dumps(rejected))
    check = read_worker_check(path, -6, "CUDA teardown aborted")
    assert check["resource_valid"] and check["built"] and not check["correct"]
    assert check["worker_return_code"] == -6
    assert json.loads(path.read_text()) == check
    row = attempt("state_v2", [("bad", check)])
    assert summarize_workload([row], "state_v2", 4)["failures"] == {"correctness_rejected": 1}


@pytest.mark.parametrize("report", [None, GOOD, {"correct": False}])
def test_unreported_worker_crash_is_an_infrastructure_failure(tmp_path, report):
    path = tmp_path / "candidate.check.json"
    if report is not None:
        path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="worker exited -6"):
        read_worker_check(path, -6, "worker abort")
