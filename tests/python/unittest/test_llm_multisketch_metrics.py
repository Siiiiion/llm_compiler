"""Multi-sketch coverage and exact candidate accounting are different metrics."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from LLM.validation.multisketch_metrics import summarize_multisketch


GOOD = {"resource_valid": True, "built": True, "correct": True}


def attempt(mode, identities=(), check=None):
    return {
        "mode": mode,
        "parsed": len(identities),
        "candidates": [
            {"identity": identity, "check": dict(GOOD if check is None else check)}
            for identity in identities
        ],
        "callbacks": [],
        "telemetry": {"generated_tokens": 10, "token_budget": 256},
        "generation_seconds": 0.25,
    }


def case(index, workload, sketch, attempts):
    return {
        "index": index,
        "workload_index": workload,
        "workload_key": f"workload-{workload}",
        "family": "test",
        "sketch_index": sketch,
        "sample_id": f"sample-{index}",
        "sketch_id": f"sketch-{workload}-{sketch}",
        "attempts": attempts,
        # Summaries must be derived from attempts rather than trusting stale data.
        "modes": {"state_v2": {"unique_correct": 1000}},
    }


def test_same_state_across_sketches_consumes_budget_without_growing_workload_pool():
    cases = [
        case(0, 0, 0, [attempt("state_v2", ["a"]), attempt("state_v2", ["a"])]),
        case(1, 0, 1, [attempt("state_v2", ["a"]), attempt("state_v2", ["b"])]),
    ]
    before = copy.deepcopy(cases)
    result = summarize_multisketch(cases, ["state_v2"], required=2)
    per_case = result["per_case"]["state_v2"]
    pooled = result["per_workload"][0]["modes"]["state_v2"]
    assert per_case["case_count"] == 2
    assert "workloads" not in per_case
    assert per_case["unique_correct"] == 3
    assert per_case["duplicate_states"] == 1
    assert pooled["unique_states"] == pooled["resource_valid"] == pooled["unique_built"] == 2
    assert pooled["unique_correct"] == 2
    assert pooled["duplicate_states"] == 2
    assert pooled["attempts"] == 4
    assert pooled["generated_tokens"] == 40
    assert pooled["token_budget"] == 1024
    assert pooled["generation_seconds"] == 1.0
    assert pooled["pooled_k_reached"] and pooled["any_sketches_k_reached"]
    assert not pooled["all_sketches_k_reached"]
    assert pooled["sketch_shortfall"] == 1
    assert pooled["min_sketch_unique_correct"] == 1
    assert pooled["max_sketch_unique_correct"] == 2
    assert cases == before


def test_pooling_small_sketches_cannot_claim_any_sketch_reached_k():
    cases = [
        case(0, 0, 0, [attempt("state_v2", ["a", "b"])]),
        case(1, 0, 1, [attempt("state_v2", ["c", "d"])]),
    ]
    result = summarize_multisketch(cases, ["state_v2"], required=4)
    workload = result["per_workload"][0]["modes"]["state_v2"]
    summary = result["workload_summary"]["state_v2"]
    assert workload["pooled_k_reached"]
    assert not workload["any_sketches_k_reached"]
    assert not workload["all_sketches_k_reached"]
    assert workload["shortfall"] == 0 and workload["sketch_shortfall"] == 4
    assert summary["pooled_k_reached_workloads"] == 1
    assert summary["all_sketches_k_reached_workloads"] == 0
    assert summary["all_sketches_k_reached_multi_sketch_rate"] == 0


def test_real_workload_macro_keeps_zero_candidates_and_single_sketch_is_not_multi_coverage():
    cases = [
        case(0, 0, 0, [attempt("state_v2", ["a", "b"])]),
        case(1, 0, 1, [attempt("state_v2", ["a", "c"])]),
        case(2, 1, 0, [attempt("state_v2")]),
    ]
    result = summarize_multisketch(cases, ["state_v2"], required=2)
    summary = result["workload_summary"]["state_v2"]
    assert result["case_count"] == 3 and result["workload_count"] == 2
    assert result["multi_sketch_workload_count"] == 1
    assert result["per_case"]["state_v2"]["unique_correct_case_macro"] == 4 / 3
    assert summary["unique_correct_workload_macro"] == 3 / 2
    assert summary["all_sketches_k_reached_rate"] == 0.5
    assert summary["all_sketches_k_reached_multi_sketch_rate"] == 1.0
    assert summary["failures"] == {"no_final_state": 1}
    assert result["per_workload"][1]["multi_sketch"] is False
    assert result["per_workload"][1]["modes"]["state_v2"]["unique_correct"] == 0


def test_modes_and_workloads_do_not_share_candidate_counts_or_gate_outcomes():
    rows = [
        case(0, 0, 0, [attempt("state_v1", ["a"]), attempt("state_v2", ["a"])]),
        case(
            1,
            0,
            1,
            [
                attempt("state_v1", ["a"]),
                attempt("state_v2", ["bad"], check={"resource_valid": False}),
            ],
        ),
        case(2, 1, 0, [attempt("state_v1", ["a"]), attempt("state_v2")]),
    ]
    result = summarize_multisketch(rows, ["state_v1", "state_v2"], required=1)
    summary = result["workload_summary"]
    assert summary["state_v1"]["unique_correct"] == 2
    assert summary["state_v1"]["duplicate_states"] == 1
    assert summary["state_v1"]["all_sketches_k_reached_workloads"] == 2
    assert summary["state_v2"]["unique_correct"] == 1
    assert summary["state_v2"]["all_sketches_k_reached_workloads"] == 0
    assert summary["state_v2"]["failures"] == {"resource_rejected": 1, "no_final_state": 1}
    assert summary["state_v1"]["attempts"] == summary["state_v2"]["attempts"] == 3


def test_no_multi_sketch_workload_has_no_multi_sketch_success_rate():
    rows = [case(0, 0, 0, [attempt("state_v2", ["a"])])]
    result = summarize_multisketch(rows, ["state_v2"], required=1)
    summary = result["workload_summary"]["state_v2"]
    assert result["multi_sketch_workload_count"] == 0
    assert summary["all_sketches_k_reached_workloads"] == 1
    assert summary["all_sketches_k_reached_multi_sketch_workloads"] == 0
    assert summary["all_sketches_k_reached_multi_sketch_rate"] is None


def test_duplicate_sketch_identity_cannot_inflate_multi_sketch_coverage():
    rows = [case(0, 0, 0, []), case(1, 0, 0, [])]
    with pytest.raises(ValueError, match="distinct sketch"):
        summarize_multisketch(rows, ["state_v2"])


@pytest.mark.parametrize("required", [0, -1, True, 1.5])
def test_invalid_candidate_target_is_rejected(required):
    with pytest.raises(ValueError, match="positive integer"):
        summarize_multisketch([case(0, 0, 0, [])], ["state_v2"], required)
