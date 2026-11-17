"""Explain truncated and corrupted fixed SPC fields without modifying model decisions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from LLM.experiments.diagnose_model_generation import inspect_split_output, split_expectations
from LLM.experiments.diagnose_generation_interventions import (
    replace_split_headers,
    raise_thread_factor,
)


def test_split_diagnosis_distinguishes_wrong_fields_and_missing_decisions():
    expected = split_expectations([["SP", 2, 5, 512, [1, 1], 1], ["PPT"]], "PPT")
    valid = "SPC 2 5 512 4 8 1".split()
    assert inspect_split_output(valid + ["TBS"], expected) == {
        "valid": True,
        "required_fields": 7,
        "trailing_fields": 1,
    }
    wrong = valid.copy()
    wrong[3] = "256"
    error = inspect_split_output(wrong, expected)
    assert error["kind"] == "fixed_field_mismatch"
    assert error["field"] == "extent" and error["expected"] == 512
    error = inspect_split_output(valid[:-1], expected)
    assert error["kind"] == "incomplete" and error["field"] == "inner_to_outer"


def test_thread_binding_checks_splits_since_ppt_without_rechecking_spc():
    steps = [
        ["SP", 2, 5, 512, [1, 1], 1],
        ["PPT"],
        ["SPC", 2, 5, 512, [4, 8], 1],
        ["SP", 1, 0, 16, [1], 1],
        ["TBS"],
    ]
    expected = split_expectations(steps, "TBS")
    assert len(expected) == 1 and expected[0]["stage"] == 1
    assert inspect_split_output("SPC 1 0 16 4 1".split(), expected)["valid"]


def test_fixed_header_intervention_preserves_every_model_factor_and_original_output():
    expected = [{"stage": 5, "iter": 0, "extent": 128, "factors": 1, "inner_to_outer": 1}]
    original = "SPC 5 0 12 8 1 PRS 4 8 11".split()
    corrected, changes = replace_split_headers(original, expected)
    assert corrected == "SPC 5 0 128 8 1 PRS 4 8 11".split()
    assert original[3] == "12"
    assert changes == [{"index": 3, "from": "12", "to": "128"}]


def test_manual_thread_factor_intervention_changes_one_field_only():
    expected = [{"stage": 2, "iter": 5, "extent": 512, "factors": 4, "inner_to_outer": 1}]
    original = "SPC 2 5 512 1 16 1 2 1".split()
    corrected, changes = raise_thread_factor(original, expected, "matmul_dense")
    assert corrected == "SPC 2 5 512 1 32 1 2 1".split()
    assert original[5] == "16"
    assert changes == [{"index": 5, "from": "16", "to": "32"}]
