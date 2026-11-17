"""Regression coverage for known-suffix generation replay and sampled-result semantics."""

import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from LLM.validation.validate_generation_roundtrip import (
    length_bucket,
    operator_family,
    representative_indices,
    summarize,
)
from LLM.validation.generation_roundtrip_worker import replay_suffix, verify_full_text


def test_representatives_preserve_length_extremes_and_collapse_singletons():
    values = [(1200, 3, 100), (200, 0, 0), (700, 1, 30), (600, 2, 60)]
    result = representative_indices(values)
    assert result == {
        0: {"length": 200, "position": 0, "roles": ["shortest"]},
        1: {"length": 700, "position": 30, "roles": ["median"]},
        3: {"length": 1200, "position": 100, "roles": ["longest"]},
    }
    assert representative_indices([(900, 4, 10)])[4]["roles"] == ["shortest", "median", "longest"]


def test_operator_family_and_length_strata_use_actual_inputs():
    key = '["hash",[1,16]]'
    assert operator_family("p0 T_batch_matmul_NT hash 1 16", key) == "batch_matmul"
    assert operator_family("p0 T_conv2d T_add hash 1 16", key) == "convolution"
    assert [length_bucket(v) for v in (512, 513, 1024, 1025)] == [
        "le512",
        "513to1024",
        "513to1024",
        "gt1024",
    ]


@pytest.mark.parametrize(
    "partial,failed,missing",
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True)],
)
def test_summary_never_claims_more_than_sampled_and_successfully_checked(partial, failed, missing):
    selection = {
        "samples": 2,
        "workloads": 2,
        "total_workloads": 3 if partial else 2,
        "partial_scope": partial,
    }
    reports = [
        {
            "split": "train",
            "operator_family": "matmul_dense",
            "length_bucket": "gt1024",
            "workload_key": "first",
            "status": "PASSED",
            "stage": "complete",
        },
        {
            "split": "validation",
            "operator_family": "convolution",
            "length_bucket": "le512",
            "workload_key": "second",
            "status": "FAILED" if failed else "PASSED",
            "stage": "parser_reconstruction" if failed else "complete",
        },
    ]
    result = summarize(reports[:1] if missing else reports, selection)
    assert result["all_selected_samples_passed"] == (not failed and not missing)
    assert result["generation_parser_roundtrip_checked"] == (not failed and not missing)
    assert result["full_workload_coverage"] == (not partial and not missing)
    assert result["all_dataset_records_roundtrip_checked"] is False
    assert result["model_generation_quality_evaluated"] is False
    assert result["runtime_correctness_checked"] is False
    assert result["latency_measured"] is False


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [[ord(c) for c in value] for value in text]}
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids)


def test_replay_uses_production_decoder_and_preserves_stage_separator(monkeypatch):
    from LLM.generation import gen_state

    tokenizer = CharacterTokenizer()
    text = "prompt PPT SPC 4 1 1024 32 1 TBS PRS 2"
    ids = tokenizer(text)["input_ids"]
    tokens, prompt_length = replay_suffix(tokenizer, text, "prompt PPT", ids)
    assert prompt_length == len("prompt PPT")
    assert tokens == "SPC 4 1 1024 32 1 TBS PRS 2".split()
    monkeypatch.setattr(gen_state, "decode_decision_tokens", lambda *args: ["SPC1024"])
    with pytest.raises(ValueError, match="suffix decoding changed"):
        replay_suffix(tokenizer, text, "prompt PPT", ids)


def test_replay_rejects_wrong_stage_prompt_or_corrupted_training_tokens():
    tokenizer = CharacterTokenizer()
    text = "prompt PPT SPC 4 1 1024 32 1"
    ids = tokenizer(text)["input_ids"]
    with pytest.raises(ValueError, match="not an exact prefix"):
        replay_suffix(tokenizer, text, "prompt PPT TBS", ids)
    with pytest.raises(ValueError, match="generation condition differ"):
        replay_suffix(tokenizer, text, "prompt PPT", ids[:-1])


def test_full_schedule_decode_retains_special_ppt_marker():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]", "prompt", "PPT", "SPC", "4"]
    backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": ["PPT"]})
    text = "prompt PPT SPC 4"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(ids, skip_special_tokens=True) != text
    verify_full_text(tokenizer, ids, text)
    with pytest.raises(ValueError, match="Full text is not preserved"):
        verify_full_text(tokenizer, ids[:-1], text)
