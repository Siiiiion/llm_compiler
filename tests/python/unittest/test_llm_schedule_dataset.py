# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Regression coverage for preserving complete LLM schedule training sequences."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from LLM.data.make_dataset_utils import make_dataset, validate_clm_dataset_integrity
from LLM.core.tokenization_protocol import (
    PPT_BOUNDARY_VERSION,
    build_ppt_marker_id_candidates,
    encode_prompt_suffix,
    find_ppt_marker_end,
)
from LLM.train.train_qwen3_dpo import _tokenize_pair


class _SpaceAwareTokenizer:
    """Small tokenizer fixture that exposes a standalone separator token."""

    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        ids = {"PPT": 10, "SPC": 11, "TBS": 12, " ": 20}
        output = []
        pending_space = False
        for piece in text.split(" "):
            if pending_space:
                output.append(ids[" "])
            if piece:
                output.append(ids.get(piece, 13))
            pending_space = True
        return {"input_ids": output}


@pytest.fixture
def schedule_inputs(tmp_path):
    """Use a local tokenizer and a >1024-token schedule with essential tail decisions."""
    words = ["[UNK]", "[PAD]", "PPT", "SPC", "TBS", "PRS", "PR", "2", "0", "1", "8", "1024"]
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {word: index for index, word in enumerate(words)}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", model_max_length=131072
    )
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    short_text = "PPT SPC 2 0 1024 8 1 TBS PRS 2 PR 2 0 8"
    long_text = "PPT " + "SPC 2 0 1024 8 1 " * 180 + "TBS SPC 2 0 1024 8 1 PRS 2 PR 2 0 8"
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    train.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"text": short_text, "workload_key": "short", "source_file": "short.json"},
                {"text": long_text, "workload_key": "long", "source_file": "long.json"},
            ]
        )
        + "\n"
    )
    validation.write_text(
        json.dumps(
            {"text": short_text, "workload_key": "held-out", "source_file": "validation.json"}
        )
        + "\n"
    )
    return tokenizer, tokenizer_path, train, validation, short_text, long_text


def test_complete_long_schedule_round_trip(schedule_inputs, tmp_path):
    tokenizer, tokenizer_path, train, validation, short_text, long_text = schedule_inputs
    output = tmp_path / "dataset"
    stats = make_dataset(
        str(train), str(output), str(tokenizer_path), "clm", validation_file=str(validation)
    )
    result = datasets.load_from_disk(str(output))
    assert stats["max_length"] == 2048
    assert len(result["train"]) == 2
    assert len(result["validation"]) == 1
    for row, text in zip(result["train"], (short_text, long_text)):
        expected = tokenizer(text)["input_ids"]
        assert row["input_ids"] == expected
        assert row["labels"] == expected
        assert row["original_length"] == row["length"] == len(expected)
        assert row["sequence_complete"] and not row["was_truncated"]
        assert row["ppt_boundary_version"] == PPT_BOUNDARY_VERSION
        assert row["ppt_end"] == 0
    assert result["train"][1]["length"] > 1024
    assert result["train"][1]["input_ids"][-6:] == tokenizer("PRS 2 PR 2 0 8")["input_ids"]
    assert stats["splits"]["train"]["original_length"]["max"] > 1024
    assert stats["splits"]["train"]["workloads"]["long"]["within_limit_records"] == 1
    assert "workload_key" not in result["train"].column_names
    validate_clm_dataset_integrity(result["train"], "train")


def test_overlength_error_reports_lost_coverage_without_saving(schedule_inputs, tmp_path):
    _, tokenizer_path, train, _, _, _ = schedule_inputs
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="No sequences were truncated"):
        make_dataset(
            str(train), str(output), str(tokenizer_path), "clm", valid_percentage=0, max_length=1024
        )
    assert not (output / "dataset_dict.json").exists()
    stats = json.loads((output / "tokenization_stats.json").read_text())
    assert stats["status"] == "rejected"
    assert stats["splits"]["train"]["overlength_records"] == 1
    assert stats["splits"]["train"]["saved_records"] == 0
    assert stats["splits"]["train"]["workloads_without_records_within_limit"] == ["long"]


def test_drop_discards_whole_schedule_and_preserves_validation(schedule_inputs, tmp_path):
    tokenizer, tokenizer_path, train, validation, short_text, _ = schedule_inputs
    output = tmp_path / "drop"
    stats = make_dataset(
        str(train),
        str(output),
        str(tokenizer_path),
        "clm",
        validation_file=str(validation),
        max_length=1024,
        overlength_policy="drop",
    )
    result = datasets.load_from_disk(str(output))
    assert len(result["train"]) == len(result["validation"]) == 1
    assert result["train"][0]["input_ids"] == tokenizer(short_text)["input_ids"]
    assert stats["splits"]["train"]["dropped_overlength_records"] == 1
    assert stats["splits"]["train"]["saved_records"] == 1
    assert stats["splits"]["validation"]["dropped_overlength_records"] == 0


def test_exact_length_limit_preserves_tail(schedule_inputs, tmp_path):
    tokenizer, tokenizer_path, train, _, _, long_text = schedule_inputs
    limit = len(tokenizer(long_text)["input_ids"])
    make_dataset(
        str(train),
        str(tmp_path / "exact"),
        str(tokenizer_path),
        "clm",
        valid_percentage=0,
        max_length=limit,
    )
    result = datasets.load_from_disk(str(tmp_path / "exact"))
    assert result["train"][1]["input_ids"] == tokenizer(long_text)["input_ids"]


def test_drop_cannot_save_an_empty_split(schedule_inputs, tmp_path):
    _, tokenizer_path, train, _, _, _ = schedule_inputs
    output = tmp_path / "empty"
    with pytest.raises(ValueError, match="no complete samples"):
        make_dataset(
            str(train),
            str(output),
            str(tokenizer_path),
            "clm",
            valid_percentage=0,
            max_length=1,
            overlength_policy="drop",
        )
    assert not (output / "dataset_dict.json").exists()


@pytest.mark.parametrize(
    "column,value",
    [
        ("original_length", 4),
        ("length", 4),
        ("was_truncated", True),
        ("sequence_complete", False),
        ("labels", [1]),
        ("original_length", None),
        ("was_truncated", None),
    ],
)
def test_training_guard_rejects_incomplete_sequences(column, value):
    row = {
        "input_ids": [1, 2, 3],
        "labels": [1, 2, 3],
        "length": 3,
        "original_length": 3,
        "was_truncated": False,
        "sequence_complete": True,
    }
    row[column] = value
    dataset = datasets.Dataset.from_list([row])
    with pytest.raises(ValueError):
        validate_clm_dataset_integrity(dataset, "train")


def test_legacy_data_requires_explicit_opt_in(caplog):
    dataset = datasets.Dataset.from_list([{"input_ids": [1, 2], "labels": [1, 2]}])
    with pytest.raises(ValueError, match="Rebuild from the untruncated raw JSON"):
        validate_clm_dataset_integrity(dataset, "train")
    with caplog.at_level(logging.WARNING):
        validate_clm_dataset_integrity(dataset, "train", allow_legacy=True)
    assert "completeness cannot be verified" in caplog.text


def test_legacy_opt_in_does_not_allow_known_truncation():
    dataset = datasets.Dataset.from_list([{"input_ids": [1], "labels": [1], "was_truncated": True}])
    with pytest.raises(ValueError, match="incomplete sequences"):
        validate_clm_dataset_integrity(dataset, "train", allow_legacy=True)


def test_ppt_boundary_excludes_trailing_separator_and_preserves_condition():
    tokenizer = _SpaceAwareTokenizer()
    patterns = build_ppt_marker_id_candidates(tokenizer)
    ids = tokenizer("PPT SPC", add_special_tokens=False)["input_ids"]
    assert find_ppt_marker_end(ids, patterns) == 0
    prompt_ids, input_ids, labels = encode_prompt_suffix(tokenizer, "PPT", "SPC")
    assert prompt_ids == [10]
    assert input_ids == [10, 20, 11]
    assert labels == [-100, 20, 11]


def test_preference_pair_uses_joint_boundary_without_partial_truncation():
    result = _tokenize_pair(
        {"prompt": "PPT", "chosen": "SPC", "rejected": "TBS"},
        _SpaceAwareTokenizer(),
        max_prompt_length=8,
        max_length=8,
        append_eos=False,
    )
    assert result["valid"] == 1
    assert result["chosen_input_ids"] == [10, 20, 11]
    assert result["chosen_labels"] == [-100, 20, 11]
    too_short = _tokenize_pair(
        {"prompt": "PPT", "chosen": "SPC SPC", "rejected": "TBS TBS"},
        _SpaceAwareTokenizer(),
        max_prompt_length=8,
        max_length=3,
        append_eos=False,
    )
    assert too_short["valid"] == 0


def test_rebuild_cli_preserves_splits_and_refuses_overwrite(schedule_inputs, tmp_path):
    _, tokenizer_path, train, validation, _, _ = schedule_inputs
    output = tmp_path / "rebuilt"
    command = [
        sys.executable,
        "-m",
        "LLM",
        "rebuild-dataset",
        "--train-file",
        str(train),
        "--validation-file",
        str(validation),
        "--tokenizer-path",
        str(tokenizer_path),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    output_lines = completed.stdout.strip().splitlines()
    result = json.loads("\n".join(output_lines[output_lines.index("{") :]))
    assert result["splits"]["train"]["saved_records"] == 2
    before = (output / "dataset_dict.json").read_bytes()
    repeated = subprocess.run(command, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "old data will not be overwritten" in repeated.stderr
    assert (output / "dataset_dict.json").read_bytes() == before
