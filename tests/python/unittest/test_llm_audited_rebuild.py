"""Checks for audited CLM publication and independent raw/Arrow verification."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from LLM.core.schedule_record_utils import (
    AUDIT_SCHEMA,
    atomic_json,
    digest,
    file_sha256,
    workload_key,
)
from LLM.data.verify_audited_clm import verify_dataset


@pytest.fixture
def audited_dataset(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from LLM.data.make_dataset_utils import make_dataset

    words = ["[UNK]", "[PAD]", "PPT", "SPC", "1", "TBS", "PRS"]
    backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    atomic_json(tokenizer_dir / "config.json", {"vocab_size": len(words)})
    audit_dir, raw, arrow = [tmp_path / name for name in ("audit", "raw", "arrow")]
    audit_dir.mkdir()
    raw.mkdir()
    sources = [
        {"name": f"{split}.json", "split": split, "workload_key": workload_key([split, [16]])}
        for split in ("train", "validation")
    ]
    manifest = {
        "schema": AUDIT_SCHEMA,
        "compiler": {},
        "compiler_fingerprint": digest({}),
        "settings": {"partial_scope": False},
        "raw_splits": {"train": {}, "validation": {}},
        "sources": sources,
        "hold_out_keys": [workload_key(["held-out", [16]])],
    }
    manifest["manifest_id"] = digest(manifest)
    atomic_json(audit_dir / "manifest.json", manifest)
    atomic_json(audit_dir / "status.json", {"state": "SUCCEEDED"})
    counts = {}
    for source in sources:
        split = source["split"]
        texts = ["PPT SPC 1 TBS PRS"]
        if split == "train":
            texts.append("PPT " + "SPC 1 " * 600 + "TBS PRS")
        rows = []
        for index, text in enumerate(texts):
            key = digest([split, index])
            rows.append(
                {
                    "text": text,
                    "workload_key": json.dumps(json.loads(source["workload_key"])),
                    "source_file": source["name"],
                    "source_input_hash": key,
                    "build_audit_key": digest([key, manifest["compiler_fingerprint"]]),
                    "compiler_fingerprint": manifest["compiler_fingerprint"],
                    "build_audit_manifest_id": manifest["manifest_id"],
                }
            )
        (raw / f"{split}_raw.json").write_text("".join(json.dumps(row) + "\n" for row in rows))
        counts[split] = len(rows)
    report = {
        "audit_dir": str(audit_dir),
        "manifest_id": manifest["manifest_id"],
        "compiler_fingerprint": manifest["compiler_fingerprint"],
        "counts": counts,
        "outputs": {split: file_sha256(raw / f"{split}_raw.json") for split in counts},
    }
    atomic_json(raw / "build_audit_manifest.json", report)
    make_dataset(
        str(raw / "train_raw.json"),
        str(arrow),
        str(tokenizer_dir),
        "clm",
        validation_file=str(raw / "validation_raw.json"),
        max_length=2048,
    )
    atomic_json(arrow / "build_audit_manifest.json", report)
    return raw, arrow, tokenizer_dir


def test_audited_arrow_verification_preserves_long_sequences_and_split_identity(audited_dataset):
    raw, arrow, tokenizer = audited_dataset
    result = verify_dataset(raw, arrow, tokenizer)
    assert result["status"] == "VERIFIED"
    assert result["splits"]["train"]["records"] == 2
    assert result["splits"]["train"]["max_length"] > 1024
    assert result["splits"]["validation"]["workloads"] == 1
    assert result["generation_parser_roundtrip_checked"] is False


def test_verify_data_cli_checks_and_reports_the_saved_dataset(audited_dataset, tmp_path):
    raw, arrow, tokenizer = audited_dataset
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "LLM/__main__.py"),
            "verify-data",
            "--raw-dir",
            str(raw),
            "--dataset-dir",
            str(arrow),
            "--tokenizer-path",
            str(tokenizer),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((arrow / "verification.json").read_text())
    assert report["status"] == "VERIFIED"
    assert report["splits"]["train"]["records"] == 2
    assert '"status": "VERIFIED"' in result.stdout
    assert (arrow / "verification.json").exists()


def test_verification_rejects_raw_changed_after_export(audited_dataset):
    raw, arrow, tokenizer = audited_dataset
    with (raw / "train_raw.json").open("a") as stream:
        stream.write('{"text":"tampered"}\n')
    with pytest.raises(ValueError, match="Raw train file differs"):
        verify_dataset(raw, arrow, tokenizer)


def test_verification_rejects_same_length_corrupted_labels(audited_dataset, monkeypatch):
    import datasets

    raw, arrow, tokenizer = audited_dataset
    loaded = datasets.load_from_disk(str(arrow))

    def corrupt(row):
        return {"labels": [0] + row["labels"][1:]}

    loaded["train"] = loaded["train"].map(corrupt, keep_in_memory=True)
    monkeypatch.setattr(datasets, "load_from_disk", lambda *args, **kwargs: loaded)
    with pytest.raises(ValueError, match="labels differ"):
        verify_dataset(raw, arrow, tokenizer)


@pytest.mark.parametrize("child_code", [0, 7])
def test_background_worker_only_publishes_after_success_and_verification(
    tmp_path, monkeypatch, child_code
):
    from LLM.data import rebuild_audited_clm as module
    from LLM.data import verify_audited_clm

    args = SimpleNamespace(
        run_dir=str(tmp_path / "run"),
        audit_dir=str(tmp_path / "audit"),
        tokenizer_path=str(tmp_path / "tokenizer"),
        max_length=2048,
    )
    monkeypatch.setattr(module, "snapshot_inputs", lambda args: {"code": {}})
    monkeypatch.setattr(module, "verify_snapshot", lambda snapshot: None)
    verified = []

    def verify(raw, pending, tokenizer, length):
        assert pending.is_dir()
        assert not pending.with_name("dataset").exists()
        verified.append(True)
        return {"status": "VERIFIED"}

    monkeypatch.setattr(verify_audited_clm, "verify_dataset", verify)

    class Child:
        pid = 123

        def __init__(self, command, **kwargs):
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return child_code

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    assert module.run(args) == int(bool(child_code))
    state = json.loads((Path(args.run_dir) / "status.json").read_text())
    assert state["state"] == ("FAILED" if child_code else "SUCCEEDED")
    assert (Path(args.run_dir) / "dataset").exists() == (not child_code)
    assert bool(verified) == (not child_code)


@pytest.mark.parametrize("problem", [None, "formatter_changed", "encoding_incomplete"])
def test_reuse_completed_stages_preserves_failed_run_and_requires_compatible_inputs(
    tmp_path, problem
):
    from LLM.data.rebuild_audited_clm import reuse_completed_stages

    source, target = tmp_path / "failed", tmp_path / "recovery"
    for directory in (source / "raw", source / "dataset.pending", source / "frozen_code", target):
        directory.mkdir(parents=True)
    (source / "raw" / "train_raw.json").write_text("raw data")
    (source / "dataset.pending" / "data.arrow").write_bytes(b"arrow data")
    (source / "dataset.pending" / ".tokenize_cache").mkdir()
    snapshot = {
        "audit_dir": "/audit",
        "audit_manifest_id": "audit identity",
        "audit_manifest_sha256": "audit hash",
        "compiler_fingerprint": "compiler hash",
        "tokenizer_path": "/tokenizer",
        "tokenizer": {"tokenizer.json": "tokenizer hash"},
        "max_length": 2048,
        "overlength_policy": "error",
        "code": {},
    }
    for name in ("make_dataset.py", "verify_audited_clm.py"):
        frozen = source / "frozen_code" / name
        frozen.write_text("previous code")
        snapshot["code"][f"/llm/{name}"] = file_sha256(frozen)
    atomic_json(source / "manifest.json", snapshot)
    atomic_json(
        source / "status.json",
        {
            "state": "FAILED",
            "stage": "ARROW_ENCODING" if problem == "encoding_incomplete" else "VERIFYING",
            "stage_seconds": {"RAW_EXPORT": 1, "ARROW_ENCODING": 2},
        },
    )
    snapshot["code"]["/llm/verify_audited_clm.py"] = "corrected verifier hash"
    if problem == "formatter_changed":
        snapshot["code"]["/llm/make_dataset.py"] = "different formatter hash"
    if problem:
        with pytest.raises(ValueError, match="code differs|completed raw/Arrow"):
            reuse_completed_stages(source, target, snapshot)
        assert not (target / "raw").exists()
    else:
        result = reuse_completed_stages(source, target, snapshot)
        assert result["changed_orchestration_or_verification_code"] == ["verify_audited_clm.py"]
        assert (target / "raw" / "train_raw.json").read_text() == "raw data"
        assert (target / "dataset.pending" / "data.arrow").read_bytes() == b"arrow data"
        assert not (target / "dataset.pending" / ".tokenize_cache").exists()
        (target / "raw" / "train_raw.json").write_text("independent copy")
        assert (source / "raw" / "train_raw.json").read_text() == "raw data"
        assert json.loads((source / "status.json").read_text())["state"] == "FAILED"
