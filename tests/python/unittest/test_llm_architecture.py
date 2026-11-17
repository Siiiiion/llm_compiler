"""Exercise package boundaries, module subprocesses, CLI forwarding and fresh TVM workers."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

from LLM.core.layout import (
    code_dependencies,
    module_command,
    project_pythonpath,
    source_path,
)


def test_command_registry_resolves_to_real_implementation_sources():
    from LLM.__main__ import COMMANDS

    for module, _ in COMMANDS.values():
        expected = REPOSITORY.joinpath(*module.split(".")).with_suffix(".py")
        assert source_path(expected.name) == expected


def test_module_command_runs_after_changing_directory(tmp_path):
    result = subprocess.run(
        module_command("rebuild_audited_clm.py", "--help"),
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=project_pythonpath(), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--run-dir" in result.stdout


def test_measured_history_observes_the_same_registered_platform(monkeypatch):
    from LLM.core import common
    from LLM.core import common as implementation
    from LLM.generation import measured_history

    monkeypatch.setattr(common, "HARDWARE_PLATFORM", "test-platform")
    monkeypatch.setattr(measured_history, "measured_pkl", {"already-measured"})
    assert implementation.HARDWARE_PLATFORM == "test-platform"
    assert measured_history.check_measured("already-measured")
    assert not measured_history.check_measured("new-state")


def test_manifests_include_extracted_implementation_dependencies():
    paths = code_dependencies(
        [
            source_path("validate_multisketch_generation.py"),
            source_path("validate_generation_roundtrip.py"),
        ]
    )
    names = {p.name for p in paths}
    assert {
        "layout.py",
        "io.py",
        "runtime.py",
        "sampling.py",
        "candidate_metrics.py",
        "multisketch_metrics.py",
        "__main__.py",
        "measured_history.py",
    } <= names
    assert len(paths) == len(set(paths))
    assert source_path("gen_state.py") in paths
    assert len({p.name for p in paths}) == len(paths), "Frozen copies use unique basenames"


def test_statistical_helpers_do_not_import_an_experiment_runner(tmp_path):
    environment = dict(os.environ, PYTHONPATH=project_pythonpath(), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from LLM.validation.multisketch_metrics import summarize_multisketch; "
            "assert 'LLM.validation.validate_multicandidate_generation' not in sys.modules; "
            "assert 'torch' not in sys.modules; assert 'tvm' not in sys.modules",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_help_needs_only_the_standard_library(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "LLM", "--help"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(REPOSITORY), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "validate-sketches" in result.stdout and "train-dpo" in result.stdout


@pytest.mark.parametrize("command", ["audit", "rebuild-audited", "validate-roundtrip", "generate"])
def test_unified_cli_forwards_command_help(command, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "LLM", command, "--help"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=project_pythonpath(), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout and "unknown command" not in result.stderr


def test_fresh_tvm_workers_import_the_package_from_another_directory(tmp_path, monkeypatch):
    from LLM.core.runtime import bounded_pool
    from LLM.core.schedule_record_utils import workload_key

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", project_pythonpath(os.environ.get("PYTHONPATH", "")))
    with bounded_pool(max_workers=1, timeout=30) as pool:
        assert pool.submit(workload_key, '["task", [16]]').result() == '["task",[16]]'


def test_importing_training_launchers_does_not_launch_training(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, subprocess; os.environ['CUDA_VISIBLE_DEVICES']=''; "
            "subprocess.Popen=lambda *a, **k: (_ for _ in ()).throw(AssertionError('spawned')); "
            "import LLM.train.run_train_qwen3_clm; import LLM.train.run_train_qwen3_dpo; "
            "import LLM.experiments.evaluate",
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=project_pythonpath(), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["clm", "dpo"])
def test_training_launcher_resolves_script_from_another_directory(kind, tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer_config.json").write_text("{}")
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    (pairs / "train_pairs.jsonl").write_text('{"prompt":"p","chosen":"a","rejected":"b"}\n')
    monkeypatch.chdir(tmp_path)
    for name, value in {
        "CUDA_VISIBLE_DEVICES": "0",
        "NPROC_PER_NODE": "1",
        "SMOKE_TEST": "true",
        "MODEL_NAME_OR_PATH": str(model),
        "POLICY_MODEL_PATH": str(model),
        "REF_MODEL_PATH": str(model),
        "TOKENIZER_NAME": str(model),
        "PREFERENCE_DATASET": str(pairs),
        "DATASET_NAME": str(tmp_path / "data"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "TRAIN_STAGE": "stage1",
    }.items():
        monkeypatch.setenv(name, value)
    module = importlib.import_module(f"LLM.train.run_train_qwen3_{kind}")
    commands = []
    method = "Popen" if kind == "clm" else "run"
    monkeypatch.setattr(
        module.subprocess, method, lambda command, **kwargs: commands.append(command)
    )
    module.main()
    assert len(commands) == 1
    assert str(REPOSITORY / f"LLM/train/train_qwen3_{kind}.py") in str(commands[0])


@pytest.mark.parametrize("direct", [False, True])
def test_cli_supplies_import_paths_and_preserves_exit_status(tmp_path, direct):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    launcher = REPOSITORY / "LLM/__main__.py"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, runpy, sys; "
            "runpy.run_module=lambda *a, **k: (_ for _ in ()).throw(SystemExit(7)); "
            f"sys.argv=[{str(launcher)!r},'audit','--help']; "
            + (
                f"runpy.run_path({str(launcher)!r},run_name='__main__')"
                if direct
                else f"sys.path.insert(0,{str(REPOSITORY)!r}); "
                "from LLM.__main__ import main; main()"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 7, result.stdout + result.stderr


def test_cli_exports_package_path_to_nested_workers(tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, runpy, sys; "
            f"sys.path.insert(0,{str(REPOSITORY)!r}); "
            "from LLM.__main__ import main; "
            "runpy.run_module=lambda *a, **k: None; main(['audit']); "
            "import subprocess; "
            "result=subprocess.run([sys.executable,'-c','from LLM.core.io import file_sha256'],"
            "capture_output=True,text=True); assert result.returncode==0,result.stderr",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
