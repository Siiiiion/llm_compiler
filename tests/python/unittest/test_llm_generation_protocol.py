"""Production/evaluation parity, sampling isolation, and candidate metric regressions."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from LLM.validation import eval_struct
from LLM.generation import gen_state
from LLM.generation.generation_protocol import isolated_generation_rng, resolve_max_new_tokens
from LLM.validation.generation_roundtrip_worker import StageProgress


class Tokenizer:
    model_max_length = 12
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, prompts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return {
            "input_ids": [[1, 2, 3] for _ in prompts],
            "attention_mask": [[1, 1, 1] for _ in prompts],
        }

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if i != self.eos_token_id)


class Model:
    def __init__(self):
        self.calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append((input_ids.tolist(), attention_mask.tolist(), kwargs))
        return torch.cat((input_ids, input_ids.new_tensor([[4, 5, 9]] * len(input_ids))), dim=1)


def test_production_and_evaluation_share_prompt_budget_and_decoding(monkeypatch):
    from LLM.data import make_dataset

    monkeypatch.setattr(make_dataset, "input_to_tokens", lambda *args: ["a PPT"])
    tokenizer, production, evaluation = Tokenizer(), Model(), Model()
    kwargs = {
        "min_length": -1,
        "num_return_sequences": 1,
        "do_sample": False,
        "max_new_tokens": 25,
        "pad_token_id": 0,
        "eos_token_id": 9,
    }
    actual = gen_state.gen_func(None, [1], None, tokenizer, production, "cpu", kwargs, False, 8)
    expected = eval_struct._generate_decisions_for_states(
        None, [1], None, tokenizer, evaluation, "cpu", 25, False, 0.95, 50, 0.7, 8, False
    )
    assert actual == expected == [["4", "5"]]
    assert production.calls == evaluation.calls
    assert production.calls[0][2]["max_new_tokens"] == 9
    assert kwargs["max_new_tokens"] == 25


def test_sampling_is_reproducible_and_restores_training_rng_on_error():
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    values = []
    for _ in range(2):
        with pytest.raises(RuntimeError), isolated_generation_rng(19):
            values.append((random.random(), np.random.rand(), torch.rand(1).item()))
            raise RuntimeError("evaluation failed")
    assert values[0] == values[1]
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_policy_stage_requests_control_budgets_and_macro_metrics(monkeypatch, tmp_path):
    calls = []

    class Policy:
        def __init__(self, task, **kwargs):
            self.task = task

        def gen_states(self, states, callback):
            callback(self.task, states, 10)
            callback(self.task, states, 100)
            return [1, 1, 2] if self.task == "first" else []

    class Input:
        def __init__(self, task, state):
            self.task, self.state = task, state

        def to_json(self):
            return str(self.state)

    class Builder:
        def __init__(self, **kwargs):
            pass

        def build(self, inputs, **kwargs):
            directory = tmp_path / "build"
            directory.mkdir()
            filename = directory / "module.tar"
            filename.write_bytes(b"module")
            return [
                SimpleNamespace(error_no=0, filename=str(filename)),
                SimpleNamespace(error_no=2),
            ]

    scheduler = SimpleNamespace(
        SketchPolicy=Policy, MeasureInput=Input, measure=SimpleNamespace(LocalBuilder=Builder)
    )
    monkeypatch.setattr(eval_struct, "_require_tvm", lambda: (None, scheduler))
    monkeypatch.setattr(
        eval_struct,
        "_generate_decisions_for_states",
        lambda *args, **kwargs: calls.append(kwargs["max_new_tokens"]) or [],
    )
    reports = []
    result = eval_struct.evaluate_structural_validity(
        None,
        None,
        [("a", [Input("first", 0)]), ("b", [Input("second", 0)])],
        "cpu",
        do_build=True,
        required_candidates=1,
        per_workload_reports=reports,
    )
    assert calls == [256, 400, 256, 400]
    assert result["unique_buildable_macro"] == 0.5
    assert result["k_attainment_rate"] == 0.5
    assert result["build_valid_rate"] == 0.5
    assert reports[0]["unique"] == 2
    assert reports[1]["built"] == 0


def test_stage_checkpoint_survives_a_failed_operation(tmp_path):
    report = {"sample_id": "sample"}
    progress = StageProgress(report, tmp_path / "progress.json")
    progress.enter("restore_reference_state")
    progress.enter("lower_generated_tir")
    checkpoint = json.loads((tmp_path / "progress.json").read_text())
    assert checkpoint["stage"] == "lower_generated_tir"
    assert checkpoint["stage_timings"][0]["seconds"] >= 0
    assert report["stage"] == checkpoint["stage"]


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_invalid_generation_scale_fails_early(scale):
    with pytest.raises(ValueError):
        resolve_max_new_tokens(10, 256, scale)


def test_empty_structural_evaluation_fails_instead_of_selecting_zero_score():
    with pytest.raises(ValueError, match="nonempty"):
        eval_struct.evaluate_structural_validity(None, None, [], "cpu")


def test_structural_callback_updates_selection_metric_and_restores_model(monkeypatch):
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(use_cache=False)
    callback = eval_struct.StructuralEvalCallback("sketches", "cuda", do_build=True)
    callback._tvm_ready = True
    callback._cached_workloads = [("workload", [object()])]

    def evaluate(**kwargs):
        assert not kwargs["model"].training
        assert kwargs["model"].config.use_cache
        return {"unique_buildable_macro": 0.75}

    monkeypatch.setattr(eval_struct, "evaluate_structural_validity", evaluate)
    metrics = {}
    callback.on_evaluate(
        SimpleNamespace(process_index=0),
        None,
        None,
        metrics=metrics,
        model=model,
        processing_class=object(),
    )
    assert metrics["eval_struct_unique_buildable_macro"] == 0.75
    assert model.training and model.config.use_cache is False

    def fail(**kwargs):
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(eval_struct, "evaluate_structural_validity", fail)
    with pytest.raises(RuntimeError, match="compiler failed"):
        callback.on_evaluate(
            SimpleNamespace(process_index=0),
            None,
            None,
            metrics={},
            model=model,
            processing_class=object(),
        )
    assert model.training and model.config.use_cache is False


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Requires Gloo")
def test_structural_metrics_and_errors_reach_all_training_ranks(tmp_path):
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            "--structural-distributed",
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "rank_0.json").read_text()) == json.loads(
        (tmp_path / "rank_1.json").read_text()
    )


def _structural_distributed_worker(directory):
    torch.distributed.init_process_group("gloo")
    rank = torch.distributed.get_rank()

    def evaluate():
        assert rank == 0, "Only rank zero should run structural generation"
        return {"unique_buildable_macro": 0.5, "k_attainment_rate": 0.25}

    metrics = eval_struct._evaluate_on_primary(evaluate)
    (directory / f"rank_{rank}.json").write_text(json.dumps(metrics))

    def fail():
        assert rank == 0
        raise ValueError("test compiler failure")

    with pytest.raises(RuntimeError, match="test compiler failure"):
        eval_struct._evaluate_on_primary(fail)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    assert sys.argv[1] == "--structural-distributed"
    _structural_distributed_worker(Path(sys.argv[2]))
