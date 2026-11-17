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
"""Regression tests for LAPO's fixed weight scale and sample-level metrics."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from LLM.train.train_qwen3_dpo import DPOCollator, DPOTrainer, _lapo_weight_statistics


class _PreferenceModel(torch.nn.Module):
    """A real differentiable LM whose preference can be calculated exactly."""

    def __init__(self, strength=0.4):
        super().__init__()
        self.strength = torch.nn.Parameter(torch.tensor(strength))
        self.config = transformers.PretrainedConfig(vocab_size=4, pad_token_id=0)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        basis = self.strength.new_tensor([0.0, 1.0, -1.0, 0.0])
        logits = (self.strength * basis).expand(*input_ids.shape, 4)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1)
            )
        return SimpleNamespace(logits=logits, loss=loss)


def _pairs(weights, reverse_last=False):
    rows = []
    for index, weight in enumerate(weights):
        chosen, rejected = (2, 1) if reverse_last and index == len(weights) - 1 else (1, 2)
        rows.append(
            {
                "chosen_input_ids": [3, chosen],
                "chosen_labels": [-100, chosen],
                "rejected_input_ids": [3, rejected],
                "rejected_labels": [-100, rejected],
                "latency_weight": weight,
            }
        )
    return datasets.Dataset.from_list(rows)


def _trainer(tmp_path, train=None, evaluation=None, **kwargs):
    argument_overrides = kwargs.pop("argument_overrides", {})
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        save_strategy="no",
        **argument_overrides,
    )
    return DPOTrainer(
        model=_PreferenceModel(),
        ref_model=_PreferenceModel(0.0),
        args=args,
        train_dataset=train if train is not None else _pairs([1.0, 3.0]),
        eval_dataset=evaluation,
        data_collator=DPOCollator(SimpleNamespace(pad_token_id=0), pad_to_multiple_of=1),
        training_mode="lapo",
        **kwargs,
    )


def test_lapo_statistics_use_retained_training_pairs_after_clipping():
    train = _pairs([1.0, 3.0, 100.0]).select([0, 2])
    stats = _lapo_weight_statistics(train, 5.0)
    assert stats["num_samples"] == 2
    assert stats["raw_mean"] == 50.5
    assert stats["clipped_mean"] == 3.0
    assert stats["num_clipped"] == 1
    assert stats["dataset_fingerprint"] == train._fingerprint


@pytest.mark.parametrize("weights", [[float("nan")], [float("inf")], [-1.0]])
def test_invalid_lapo_weights_fail_early(weights):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _lapo_weight_statistics(_pairs(weights), 10.0)


@pytest.mark.parametrize("clip", [-1.0, float("nan"), float("inf")])
def test_invalid_lapo_clip_is_rejected(clip):
    with pytest.raises(ValueError, match="lapo_weight_clip"):
        _lapo_weight_statistics(_pairs([1.0]), clip)


def test_zero_mean_cannot_enable_lapo_normalization(tmp_path):
    with pytest.raises(ValueError, match="positive training weight mean"):
        _trainer(tmp_path, train=_pairs([0.0, 0.0]))


def test_batch_one_preserves_weighted_loss_and_gradient(tmp_path):
    trainer = _trainer(tmp_path)
    losses, gradients = [], []
    for row in trainer.train_dataset:
        trainer.model.zero_grad()
        loss = trainer.compute_loss(trainer.model, trainer.data_collator([row]))
        loss.backward()
        losses.append(loss.item())
        gradients.append(trainer.model.strength.grad.item())
    assert trainer.lapo_weight_mean == 2.0
    assert losses[1] / losses[0] == pytest.approx(3.0)
    assert gradients[1] / gradients[0] == pytest.approx(3.0)


@pytest.mark.parametrize("normalize,expected", [(True, [0.4, 1.6]), (False, [1.0, 4.0])])
def test_lapo_clips_before_fixed_normalization(tmp_path, normalize, expected):
    trainer = _trainer(
        tmp_path, train=_pairs([1.0, 100.0]), lapo_weight_clip=4.0, lapo_normalize=normalize
    )
    _, outputs = trainer.compute_loss(
        trainer.model, trainer.data_collator(list(trainer.train_dataset)), return_outputs=True
    )
    assert outputs["lapo/weight_mean"].tolist() == pytest.approx(expected)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_evaluation_aggregates_all_pairs_without_training_leakage(tmp_path, batch_size):
    trainer = _trainer(
        tmp_path,
        evaluation=_pairs([8.0, 8.0, 8.0], reverse_last=True),
        lapo_weight_clip=4.0,
        argument_overrides={"per_device_eval_batch_size": batch_size, "prediction_loss_only": True},
    )
    # A pending training metric must survive validation without contamination.
    trainer.compute_loss(trainer.model, trainer.data_collator([trainer.train_dataset[0]]))
    trainer.model.eval()
    expected_loss = (
        torch.stack(
            [
                trainer.compute_loss(trainer.model, trainer.data_collator([row]))
                for row in trainer.eval_dataset
            ]
        )
        .mean()
        .item()
    )
    metrics = trainer.evaluate()
    assert metrics["eval_reward/accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["eval_preference_samples"] == 3
    assert metrics["eval_loss"] == pytest.approx(expected_loss)
    assert metrics["eval_loss/dpo"] == pytest.approx(expected_loss)
    # Validation weights use training mean 2.0, never validation mean 4.0.
    assert metrics["eval_lapo/weight_mean"] == 2.0
    assert not any(key.startswith("train_") for key in metrics)
    again = trainer.evaluate(metric_key_prefix="validation")
    assert again["validation_preference_samples"] == 3
    assert again["validation_reward/accuracy"] == pytest.approx(2.0 / 3.0)
    trainer.log({"loss": 0.0})
    assert trainer.state.log_history[-1]["train_reward/accuracy"] == 1.0
    assert "eval_reward/accuracy" not in trainer.state.log_history[-1]


def test_dpo_evaluation_remains_unweighted(tmp_path):
    trainer = _trainer(tmp_path, evaluation=_pairs([100.0, 0.0], reverse_last=True))
    trainer.training_mode = "dpo"
    metrics = trainer.evaluate()
    assert metrics["eval_reward/accuracy"] == 0.5
    assert "eval_lapo/weight_mean" not in metrics


def test_equal_microbatches_and_accumulated_step_match_full_batch(tmp_path):
    strengths = []
    for batch_size, accumulation in [(2, 1), (1, 2)]:
        trainer = _trainer(
            tmp_path / str(batch_size),
            argument_overrides={
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": accumulation,
                "max_steps": 1,
                "learning_rate": 0.1,
                "optim": "sgd",
                "max_grad_norm": 0.0,
                "logging_steps": 1,
            },
        )
        assert not trainer.model_accepts_loss_kwargs
        trainer.train()
        strengths.append(trainer.model.strength.item())
        logged = [entry for entry in trainer.state.log_history if "train_reward/accuracy" in entry]
        assert len(logged) == 1
        assert logged[0]["train_lapo/weight_mean"] == pytest.approx(1.0)
    assert strengths[0] == pytest.approx(strengths[1], abs=1e-7)
    assert strengths[0] > 0.4


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="CPU DDP requires Gloo")
def test_distributed_evaluation_removes_padding_rows(tmp_path):
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        ACCELERATE_USE_CPU="true",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            "--distributed-eval",
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads((tmp_path / "distributed_metrics.json").read_text())
    assert metrics["eval_preference_samples"] == 5
    assert metrics["eval_reward/accuracy"] == pytest.approx(0.8)
    assert metrics["eval_lapo/weight_mean"] == 2.0


def _distributed_evaluation_worker(output_dir):
    trainer = _trainer(
        output_dir / "trainer",
        evaluation=_pairs([8.0] * 5, reverse_last=True),
        lapo_weight_clip=4.0,
        argument_overrides={
            "per_device_eval_batch_size": 2,
            "per_device_train_batch_size": 1,
            "max_steps": 1,
            "logging_steps": 1,
            "ddp_find_unused_parameters": False,
        },
    )
    trainer.model.eval()
    with torch.no_grad():
        expected_loss = (
            torch.stack(
                [
                    trainer.compute_loss(trainer.model, trainer.data_collator([row]))
                    for row in trainer.eval_dataset
                ]
            )
            .mean()
            .item()
        )
    metrics = trainer.evaluate()
    assert metrics["eval_loss"] == pytest.approx(expected_loss)
    assert metrics["eval_preference_samples"] == 5
    assert metrics["eval_reward/accuracy"] == pytest.approx(0.8)
    trainer.train()
    logged = [entry for entry in trainer.state.log_history if "train_lapo/weight_mean" in entry]
    assert logged[0]["train_lapo/weight_mean"] == pytest.approx(1.0)
    if trainer.is_world_process_zero():
        (output_dir / "distributed_metrics.json").write_text(json.dumps(metrics))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    assert sys.argv[1] == "--distributed-eval"
    _distributed_evaluation_worker(Path(sys.argv[2]))
