"""CPU lowering checks for independent, concrete PPT resource previews."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from LLM.generation.resource_feasibility import preview_ppt_resources


def make_preview(shared_limit=256, factors=(1, 8, 4)):
    import tvm
    from tvm import auto_scheduler, te

    source = te.placeholder((64,), name="preview_source")
    output = te.compute((64,), lambda i: source[i] + 1, name="preview_output")
    dag = auto_scheduler.ComputeDAG([source, output])
    hardware = auto_scheduler.HardwareParams(
        num_cores=-1,
        vector_unit_bytes=16,
        cache_line_bytes=64,
        max_shared_memory_per_block=shared_limit,
        max_local_memory_per_block=32768,
        max_threads_per_block=64,
        max_vthread_extent=8,
        warp_size=32,
    )
    task = auto_scheduler.SearchTask(
        workload_key=dag.workload_key(),
        compute_dag=dag,
        target="cuda -arch=sm_89",
        hardware_params=hardware,
        layout_rewrite_option=0,
    )
    state = dag.get_init_state()
    shared = state.cache_read(0, "shared", [1])
    loops = state.split(2, state.stages[2].iters[0], list(factors))
    state.compute_at(shared, 2, loops[0])
    serialized = json.loads(auto_scheduler.MeasureInput(task, state).to_json())
    serialized[1][1].append(["PPT"])
    inp = tvm.get_global_func("auto_scheduler.DeserializeMeasureInput")(json.dumps(serialized))
    state = dag.infer_bound_from_state(inp.state)
    steps = json.loads(auto_scheduler.MeasureInput(task, state).to_json())[1][1]
    assigned = {(1, i): value for i, value in enumerate(factors)}
    return task, state, steps, [{"stage": 2}], assigned


def state_json(task, state):
    from tvm import auto_scheduler

    return auto_scheduler.MeasureInput(task, state).to_json()


def test_preview_uses_assigned_factors_and_preserves_the_source_state():
    task, state, steps, groups, assigned = make_preview(shared_limit=64)
    before = state_json(task, state)
    rejected = preview_ppt_resources(task, state, steps, groups, assigned)
    assert not rejected["passed"]
    assert rejected["stage"] == "verify_gpu_code"
    assert "shared memory" in rejected["error"]
    assert [row["bytes"] for row in rejected["shared_allocations"]] == [128]
    assert rejected["limits"]["max_shared_memory_per_block"] == 64
    # Reduce the block's live tile through the supplied completion. The input
    # state still contains the original factors and must remain byte-identical.
    accepted = preview_ppt_resources(task, state, steps, groups, {(1, 0): 1, (1, 1): 8, (1, 2): 1})
    assert accepted["passed"]
    assert accepted["stage"] == "complete"
    assert [row["bytes"] for row in accepted["shared_allocations"]] == [32]
    assert state_json(task, state) == before
    assert steps[1][4] == [1, 8, 4]


@pytest.mark.parametrize(
    "assigned",
    [
        {(1, 0): 1, (1, 1): 8},
        {(1, 0): 1, (1, 1): 8, (1, 2): 3},
        {(1, 0): 1, (1, 1): 8, (1, 2): 0},
        {(1, 0): 1, (1, 1): 8, (1, 2): True},
    ],
)
def test_preview_rejects_incomplete_or_illegal_factor_assignments(assigned):
    task, state, steps, groups, _ = make_preview()
    result = preview_ppt_resources(task, state, steps, groups, assigned)
    assert not result["passed"]
    assert result["stage"] == "input"
    assert result["error_type"] == "ValueError"


def test_preview_rejects_mismatched_state_steps_and_unsupported_geometry():
    task, state, steps, groups, assigned = make_preview()
    mismatched = json.loads(json.dumps(steps))
    mismatched[1][4] = [1, 1, 1]
    result = preview_ppt_resources(task, state, mismatched, groups, assigned)
    assert not result["passed"]
    assert "differ" in result["error"]
    result = preview_ppt_resources(task, state, steps, [{"stage": 1}], assigned)
    assert not result["passed"]
    assert result["stage"] == "bind_tiled_roots"
    assert "standard tiled root" in result["error"]


def test_lowering_error_is_an_explicit_failed_probe(monkeypatch):
    import tvm

    task, state, steps, groups, assigned = make_preview()

    def fail(*args, **kwargs):
        raise tvm.TVMError("deliberate lowering failure")

    monkeypatch.setattr(tvm, "lower", fail)
    result = preview_ppt_resources(task, state, steps, groups, assigned)
    assert not result["passed"]
    assert result["stage"] == "lower"
    assert result["error_type"] == "TVMError"
    assert "deliberate lowering failure" in result["error"]
