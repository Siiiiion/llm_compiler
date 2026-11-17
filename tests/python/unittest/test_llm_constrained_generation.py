"""Decision legality, token boundaries, and termination of the CUDA grammar."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from LLM.generation.constrained_generation import (
    CompletionPool,
    ConstraintError,
    DecisionPlan,
    TokenGrammar,
)
from LLM.generation.constrained_generation import resource_groups, resource_plan
from LLM.generation.constrained_generation import generate_constrained
from LLM.generation.constrained_generation import verify_gpu_resources


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


def test_resource_verifier_checks_real_lowered_threads_and_shared_memory():
    import tvm
    from tvm import te

    source = te.placeholder((64,), name="source")
    output = te.compute((64,), lambda i: source[i] + 1, name="output")
    schedule = te.create_schedule(output.op)
    shared = schedule.cache_read(source, "shared", [output])
    block, thread = schedule[output].split(output.op.axis[0], factor=64)
    schedule[output].bind(block, te.thread_axis("blockIdx.x"))
    schedule[output].bind(thread, te.thread_axis("threadIdx.x"))
    schedule[shared].compute_at(schedule[output], block)
    hardware = SimpleNamespace(
        max_shared_memory_per_block=256,
        max_local_memory_per_block=32768,
        max_threads_per_block=64,
        vector_unit_bytes=16,
        max_vthread_extent=8,
    )
    task = SimpleNamespace(
        target=tvm.target.Target("cuda -arch=sm_89"),
        hardware_params=hardware,
        layout_rewrite_option=0,
        compute_dag=SimpleNamespace(
            apply_steps_from_state=lambda *args, **kwargs: (schedule, [source, output])
        ),
    )
    verify_gpu_resources(task, None)
    hardware.max_threads_per_block = 32
    with pytest.raises(tvm.TVMError, match="threads per block"):
        verify_gpu_resources(task, None)
    hardware.max_threads_per_block = 64
    hardware.max_shared_memory_per_block = 128
    with pytest.raises(tvm.TVMError, match="shared memory"):
        verify_gpu_resources(task, None)


def consume(grammar, text):
    ids = list(grammar.ids)
    for token in map(ord, text):
        assert token in grammar.advance(ids)
        ids.append(token)
    grammar.advance(ids)
    return ids


def test_prefix_numbers_are_not_committed_until_the_separator():
    plan = DecisionPlan([["SP", 2, 5, 16, [1], 1], ["PPT"]])
    grammar = TokenGrammar(CharacterTokenizer(), "PPT", list(map(ord, "PPT")), plan)
    ids = consume(grammar, " SPC 2 5 16 1")
    assert grammar.assigned == {}
    assert ord("6") in grammar.advance(ids)
    assert ord(" ") in grammar.advance(ids)
    ids.append(ord("6"))
    ids.append(ord(" "))
    grammar.advance(ids)
    assert grammar.assigned == {(0, 0): 16}
    for token in map(ord, "1 "):
        assert token in grammar.advance(ids)
        ids.append(token)
    grammar.advance(ids)
    assert grammar.complete


def test_factor_products_divide_extent_and_never_emit_zero():
    plan = DecisionPlan([["SP", 2, 5, 24, [1, 1], 1], ["PPT"]])
    options = plan.choices(("factor", (0, 1)), {(0, 0): 4})
    assert [value for _, value in options] == [1, 2, 3, 6]
    assert all(24 % (4 * value) == 0 for _, value in options)


def test_warp_thread_and_vthread_limits_constrain_choices_jointly():
    steps = [["SP", 2, 0, 4, [1, 1, 1, 1], 1], ["SP", 2, 5, 512, [1, 1, 1, 1], 1], ["PPT"]]
    group = {"sources": (0, 1), "min_threads": 32, "max_threads": 1024, "max_vthread": 8}
    plan = DecisionPlan(steps, [group])
    options = plan.choices(("factor", (1, 1)), {(0, 0): 1, (0, 1): 1, (1, 0): 1})
    assert min(value for _, value in options) == 32
    assert not plan.feasible({(0, 0): 4, (1, 0): 4})
    assert not plan.feasible({(0, 1): 4, (1, 1): 512})
    assert plan.feasible({(0, 1): 1, (1, 1): 32})


def test_small_roots_allow_less_than_a_warp():
    steps = [["SP", 0, 0, 8, [1, 1, 1, 1], 1], ["FSP", 1, 0, 0, 3], ["PPT"]]
    stage = SimpleNamespace(
        compute_at=0,
        op_type=1,
        op=SimpleNamespace(axis=[SimpleNamespace(dom=SimpleNamespace(extent=8))]),
        iters=[SimpleNamespace(name=f"x.{i}", annotation=0) for i in range(4)],
    )
    state = SimpleNamespace(stages=[SimpleNamespace(compute_at=1), stage])
    hardware = SimpleNamespace(warp_size=32, max_threads_per_block=1024, max_vthread_extent=8)
    groups = resource_groups(steps, state, hardware)
    assert groups[0]["min_threads"] == 1
    assert groups[0]["sources"] == (0,)


def test_prs_choices_are_finite_and_follow_current_stage_order():
    plan = DecisionPlan([["PPT"], ["PRS", [4, 8, 11]]])
    assert [n for _, n in plan.choices(("unroll", None), {})] == [0, 16, 64, 512, 1024]
    grammar = TokenGrammar(CharacterTokenizer(), "PRS 4 8 11", list(map(ord, "PRS 4 8 11")), plan)
    text = " PR 4 0 auto_unroll_max_step$16 PR 8 0 auto_unroll_max_step$0 PR 11 0 auto_unroll_max_step$1024"
    consume(grammar, text)
    assert grammar.complete


def test_partial_stage_is_not_complete_and_wrong_fixed_fields_are_masked():
    plan = DecisionPlan([["SP", 2, 5, 16, [1], 1], ["PPT"]])
    grammar = TokenGrammar(CharacterTokenizer(), "PPT", list(map(ord, "PPT")), plan)
    ids = consume(grammar, " SPC 2 ")
    assert not grammar.complete
    assert grammar.advance(ids) == [ord("5")]


def test_unstable_tokenizer_boundary_is_rejected():
    class UnstableTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1] * len(text)}

    plan = DecisionPlan([["SP", 0, 0, 1, [1], 1], ["PPT"]])
    grammar = TokenGrammar(UnstableTokenizer(), "PPT", [9], plan)
    with pytest.raises(ConstraintError, match="boundary"):
        grammar.advance([9])


def test_tbs_does_not_reemit_ppt_splits():
    steps = [
        ["SP", 2, 5, 512, [1, 1], 1],
        ["PPT"],
        ["SPC", 2, 5, 512, [4, 8], 1],
        ["SP", 1, 0, 16, [1], 1],
        ["TBS"],
    ]
    plan = DecisionPlan(steps)
    assert list(plan.splits) == [3]


def test_checked_completion_pool_keeps_each_prefix_inside_verified_completions():
    plan = DecisionPlan([["SP", 0, 0, 16, [1, 1], 1], ["PPT"]])
    accepted = {(1, 2), (2, 4), (4, 2)}
    checked = []

    def check(assigned):
        factors = (assigned[0, 0], assigned[0, 1])
        checked.append(factors)
        return {"passed": factors in accepted}

    pool = CompletionPool(plan, check, proposals=1000, capacity=len(accepted))
    assert len(pool.assignments) == len(accepted)
    assert len(checked) == len(set(checked))
    assert set(checked).issuperset(accepted)

    def walk(assigned, index=0):
        if index == 2:
            assert (assigned[0, 0], assigned[0, 1]) in accepted
            return
        key = (0, index)
        options = pool.choices(("factor", key), assigned)
        assert options
        for _, value in options:
            walk({**assigned, key: value}, index + 1)

    walk({})
    assert pool.choices(("fixed", " SPC 0 0 16 "), {}) == [(" SPC 0 0 16 ", None)]


def test_checked_completion_pool_stops_at_capacity_and_rejects_empty_search():
    plan = DecisionPlan([["SP", 0, 0, 16, [1], 1], ["PPT"]])
    checked = []

    def check(assigned):
        checked.append(dict(assigned))
        return {"passed": True}

    pool = CompletionPool(plan, check, proposals=1000, capacity=2)
    assert len(pool.assignments) == len(checked) == pool.proposals == 2
    checked.clear()

    def reject(assigned):
        checked.append(dict(assigned))
        return {"passed": False, "reason": "shared memory exceeds limit"}

    with pytest.raises(ConstraintError, match="no checked completion"):
        CompletionPool(plan, reject, proposals=3, capacity=2)
    assert 1 <= len(checked) <= 3


@pytest.mark.parametrize(
    "dtype,expected_limit",
    [
        ("float32", 4),
        ("float16", 8),
        ("float64", 2),
        ("float64x2", 1),
        ("int8", 16),
        ("int8x4", 4),
    ],
)
def test_vector_factor_cap_keeps_dtype_byte_limit_and_requires_alignment(dtype, expected_limit):
    steps = [["PPT"], ["SP", 0, 0, 64, [1], 1], ["AN", 0, 1, 2], ["TBS"]]
    task = SimpleNamespace(hardware_params=SimpleNamespace(vector_unit_bytes=16))
    state = SimpleNamespace(
        stages=[
            SimpleNamespace(op=SimpleNamespace(output=lambda index: SimpleNamespace(dtype=dtype)))
        ]
    )
    plan, trace = DecisionPlan(steps), {}
    # The state_v1 plan retains its original domain until state_v2 adds the
    # resource policy; neither the byte cap nor alignment policy is global.
    assert max(value for _, value in plan.choices(("factor", (1, 0)), {})) == 64
    assert resource_plan(task, state, steps, plan, trace) is plan
    assert plan.factor_limits == {(1, 0): 1}
    assert [value for _, value in plan.choices(("factor", (1, 0)), {})] == [1]
    assert trace["vector_byte_factor_limits"] == {"1:0": expected_limit}
    assert trace["vector_factor_limits"] == {"1:0": 1}
    assert trace["vector_alignment_policy"] == "factor_1_until_base_and_index_alignment_proven"


def test_vector_cap_leaves_thread_binding_splits_available():
    steps = [["PPT"], ["SP", 0, 0, 64, [1], 1], ["AN", 0, 1, 6], ["TBS"]]
    task = SimpleNamespace(hardware_params=SimpleNamespace(vector_unit_bytes=16))
    plan = DecisionPlan(steps)
    resource_plan(task, SimpleNamespace(stages=[]), steps, plan, {})
    assert plan.factor_limits == {}
    assert max(value for _, value in plan.choices(("factor", (1, 0)), {})) == 64


@pytest.mark.parametrize("dtype,vector_bytes", [("int64x4", 16), ("float32", 0)])
def test_element_wider_than_vector_limit_has_no_fake_legal_factor(dtype, vector_bytes):
    steps = [["PPT"], ["SP", 0, 0, 64, [1], 1], ["AN", 0, 1, 2], ["TBS"]]
    task = SimpleNamespace(hardware_params=SimpleNamespace(vector_unit_bytes=vector_bytes))
    state = SimpleNamespace(
        stages=[
            SimpleNamespace(op=SimpleNamespace(output=lambda index: SimpleNamespace(dtype=dtype)))
        ]
    )
    with pytest.raises(ConstraintError, match="dtype exceeds the vector byte limit"):
        resource_plan(task, state, steps, DecisionPlan(steps), {})


@pytest.mark.parametrize("budget,complete", [(2, False), (80, True)])
def test_budget_exhaustion_rejects_partial_stage_and_complete_stage_stops(
    monkeypatch, budget, complete
):
    import json
    import torch
    from tvm import auto_scheduler

    steps = [["SP", 0, 0, 16, [1], 1], ["PPT"]]

    class Input:
        def __init__(self, *args):
            pass

        def to_json(self):
            return json.dumps([[], [[], steps]])

    monkeypatch.setattr(auto_scheduler, "MeasureInput", Input)

    class Tokenizer(CharacterTokenizer):
        model_max_length = 512

        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))

    class Model:
        generated = 0

        def generate(self, input_ids, attention_mask, **kwargs):
            assert kwargs["eos_token_id"] is None
            for _ in range(kwargs["max_new_tokens"]):
                legal = kwargs["prefix_allowed_tokens_fn"](0, input_ids[0])
                input_ids = torch.cat((input_ids, input_ids.new_tensor([[legal[0]]])), dim=1)
                self.generated += 1
                if kwargs["stopping_criteria"](input_ids, None).all():
                    break
            return input_ids

    task = SimpleNamespace(
        target=SimpleNamespace(kind=SimpleNamespace(name="cuda")),
        hardware_params=SimpleNamespace(max_threads_per_block=1024),
    )
    model, traces, telemetry = Model(), [], {}
    outputs = generate_constrained(
        task,
        [SimpleNamespace(stages=[])],
        ["PPT"],
        Tokenizer(),
        model,
        "cpu",
        {"max_new_tokens": budget},
        telemetry,
        traces,
    )
    assert traces[0]["complete"] == complete
    assert bool(outputs[0]) == complete
    assert telemetry["generated_tokens"] == model.generated
    if complete:
        assert model.generated < budget
        assert "failure" not in traces[0]
    else:
        assert model.generated == budget
        assert telemetry["constraint_rejected"] == 1


@pytest.mark.parametrize("constraint_mode", ["state_v1", "state_v2"])
def test_production_and_evaluation_route_the_same_constraint_settings(monkeypatch, constraint_mode):
    from LLM.generation import constrained_generation
    from LLM.validation import eval_struct
    from LLM.generation import gen_state
    from LLM.data import make_dataset

    monkeypatch.setattr(make_dataset, "input_to_tokens", lambda *args: ["PPT"])
    calls = []

    def constrained(*args, **kwargs):
        calls.append((args, kwargs))
        return [["SPC", "0", "0", "16", "1", "1"]]

    monkeypatch.setattr(constrained_generation, "generate_constrained", constrained)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9)
    kwargs = {
        "min_length": -1,
        "num_return_sequences": 1,
        "do_sample": False,
        "max_new_tokens": 80,
        "pad_token_id": 0,
        "eos_token_id": 9,
    }
    first = gen_state.gen_func(
        None, [1], None, tokenizer, None, "cpu", kwargs, False, 1, constraint_mode=constraint_mode
    )
    second = eval_struct._generate_decisions_for_states(
        None,
        [1],
        None,
        tokenizer,
        None,
        "cpu",
        80,
        False,
        0.95,
        50,
        0.7,
        1,
        False,
        constraint_mode=constraint_mode,
    )
    assert first == second
    assert calls[0] == calls[1]
    assert calls[0][1].get("constraint_mode", "state_v1") == constraint_mode
