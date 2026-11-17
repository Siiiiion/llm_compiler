"""Optional CUDA decision grammar derived from the current policy state.

Fixed fields are forced during decoding, never repaired after generation. Factor
choices use a conservative divisor subset; CUDA also accepts padded tiles outside
this subset. Parsing, GPU resource verification, and building remain separate gates.
"""

import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import torch


class ConstraintError(ValueError):
    """A state, tokenizer boundary, or budget cannot satisfy the grammar."""


def verify_gpu_resources(task, state):
    """Check lowered allocations and bindings using this task's hardware limits."""
    import tvm

    hardware = task.hardware_params
    limits = {
        "max_shared_memory_per_block": int(hardware.max_shared_memory_per_block),
        "max_local_memory_per_block": int(hardware.max_local_memory_per_block),
        "max_threads_per_block": int(hardware.max_threads_per_block),
        "max_vector_bytes": int(hardware.vector_unit_bytes),
        "max_vthread": int(hardware.max_vthread_extent),
    }
    schedule, tensors = task.compute_dag.apply_steps_from_state(
        state, layout_rewrite=task.layout_rewrite_option
    )
    with task.target:
        module = tvm.lower(schedule, tensors)
        # This TVM build registers the C++ pass without a Python wrapper.
        tvm.get_global_func("tir.transform.VerifyGPUCode")(limits)(module)


@lru_cache(maxsize=4096)
def divisors(value):
    if value <= 0:
        raise ConstraintError("Split extent must be a positive constant")
    small = [i for i in range(1, math.isqrt(value) + 1) if value % i == 0]
    return tuple(sorted(set(small + [value // i for i in small])))


@dataclass(frozen=True)
class SplitSpec:
    step_id: int
    stage: int
    iterator: int
    extent: int
    factors: int
    direction: int


def resource_groups(steps, state, hardware):
    """Follow SP/FSP sources through stage insertion, checking actual root geometry.

    Only the standard three-level spatial root pattern is recognized. Unknown
    tiled root patterns fail explicitly instead of claiming resource coverage.
    Shared memory is checked after lowering, not estimated from factor products.
    """
    sources = {}
    for index, step in enumerate(steps):
        kind = step[0]
        if kind in {"CHR", "CHW", "RF"}:
            insertion = step[1] + (kind == "CHR")
            sources = {k + (k >= insertion): v for k, v in sources.items()}
        if kind in {"SP", "FSP"}:
            sources.setdefault(step[1], []).append(
                (
                    index if kind == "SP" else step[3],
                    len(step[4]) if kind == "SP" else step[4],
                    kind,
                )
            )
    groups = []
    for stage_id, stage in enumerate(state.stages):
        if int(stage.compute_at) != 0 or int(stage.op_type) == 0:
            continue
        if any(int(it.annotation) == 6 for it in stage.iters):
            continue
        axes = list(stage.op.axis)
        count = len(axes)
        iters = list(stage.iters)
        if not any(str(it.name).endswith(".2") for it in iters):
            continue
        follows = sources.get(stage_id, [])
        # Standard GPU root follows three spatial levels from four-factor splits.
        if (
            not count
            or len(iters) != count * 4
            or len(follows) != count
            or any(n != 3 or kind != "FSP" for _, n, kind in follows)
            or any(
                not str(iters[level * count + i].name).endswith(f".{level}")
                for level in range(3)
                for i in range(count)
            )
        ):
            raise ConstraintError(f"Unsupported tiled root geometry at stage {stage_id}")
        refs = tuple(source for source, _, _ in follows)
        if len(set(refs)) != len(refs) or any(
            steps[source][0] != "SP" or len(steps[source][4]) < 2 or int(steps[source][5]) != 1
            for source in refs
        ):
            raise ConstraintError("Unsupported spatial split dependencies")
        total = math.prod(int(axis.dom.extent) for axis in axes)
        groups.append(
            {
                "stage": stage_id,
                "sources": refs,
                "min_threads": int(hardware.warp_size) if total > hardware.warp_size * 2 else 1,
                "max_threads": int(hardware.max_threads_per_block),
                "max_vthread": int(hardware.max_vthread_extent),
            }
        )
    return groups


class DecisionPlan:
    def __init__(self, steps, groups=()):
        self.steps = steps
        self.groups = list(groups)
        self.factor_limits = {}
        markers = [(i, s) for i, s in enumerate(steps) if s[0] in {"PPT", "TBS", "PRS"}]
        if not markers:
            raise ConstraintError("Missing CUDA decision stage")
        _, marker = markers[-1]
        self.stage = marker[0]
        ppt = max(i for i, s in enumerate(steps) if s[0] == "PPT")
        indices = range(ppt) if self.stage == "PPT" else range(ppt + 1, len(steps))
        self.splits = (
            {
                i: SplitSpec(i, s[1], s[2], int(s[3] or 0), len(s[4]), int(s[5]))
                for i in indices
                for s in [steps[i]]
                if s[0] == "SP"
            }
            if self.stage != "PRS"
            else {}
        )
        for spec in self.splits.values():
            divisors(spec.extent)
        self.fields = []
        for spec in self.splits.values():
            self.fields.append(("fixed", f" SPC {spec.stage} {spec.iterator} {spec.extent} "))
            self.fields.extend(("factor", (spec.step_id, i)) for i in range(spec.factors))
            self.fields.append(("fixed", f"{spec.direction} "))
        if self.stage == "PRS":
            for stage in marker[1]:
                self.fields.append(("fixed", f" PR {int(stage)} 0"))
                self.fields.append(("unroll", None))

    def feasible(self, assigned):
        for group in self.groups:
            vthread = math.prod(assigned.get((source, 0), 1) for source in group["sources"])
            if vthread > group["max_vthread"]:
                return False
            products = {1}
            for source in group["sources"]:
                if (source, 1) in assigned:
                    choices = (assigned[source, 1],)
                else:
                    used = math.prod(v for (s, _), v in assigned.items() if s == source)
                    choices = divisors(self.splits[source].extent // used)
                products = {
                    a * b for a in products for b in choices if a * b <= group["max_threads"]
                }
            if not any(value >= group["min_threads"] for value in products):
                return False
        return True

    def choices(self, field, assigned):
        kind, value = field
        if kind == "fixed":
            return [(value, None)]
        if kind == "unroll":
            return [(f" auto_unroll_max_step${n}", n) for n in (0, 16, 64, 512, 1024)]
        source, _ = value
        used = math.prod(v for (s, _), v in assigned.items() if s == source)
        return [
            (f"{n} ", n)
            for n in divisors(self.splits[source].extent // used)
            if n <= self.factor_limits.get(value, self.splits[source].extent)
            and self.feasible({**assigned, value: n})
        ]


class CompletionPool:
    """A bounded conservative domain whose every prefix has a checked completion.

    The proposals are compiler probes, not model attempts or fallback schedules.
    No candidate is emitted without the model selecting its factors via the trie.
    """

    def __init__(self, plan, check, proposals=128, capacity=32):
        self.plan = plan
        self.assignments = []
        self.checks = []
        self.proposals = 0
        keys = [value for kind, value in plan.fields if kind == "factor"]
        rng = random.Random(0)
        seen = set()
        for _ in range(proposals):
            assigned = {}
            for key in keys:
                options = plan.choices(("factor", key), assigned)
                if not options:
                    break
                values = [value for _, value in options]
                assigned[key] = rng.choices(values, weights=[v**-0.5 for v in values])[0]
            if len(assigned) != len(keys):
                continue
            identity = tuple(assigned[k] for k in keys)
            if identity in seen:
                continue
            seen.add(identity)
            self.proposals += 1
            result = check(assigned)
            self.checks.append({"factors": list(identity), **result})
            if result["passed"]:
                self.assignments.append(assigned)
                if len(self.assignments) >= capacity:
                    break
        if not self.assignments:
            raise ConstraintError("Bounded resource search found no checked completion")

    def choices(self, field, assigned):
        options = self.plan.choices(field, assigned)
        if field[0] != "factor":
            return options
        matching = [
            row
            for row in self.assignments
            if all(row.get(key) == value for key, value in assigned.items())
        ]
        allowed = {row[field[1]] for row in matching}
        return [(text, value) for text, value in options if value in allowed]


_RESOURCE_POOLS = OrderedDict()


def resource_plan(task, state, steps, plan, trace):
    """Add conservative TBS vector caps or a checked PPT completion domain."""
    if plan.stage == "TBS":
        import tvm

        byte_factor_limits = {}
        for source, spec in plan.splits.items():
            following = steps[source + 1] if source + 1 < len(steps) else []
            if (
                spec.factors == 1
                and spec.direction == 1
                and following == ["AN", spec.stage, spec.iterator + 1, 2]
            ):
                dtype = tvm.DataType(state.stages[spec.stage].op.output(0).dtype)
                limit = (
                    int(task.hardware_params.vector_unit_bytes) * 8 // (dtype.bits * dtype.lanes)
                )
                if limit < 1:
                    raise ConstraintError("Element dtype exceeds the vector byte limit")
                byte_factor_limits[f"{source}:0"] = limit
                # A legal vector width does not prove alignment of its base or
                # index. In particular, scalar CUDA shared allocations can have
                # weaker alignment than a later vectorized store requires.
                plan.factor_limits[source, 0] = 1
        trace["vector_byte_factor_limits"] = byte_factor_limits
        trace["vector_factor_limits"] = {f"{s}:{i}": v for (s, i), v in plan.factor_limits.items()}
        trace["vector_alignment_policy"] = "factor_1_until_base_and_index_alignment_proven"
        return plan
    if plan.stage != "PPT" or not plan.splits:
        return plan
    from LLM.core.schedule_record_utils import digest
    from LLM.generation.resource_feasibility import preview_ppt_resources

    # Task/state serialization includes shapes, target, hardware and exact sketch steps.
    from tvm import auto_scheduler

    key = digest(json.loads(auto_scheduler.MeasureInput(task, state).to_json()))
    started = time.monotonic()
    hit = key in _RESOURCE_POOLS
    if not hit:
        pool = CompletionPool(
            plan, lambda assigned: preview_ppt_resources(task, state, steps, plan.groups, assigned)
        )
        _RESOURCE_POOLS[key] = pool
        if len(_RESOURCE_POOLS) > 32:
            _RESOURCE_POOLS.popitem(last=False)
    pool = _RESOURCE_POOLS[key]
    _RESOURCE_POOLS.move_to_end(key)
    trace["resource_pool"] = {
        "key": key,
        "cache_hit": hit,
        "seconds": time.monotonic() - started,
        "proposals": pool.proposals,
        "accepted": len(pool.assignments),
        "checks": pool.checks if not hit else [],
        "domain": "bounded_checked_completions_not_complete_legal_space",
    }
    # Keep the stage/fields interface used by TokenGrammar while narrowing choices.
    pool.fields, pool.stage = plan.fields, plan.stage
    return pool


class TokenGrammar:
    """One row's dynamic prefix trie; spaces disambiguate values such as 1/16."""

    def __init__(self, tokenizer, prompt, prompt_ids, plan):
        self.tokenizer, self.text, self.ids, self.plan = tokenizer, prompt, list(prompt_ids), plan
        self.start = len(self.ids)
        self.field = 0
        self.assigned = {}
        self.options = None
        self.complete = not plan.fields

    def advance(self, ids):
        ids = list(ids)
        while not self.complete:
            if self.options is None:
                self.options = []
                for text, value in self.plan.choices(self.plan.fields[self.field], self.assigned):
                    if self.text[-1:].isspace():
                        text = text.lstrip()
                    combined = self.tokenizer(self.text + text, add_special_tokens=False)[
                        "input_ids"
                    ]
                    if combined[: len(self.ids)] != self.ids or len(combined) <= len(self.ids):
                        raise ConstraintError("Tokenizer changes an established field boundary")
                    self.options.append((list(combined), text, value))
                if not self.options:
                    raise ConstraintError("No legal completion for current state")
            matched = next((opt for opt in self.options if ids[: len(opt[0])] == opt[0]), None)
            if matched is None:
                return sorted(
                    {
                        opt[0][len(ids)]
                        for opt in self.options
                        if len(opt[0]) > len(ids) and opt[0][: len(ids)] == ids
                    }
                )
            self.ids, text, value = matched
            kind, key = self.plan.fields[self.field]
            if kind == "factor":
                self.assigned[key] = value
            self.text += text
            self.field += 1
            self.options = None
            self.complete = self.field == len(self.plan.fields)
        return []


def generate_constrained(
    task,
    states,
    prompts,
    tokenizer,
    model,
    device,
    kwargs,
    telemetry=None,
    traces=None,
    constraint_mode="state_v1",
):
    from tvm import auto_scheduler
    from transformers import StoppingCriteria, StoppingCriteriaList

    if task.target.kind.name != "cuda":
        raise ValueError("State constrained generation currently supports CUDA only")
    if kwargs.get("num_beams", 1) != 1 or kwargs.get("num_return_sequences", 1) != 1:
        raise ValueError("State constrained generation requires one beam and one sequence")
    results = []
    for state, prompt in zip(states, prompts):
        trace = {"mode": constraint_mode, "prompt": prompt, "model_tokens": [], "complete": False}
        if traces is not None:
            traces.append(trace)
        try:
            steps = json.loads(auto_scheduler.MeasureInput(task, state).to_json())[1][1]
            marker = next(s[0] for s in reversed(steps) if s[0] in {"PPT", "TBS", "PRS"})
            groups = resource_groups(steps, state, task.hardware_params) if marker == "PPT" else []
            plan = DecisionPlan(steps, groups)
            if plan.stage == "TBS":
                # Non-tiled roots may request a fresh thread split. Keep its
                # factors within hardware bounds; final lowering checks FFS too.
                max_threads = int(task.hardware_params.max_threads_per_block)
                original_choices = plan.choices

                def bounded_choices(field, assigned):
                    options = original_choices(field, assigned)
                    if field[0] == "factor":
                        options = [(text, value) for text, value in options if value <= max_threads]
                    return options

                plan.choices = bounded_choices
            if constraint_mode == "state_v2":
                plan = resource_plan(task, state, steps, plan, trace)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            grammar = TokenGrammar(tokenizer, prompt, prompt_ids, plan)
            trace.update(stage=plan.stage, resource_groups=groups)
            if grammar.complete:
                results.append([])
                trace["complete"] = True
                continue
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            available = int(tokenizer.model_max_length) - len(prompt_ids)
            budget = min(int(kwargs["max_new_tokens"]), available)
            if budget <= 0:
                raise ConstraintError("Context budget exhausted")

            def allowed(batch_id, current_ids):
                legal = grammar.advance(current_ids.tolist())
                if not legal:
                    raise ConstraintError("No legal next token")
                return legal

            class StageComplete(StoppingCriteria):
                def __call__(self, input_ids, scores, **unused):
                    grammar.advance(input_ids[0].tolist())
                    return torch.tensor([grammar.complete], device=input_ids.device)

            generation_kwargs = dict(kwargs)
            generation_kwargs.update(
                max_new_tokens=budget,
                prefix_allowed_tokens_fn=allowed,
                stopping_criteria=StoppingCriteriaList([StageComplete()]),
                # Stage grammar owns termination; EOS may be a special field token.
                eos_token_id=None,
                forced_eos_token_id=None,
            )
            with torch.no_grad():
                output = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), **generation_kwargs
                )[0].tolist()
            grammar.advance(output)
            continuation = output[len(prompt_ids) :]
            trace.update(
                model_tokens=continuation,
                budget=budget,
                complete=grammar.complete,
                raw_continuation=tokenizer.decode(continuation, skip_special_tokens=False),
                free_factors={f"{s}:{i}": v for (s, i), v in grammar.assigned.items()},
            )
            if telemetry is not None:
                for key, value in {
                    "generation_calls": 1,
                    "token_budget": budget,
                    "generated_tokens": len(continuation),
                    "constraint_completed": int(grammar.complete),
                }.items():
                    telemetry[key] = telemetry.get(key, 0) + value
            if not grammar.complete:
                raise ConstraintError("Stage token budget exhausted before completion")
            # Do not skip special tokens: SPC and numeric values are added tokens.
            results.append(tokenizer.decode(continuation, skip_special_tokens=False).split())
        except ConstraintError as exc:
            trace["failure"] = str(exc)
            if telemetry is not None:
                telemetry["constraint_rejected"] = telemetry.get("constraint_rejected", 0) + 1
            results.append([])
    return results
