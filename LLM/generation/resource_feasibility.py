"""Lower an independent PPT completion before admitting its split factors.

The preview supplies only the standard tiled-root bindings already required by
the policy. It neither changes the caller's state nor repairs a rejected state.
Later cooperative-fetch and unroll decisions still need the final resource gate.
"""

import json
import math


def _hardware_limits(hardware):
    return {
        "max_shared_memory_per_block": int(hardware.max_shared_memory_per_block),
        "max_local_memory_per_block": int(hardware.max_local_memory_per_block),
        "max_threads_per_block": int(hardware.max_threads_per_block),
        "max_vector_bytes": int(hardware.vector_unit_bytes),
        "max_vthread": int(hardware.max_vthread_extent),
    }


def _completed_steps(steps, assigned):
    """Require an entire PPT completion; an unknown factor is never guessed."""
    completed = json.loads(json.dumps(steps))
    markers = [step[0] for step in completed if step[0] in {"PPT", "TBS", "PRS"}]
    if markers != ["PPT"] or not completed or completed[-1][0] != "PPT":
        raise ValueError("Resource preview requires the incomplete PPT state")
    required = {
        (source, factor)
        for source, step in enumerate(completed)
        if step[0] == "SP"
        for factor in range(len(step[4]))
    }
    if set(assigned) != required:
        raise ValueError("Resource preview requires exactly all PPT split factors")
    for source, step in enumerate(completed):
        if step[0] != "SP":
            continue
        factors = [assigned[source, i] for i in range(len(step[4]))]
        if any(isinstance(value, bool) or int(value) != value or value <= 0 for value in factors):
            raise ValueError("Resource preview factors must be positive integers")
        factors = list(map(int, factors))
        extent = int(step[3] or 0)
        product = math.prod(factors)
        if extent <= 0 or product > extent or extent % product:
            raise ValueError("Resource preview factors must divide their split extent")
        step[4] = factors
    return completed


def _bind_tiled_roots(state, groups):
    """Match InitThreadBind's three outer spatial levels, on a private state."""
    seen = set()
    for group in groups:
        stage_id = int(group["stage"])
        if stage_id in seen:
            raise ValueError("Resource preview received a duplicate tiled root")
        seen.add(stage_id)
        stage = state.stages[stage_id]
        count = len(stage.op.axis)
        if int(stage.compute_at) != 0 or not count or len(stage.iters) != count * 4:
            raise ValueError("Resource preview requires a standard tiled root")
        if any(
            not str(stage.iters[level * count + i].name).endswith(f".{level}")
            for level in range(3)
            for i in range(count)
        ):
            raise ValueError("Resource preview encountered unsupported tiled root geometry")
        for offset, binding in ((0, "blockIdx.x"), (1, "vthread"), (2, "threadIdx.x")):
            axes = list(state.stages[stage_id].iters[offset : offset + count])
            state.bind(stage_id, state.fuse(stage_id, axes), binding)


def _shared_allocations(module, tvm):
    allocations = []
    for name, function in module.functions.items():
        if not isinstance(function, tvm.tir.PrimFunc):
            continue

        def visit(node):
            if not isinstance(node, tvm.tir.Allocate):
                return
            annotation = node.buffer_var.type_annotation
            if annotation.storage_scope != "shared":
                return
            dtype = tvm.DataType(node.dtype)
            allocations.append(
                {
                    "function": str(name.name_hint),
                    "buffer": str(node.buffer_var),
                    "bytes": math.prod(int(value) for value in node.extents)
                    * ((int(dtype.bits) * int(dtype.lanes) + 7) // 8),
                }
            )

        tvm.tir.stmt_functor.post_order_visit(function.body, visit)
    return sorted(allocations, key=lambda row: (row["function"], row["buffer"]))


def preview_ppt_resources(task, state, steps, groups, assigned):
    """Return actual lowering/VerifyGPUCode evidence for one full PPT completion.

    ``assigned`` maps ``(original_split_step_id, factor_index)`` to an integer.
    The returned ``passed`` describes this preview only, never final build or
    correctness. TVM and unsupported-input failures are explicit failed probes.
    """
    import tvm
    from tvm import auto_scheduler

    report = {"passed": False, "stage": "input", "shared_allocations": []}
    try:
        if task.target.kind.name != "cuda":
            raise ValueError("Resource preview supports CUDA targets only")
        report["limits"] = _hardware_limits(task.hardware_params)
        completed = _completed_steps(steps, assigned)
        serialized = json.loads(auto_scheduler.MeasureInput(task, state).to_json())
        if serialized[1][1] != steps:
            raise ValueError("Resource preview steps differ from the supplied state")
        serialized[1][1] = completed
        report["stage"] = "rebuild"
        # Deserialization creates empty stages; InferBound replays concrete split
        # factors into an independent state, without editing the original DAG.
        inp = tvm.get_global_func("auto_scheduler.DeserializeMeasureInput")(json.dumps(serialized))
        rebuilt = task.compute_dag.infer_bound_from_state(inp.state)
        report["stage"] = "bind_tiled_roots"
        _bind_tiled_roots(rebuilt, groups)
        report["bound_stages"] = [int(group["stage"]) for group in groups]
        report["stage"] = "lower"
        schedule, tensors = task.compute_dag.apply_steps_from_state(
            rebuilt, layout_rewrite=task.layout_rewrite_option
        )
        with task.target:
            module = tvm.lower(schedule, tensors)
            report["shared_allocations"] = _shared_allocations(module, tvm)
            report["stage"] = "verify_gpu_code"
            tvm.get_global_func("tir.transform.VerifyGPUCode")(report["limits"])(module)
        report["passed"] = True
        report["stage"] = "complete"
    except (tvm.TVMError, ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    return report
