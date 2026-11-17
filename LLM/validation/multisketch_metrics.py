"""Count exact schedule candidates without confusing sketches with workloads."""

from collections import defaultdict

from LLM.validation.candidate_metrics import aggregate, summarize_workload


def summarize_multisketch(cases, modes, required=4):
    """Recompute case and workload results from attempts, independently of cached summaries.

    A case is one distinct normalized input sketch. Exact final candidate identities
    are deduplicated within each mode/workload, including across its sketches.
    ``all_sketches_k_reached`` requires every case to pass; pooled success cannot
    substitute for a failed sketch. A workload with only one sketch is explicitly
    excluded from the multi-sketch coverage denominator.
    """
    cases, modes = list(cases), list(modes)
    if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
        raise ValueError("required must be a positive integer")
    if not cases or not modes or len(set(modes)) != len(modes):
        raise ValueError("Nonempty cases and distinct nonempty modes are required")

    groups = defaultdict(list)
    case_rows = []
    sketch_ids = set()
    for case in cases:
        key = (case["workload_index"], case["sketch_id"])
        if key in sketch_ids:
            raise ValueError("Each workload must contain distinct sketch identities")
        sketch_ids.add(key)
        row = {
            **case,
            "modes": {mode: summarize_workload(case["attempts"], mode, required) for mode in modes},
        }
        case_rows.append(row)
        groups[case["workload_index"]].append(row)

    per_case = aggregate(case_rows, modes)
    for summary in per_case.values():
        summary["case_count"] = summary.pop("workloads")
        summary["k_reached_cases"] = summary.pop("k_reached_workloads")
        summary["unique_correct_case_macro"] = summary.pop("unique_correct_macro")

    per_workload = []
    for index, rows in sorted(groups.items()):
        if (
            len({row["workload_key"] for row in rows}) != 1
            or len({row["family"] for row in rows}) != 1
        ):
            raise ValueError("Cases grouped as one workload must have matching metadata")
        attempts = [attempt for row in rows for attempt in row["attempts"]]
        workload = {
            "workload_index": index,
            "workload_key": rows[0]["workload_key"],
            "family": rows[0]["family"],
            "sketch_count": len(rows),
            "multi_sketch": len(rows) > 1,
            "case_indexes": [row["index"] for row in rows],
            "modes": {},
        }
        for mode in modes:
            pooled = summarize_workload(attempts, mode, required)
            sketches = [row["modes"][mode] for row in rows]
            workload["modes"][mode] = {
                **pooled,
                "all_sketches_k_reached": all(row["k_reached"] for row in sketches),
                "any_sketches_k_reached": any(row["k_reached"] for row in sketches),
                "pooled_k_reached": pooled["k_reached"],
                "sketch_shortfall": sum(row["shortfall"] for row in sketches),
                "min_sketch_unique_correct": min(row["unique_correct"] for row in sketches),
                "max_sketch_unique_correct": max(row["unique_correct"] for row in sketches),
            }
        per_workload.append(workload)

    workload_count = len(per_workload)
    multi_rows = [row for row in per_workload if row["multi_sketch"]]
    workload_summary = aggregate(per_workload, modes)
    for mode, summary in workload_summary.items():
        summary["workload_count"] = summary.pop("workloads")
        summary["pooled_k_reached_workloads"] = summary.pop("k_reached_workloads")
        summary["pooled_k_reached_rate"] = summary.pop("k_reached_rate")
        summary["unique_correct_workload_macro"] = summary.pop("unique_correct_macro")
        summary["multi_sketch_workload_count"] = len(multi_rows)
        for field in ("all_sketches_k_reached", "any_sketches_k_reached"):
            count = sum(row["modes"][mode][field] for row in per_workload)
            summary[f"{field}_workloads"] = count
            summary[f"{field}_rate"] = count / workload_count
        multi_success = sum(row["modes"][mode]["all_sketches_k_reached"] for row in multi_rows)
        summary["all_sketches_k_reached_multi_sketch_workloads"] = multi_success
        summary["all_sketches_k_reached_multi_sketch_rate"] = (
            multi_success / len(multi_rows) if multi_rows else None
        )
        summary["sketch_shortfall"] = sum(
            row["modes"][mode]["sketch_shortfall"] for row in per_workload
        )

    return {
        "required": required,
        "case_count": len(cases),
        "workload_count": workload_count,
        "multi_sketch_workload_count": len(multi_rows),
        "per_case": per_case,
        "per_workload": per_workload,
        "workload_summary": workload_summary,
    }
