"""Shared validation.candidate_metrics functions; import without CLI side effects."""

from collections import Counter


def summarize_workload(attempts, mode, required):
    rows = [row for row in attempts if row["mode"] == mode]
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    unique = {candidate["identity"]: candidate for candidate in candidates}
    correct = {key for key, candidate in unique.items() if candidate["check"].get("correct")}
    failures = Counter()
    for row in rows:
        if not row["parsed"]:
            rejected = [
                trace.get("failure")
                for call in row["callbacks"]
                for trace in call["constraint_traces"]
                if trace.get("failure")
            ]
            failures["constraint_rejected" if rejected else "no_final_state"] += 1
    for candidate in unique.values():
        check = candidate["check"]
        if check.get("timeout"):
            failures["check_timeout"] += 1
        elif not check.get("resource_valid"):
            failures["resource_rejected"] += 1
        elif not check.get("built"):
            failures["build_rejected"] += 1
        elif not check.get("correct"):
            failures["correctness_rejected"] += 1
    reached, seen = None, set()
    for index, row in enumerate(rows, 1):
        seen.update(c["identity"] for c in row["candidates"] if c["check"].get("correct"))
        if reached is None and len(seen) >= required:
            reached = index
    return {
        "attempts": len(rows),
        "parsed_attempts": sum(row["parsed"] > 0 for row in rows),
        "parsed_states": sum(row["parsed"] for row in rows),
        "unique_states": len(unique),
        "duplicate_states": len(candidates) - len(unique),
        "duplicate_rate_of_parsed": (len(candidates) - len(unique)) / max(len(candidates), 1),
        "resource_valid": sum(bool(c["check"].get("resource_valid")) for c in unique.values()),
        "unique_built": sum(bool(c["check"].get("built")) for c in unique.values()),
        "unique_correct": len(correct),
        "required": required,
        "shortfall": max(required - len(correct), 0),
        "k_reached": len(correct) >= required,
        "attempts_to_k": reached,
        "failures": dict(failures),
        "generated_tokens": sum(row["telemetry"].get("generated_tokens", 0) for row in rows),
        "token_budget": sum(row["telemetry"].get("token_budget", 0) for row in rows),
        "generation_seconds": sum(row["generation_seconds"] for row in rows),
    }


def aggregate(workloads, modes):
    result = {}
    for mode in modes:
        rows = [workload["modes"][mode] for workload in workloads]
        count = len(rows)
        fields = (
            "attempts",
            "parsed_attempts",
            "parsed_states",
            "unique_states",
            "duplicate_states",
            "resource_valid",
            "unique_built",
            "unique_correct",
            "shortfall",
            "generated_tokens",
            "token_budget",
            "generation_seconds",
        )
        sums = {key: sum(row[key] for row in rows) for key in fields}
        failures = Counter()
        for row in rows:
            failures.update(row["failures"])
        result[mode] = {
            "workloads": count,
            **sums,
            "k_reached_workloads": sum(row["k_reached"] for row in rows),
            "k_reached_rate": sum(row["k_reached"] for row in rows) / count,
            "unique_correct_macro": sums["unique_correct"] / count,
            "duplicate_rate_of_parsed": sums["duplicate_states"] / max(sums["parsed_states"], 1),
            "failures": dict(failures),
        }
    return result
