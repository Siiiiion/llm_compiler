"""Verify audited raw/Arrow identity, complete sequences and sampled full token encodings."""

import json
import random
from collections import Counter
from pathlib import Path

from LLM.core.io import atomic_json
from LLM.core.schedule_record_utils import digest, workload_key


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_dataset(raw_dir, dataset_dir, tokenizer_path, max_length=2048):
    import pyarrow.compute as pc
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    from LLM.data.make_dataset_utils import validate_clm_dataset_integrity
    from LLM.data.prepare_audited_clm import load_manifest
    from LLM.data.rebuild_clm_dataset import verify_build_audit
    from LLM.core.tokenization_protocol import (
        PPT_BOUNDARY_VERSION,
        build_ppt_marker_id_candidates,
        find_ppt_marker_end,
    )

    raw_dir, dataset_dir = Path(raw_dir).resolve(), Path(dataset_dir).resolve()
    report = verify_build_audit(
        raw_dir / "build_audit_manifest.json",
        raw_dir / "train_raw.json",
        raw_dir / "validation_raw.json",
    )
    manifest = load_manifest(report["audit_dir"])
    require(
        json.loads((dataset_dir / "build_audit_manifest.json").read_text()) == report,
        "Arrow build audit report differs from raw export",
    )
    stats = json.loads((dataset_dir / "tokenization_stats.json").read_text())
    require(
        stats["status"] == "ready"
        and stats["max_length"] == max_length
        and stats["overlength_policy"] == "error"
        and stats["truncation"] is False
        and stats["ppt_boundary_version"] == PPT_BOUNDARY_VERSION,
        "Arrow tokenization settings do not match the complete-sequence protocol",
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    patterns = build_ppt_marker_id_candidates(tokenizer)
    datasets = load_from_disk(str(dataset_dir), keep_in_memory=False)
    require(set(datasets) == {"train", "validation"}, "Unexpected dataset splits")
    sources = {source["name"]: source for source in manifest["sources"]}
    holdouts = set(manifest["hold_out_keys"])
    all_workloads, details = {}, {}
    model_config = json.loads((Path(tokenizer_path) / "config.json").read_text())
    vocab_limit = min(len(tokenizer), int(model_config["vocab_size"]))

    for split, dataset in datasets.items():
        expected = report["counts"][split]
        audit = stats["splits"][split]
        require(
            len(dataset) == expected == audit["input_records"] == audit["saved_records"],
            f"{split}: raw/Arrow row count mismatch",
        )
        require(
            audit["overlength_records"] == audit["dropped_overlength_records"] == 0
            and audit["missing_ppt_records"] == 0
            and not audit["workloads_without_records_within_limit"],
            f"{split}: overlength, missing marker or lost coverage",
        )
        validate_clm_dataset_integrity(dataset, split)
        require(
            audit["original_length"]["max"] <= max_length,
            f"{split}: sequence exceeds the configured length",
        )
        # Arrow operations check every saved token and label without expanding the
        # complete dataset into Python lists.
        for start in range(0, len(dataset), 8192):
            table = dataset.data.slice(start, min(8192, len(dataset) - start))
            ids = pc.list_flatten(table.column("input_ids"))
            labels = pc.list_flatten(table.column("labels"))
            require(
                pc.all(pc.fill_null(pc.equal(ids, labels), False)).as_py() is True,
                f"{split}: saved labels differ from full input ids",
            )
            bounds = pc.min_max(ids).as_py()
            require(
                0 <= bounds["min"] <= bounds["max"] < vocab_limit,
                f"{split}: token id outside tokenizer/model vocabulary",
            )
            ends, lengths = table.column("ppt_end"), table.column("length")
            require(
                pc.all(pc.fill_null(pc.greater_equal(ends, 0), False)).as_py() is True
                and pc.all(pc.fill_null(pc.less(ends, pc.subtract(lengths, 1)), False)).as_py()
                is True,
                f"{split}: invalid PPT boundary or empty suffix",
            )
            require(
                pc.all(
                    pc.fill_null(
                        pc.equal(pc.list_flatten(table.column("attention_mask")), 1), False
                    )
                ).as_py()
                is True,
                f"{split}: unpadded dataset has an unexpected attention mask",
            )

        lengths = dataset.data.column("length").to_pylist()
        workloads, first, longest, reservoir = Counter(), {}, {}, []
        randomizer = random.Random(20261102)
        rows = 0
        for index, line in enumerate((raw_dir / f"{split}_raw.json").open()):
            row = json.loads(line)
            key = workload_key(row["workload_key"])
            source = sources.get(row["source_file"])
            require(
                source is not None
                and source["split"] == split
                and source["workload_key"] == key
                and key not in holdouts,
                f"{split}: raw source membership or hold-out mismatch",
            )
            require(
                row["build_audit_manifest_id"] == manifest["manifest_id"]
                and row["compiler_fingerprint"] == manifest["compiler_fingerprint"]
                and row["build_audit_key"]
                == digest([row["source_input_hash"], manifest["compiler_fingerprint"]]),
                f"{split}: raw build provenance mismatch",
            )
            require(index < len(lengths), f"{split}: raw has extra rows")
            workloads[key] += 1
            sample = (index, row["text"])
            first.setdefault(key, sample)
            if key not in longest or lengths[index] > lengths[longest[key][0]]:
                longest[key] = sample
            if len(reservoir) < 1024:
                reservoir.append(sample)
            else:
                selected = randomizer.randrange(index + 1)
                if selected < len(reservoir):
                    reservoir[selected] = sample
            rows += 1
        require(rows == expected, f"{split}: raw line count mismatch")
        canonical_workloads = {
            workload_key(key): values for key, values in audit["workloads"].items()
        }
        require(
            len(canonical_workloads) == len(audit["workloads"])
            and set(workloads) == set(canonical_workloads)
            and len(workloads) == audit["workload_count"]
            and all(
                canonical_workloads[key]["records"] == count for key, count in workloads.items()
            ),
            f"{split}: raw/Arrow workload coverage differs",
        )
        all_workloads[split] = set(workloads)
        samples = dict([*first.values(), *longest.values(), *reservoir])
        items = sorted(samples.items())
        for start in range(0, len(items), 256):
            batch = items[start : start + 256]
            encoded = tokenizer(
                [text for _, text in batch],
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            saved = dataset.select([index for index, _ in batch])
            for ids, actual in zip(encoded, saved):
                require(ids == actual["input_ids"], f"{split}: full raw token encoding differs")
                require(
                    find_ppt_marker_end(ids, patterns) == actual["ppt_end"],
                    f"{split}: PPT token boundary differs",
                )
        details[split] = {
            "records": expected,
            "workloads": len(workloads),
            "sampled_full_encodings": len(samples),
            "min_length": audit["original_length"]["min"],
            "max_length": audit["original_length"]["max"],
            "min_suffix_tokens": audit["suffix_length"]["min"],
            "workloads_without_raw_samples": sum(
                source["split"] == split and source["workload_key"] not in workloads
                for source in manifest["sources"]
            ),
        }
        print(f"[verify] {split}: {details[split]}", flush=True)
    require(
        not all_workloads["train"].intersection(all_workloads["validation"]),
        "Training and validation workload sets overlap",
    )
    result = {
        "status": "VERIFIED",
        "manifest_id": manifest["manifest_id"],
        "raw_hashes_verified": True,
        "all_saved_labels_and_token_ranges_verified": True,
        "sequence_integrity_and_split_isolation_verified": True,
        "sample_policy": "first_and_longest_per_workload_plus_seeded_reservoir_1024_per_split",
        "max_length": max_length,
        "splits": details,
        "runtime_correctness_checked": False,
        "generation_parser_roundtrip_checked": False,
    }
    atomic_json(dataset_dir / "verification.json", result)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    result = verify_dataset(args.raw_dir, args.dataset_dir, args.tokenizer_path, args.max_length)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
