import json
import logging
import math
import os
from collections import Counter
from typing import List, Optional

from transformers import AutoTokenizer
from datasets import load_dataset
import pyarrow.compute as pc

from LLM.core.tokenization_protocol import (
    PPT_BOUNDARY_VERSION,
    build_ppt_marker_id_candidates,
    find_ppt_marker_end,
)


DEFAULT_CLM_MAX_LENGTH = 2048
CLM_INTEGRITY_COLUMNS = (
    "original_length",
    "was_truncated",
    "sequence_complete",
    "ppt_boundary_version",
)
logger = logging.getLogger(__name__)


def _resolve_max_length(tokenizer, max_length=None, default_max_length=544):
    if max_length is not None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        return max_length
    model_max_length = getattr(tokenizer, "model_max_length", None)
    # Qwen-like tokenizers may expose very large context windows (e.g. 131072),
    # but this pipeline expects fixed-size CLM training samples similar to legacy setup.
    if model_max_length is None or model_max_length > 4096:
        return default_max_length
    return model_max_length


def json_dfs_without_bracket(json, text_list: list):
    if isinstance(json, dict):
        json = dict(sorted(json.items()))
        for key, val in json.items():
            text_list.append(key)
            json_dfs_without_bracket(val, text_list)
    elif isinstance(json, (list, tuple)):
        for it in json:
            json_dfs_without_bracket(it, text_list)
    elif isinstance(json, (str, int, float, bool)):
        text_list.append(str(json))
    else:
        assert False


def json_dfs_with_bracket(json, text_list: list):
    if isinstance(json, dict):
        text_list.append("{")
        json = dict(sorted(json.items()))
        for idx, (key, val) in enumerate(json.items()):
            if idx != 0:
                text_list.append(",")
            text_list.append(key)
            json_dfs_with_bracket(val, text_list)
        text_list.append("}")
    elif isinstance(json, list):
        text_list.append("[")
        for idx, it in enumerate(json):
            if idx != 0:
                text_list.append(",")
            json_dfs_with_bracket(it, text_list)
        text_list.append("]")
    elif isinstance(json, (str, int, float, bool)):
        text_list.append(str(json))
    else:
        assert False


def json_to_token(json_lines):
    token_list = []
    for json_line in json_lines:
        text_list = []
        json_dfs_without_bracket(json_line["text"], text_list)
        json_line["text"] = " ".join(text_list)
        token_list.append(json_line)
    return token_list


def _load_raw_datasets(file, valid_percentage=5, validation_file=None, cache_dir=None):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    if validation_file is not None:
        data_files["validation"] = validation_file
        return load_dataset(
            extension,
            data_files=data_files,
            keep_in_memory=False,
            cache_dir=cache_dir,
        )

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        keep_in_memory=False,
        cache_dir=cache_dir,
    )
    if valid_percentage > 0:
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{valid_percentage}%]",
            keep_in_memory=False,
            cache_dir=cache_dir,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{valid_percentage}%:]",
            keep_in_memory=False,
            cache_dir=cache_dir,
        )
    else:
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split="train",
            keep_in_memory=False,
            cache_dir=cache_dir,
        )
    return raw_datasets


_find_ppt_end = find_ppt_marker_end


def _length_summary(counts):
    """Summarize exact length counts without materializing every sample length."""
    total = sum(counts.values())
    result = {"count": total, "min": min(counts, default=0), "max": max(counts, default=0)}
    cumulative = 0
    quantiles = [
        (name, math.ceil(total * q)) for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
    ]
    for length, count in sorted(counts.items()):
        cumulative += count
        for name, rank in quantiles:
            if name not in result and cumulative >= rank:
                result[name] = length
    for name, _ in quantiles:
        result.setdefault(name, 0)
    result["histogram"] = {str(length): count for length, count in sorted(counts.items())}
    return result


def _clm_length_stats(dataset, max_length):
    """Audit full sequences and workload coverage before applying the length limit."""
    columns = ["original_length"]
    columns += [
        name for name in ("ppt_end", "workload_key", "source_file") if name in dataset.column_names
    ]
    lengths, prompts, suffixes = Counter(), Counter(), Counter()
    workloads = {}
    overlength_count = 0
    missing_ppt_count = 0
    for batch in dataset.select_columns(columns).iter(batch_size=10000):
        lengths.update(batch["original_length"])
        for index, length in enumerate(batch["original_length"]):
            overlength = length > max_length
            overlength_count += int(overlength)
            if "ppt_end" in batch:
                end = batch["ppt_end"][index]
                if end < 0:
                    missing_ppt_count += 1
                else:
                    prompts[end + 1] += 1
                    suffixes[length - end - 1] += 1
            key = batch.get("workload_key", batch.get("source_file"))
            if key is None:
                continue
            key = key[index]
            stats = workloads.setdefault(
                key,
                {"records": 0, "overlength_records": 0, "within_limit_records": 0, "max_length": 0},
            )
            stats["records"] += 1
            stats["overlength_records"] += int(overlength)
            stats["within_limit_records"] += int(not overlength)
            stats["max_length"] = max(stats["max_length"], length)
    return {
        "input_records": len(dataset),
        "overlength_records": overlength_count,
        "within_limit_records": len(dataset) - overlength_count,
        "missing_ppt_records": missing_ppt_count,
        "original_length": _length_summary(lengths),
        "prompt_length": _length_summary(prompts),
        "suffix_length": _length_summary(suffixes),
        "workload_count": len(workloads),
        "workloads_without_records_within_limit": sorted(
            key for key, stats in workloads.items() if stats["within_limit_records"] == 0
        ),
        "workloads": workloads,
    }


def validate_clm_dataset_integrity(dataset, split_name, allow_legacy=False):
    """Reject known truncated sequences; require length provenance for new training runs."""
    required = set(CLM_INTEGRITY_COLUMNS) | {"length"}
    missing = required - set(dataset.column_names)
    # Even the legacy opt-in must not allow a dataset explicitly marked as truncated.
    for name, invalid_value in (("was_truncated", True), ("sequence_complete", False)):
        if name in dataset.column_names:
            values = dataset.data.column(name)
            if pc.any(pc.fill_null(pc.equal(values, invalid_value), True)).as_py():
                raise ValueError(f"CLM split={split_name} contains incomplete sequences ({name})")
    if missing:
        message = (
            f"CLM split={split_name} is missing sequence integrity columns: {sorted(missing)}. "
            "Rebuild from the untruncated raw JSON with python -m LLM rebuild-dataset; "
            "increasing the training context cannot recover an already truncated Arrow dataset."
        )
        if not allow_legacy:
            raise ValueError(
                message + " Use --allow_legacy_dataset=True only to reproduce old runs."
            )
        logger.warning(
            "%s Legacy dataset explicitly allowed; completeness cannot be verified.", message
        )
        return
    versions = dataset.data.column("ppt_boundary_version")
    if pc.any(pc.fill_null(pc.not_equal(versions, PPT_BOUNDARY_VERSION), True)).as_py():
        raise ValueError(
            f"CLM split={split_name} uses an unsupported PPT boundary protocol; "
            "rebuild it with python -m LLM rebuild-dataset"
        )
    actual_lengths = pc.list_value_length(dataset.data.column("input_ids"))
    original_lengths = dataset.data.column("original_length")
    for values in (actual_lengths, dataset.data.column("length")):
        mismatch = pc.fill_null(pc.not_equal(values, original_lengths), True)
        if pc.any(mismatch).as_py():
            raise ValueError(
                f"CLM split={split_name} contains changed or truncated sequence lengths"
            )
    label_lengths = pc.list_value_length(dataset.data.column("labels"))
    if pc.any(pc.fill_null(pc.not_equal(label_lengths, actual_lengths), True)).as_py():
        raise ValueError(
            f"CLM split={split_name} has labels that do not cover the full input sequence"
        )


def make_dataset(
    file,
    dataset_path,
    tokenizer_path,
    for_clm_or_mlm,
    valid_percentage=5,
    max_length=None,
    validation_file=None,
    ppt_marker: Optional[str] = "PPT",
    overlength_policy: str = "error",
):
    """Tokenize the raw JSON file into a HuggingFace dataset on disk.

    CLM sequences are encoded in full. Overlength samples cause an error by
    default, or are dropped whole with ``overlength_policy="drop"``. They are
    never shortened. The audit report includes full lengths and workload coverage.
    ``sequence_complete`` describes token preservation, not TVM build validity.
    """
    if overlength_policy not in ("error", "drop"):
        raise ValueError("overlength_policy must be 'error' or 'drop'")
    os.makedirs(dataset_path, exist_ok=True)
    cache_dir = os.path.join(dataset_path, ".tokenize_cache")
    raw_datasets = _load_raw_datasets(
        file,
        valid_percentage=valid_percentage,
        validation_file=validation_file,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    effective_max_length = _resolve_max_length(
        tokenizer,
        max_length=max_length,
        default_max_length=DEFAULT_CLM_MAX_LENGTH if for_clm_or_mlm == "clm" else 544,
    )

    ppt_patterns: List[List[int]] = []
    if for_clm_or_mlm == "clm" and ppt_marker:
        ppt_patterns = build_ppt_marker_id_candidates(tokenizer, ppt_marker)

    column_names = list(raw_datasets["train"].features)
    if for_clm_or_mlm == "clm":
        # Keep identities until the audit has recorded coverage. They are removed
        # before saving so Trainer/collators only receive numeric model metadata.
        column_names = [
            name for name in column_names if name not in ("workload_key", "source_file")
        ]

        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
            input_ids_batch = output["input_ids"]
            output["labels"] = [list(ids) for ids in input_ids_batch]
            output["length"] = [len(ids) for ids in input_ids_batch]
            output["original_length"] = list(output["length"])
            output["was_truncated"] = [False] * len(input_ids_batch)
            output["sequence_complete"] = [True] * len(input_ids_batch)
            output["ppt_boundary_version"] = [PPT_BOUNDARY_VERSION] * len(input_ids_batch)
            if ppt_patterns:
                output["ppt_end"] = [_find_ppt_end(ids, ppt_patterns) for ids in input_ids_batch]
            output.pop("token_type_ids", None)
            return output

    elif for_clm_or_mlm == "mlm":
        column_names.remove("labels")

        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=effective_max_length,
            )
            output["length"] = [len(ids) for ids in output["input_ids"]]
            output.pop("token_type_ids", None)
            return output

    else:
        assert False

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=False,
    )

    stats_payload = {
        "mode": for_clm_or_mlm,
        "ppt_boundary_version": PPT_BOUNDARY_VERSION if for_clm_or_mlm == "clm" else None,
        "train_file": os.path.abspath(file),
        "validation_file": os.path.abspath(validation_file) if validation_file else None,
        "tokenizer_path": (
            os.path.abspath(tokenizer_path) if os.path.isdir(tokenizer_path) else tokenizer_path
        ),
        "max_length": effective_max_length,
        "overlength_policy": overlength_policy,
        "truncation": for_clm_or_mlm != "clm",
        "status": "ready",
        "splits": {},
    }
    if for_clm_or_mlm == "clm":
        for split_name, dataset in tokenized_datasets.items():
            stats = _clm_length_stats(dataset, effective_max_length)
            stats_payload["splits"][split_name] = stats
        overlength_count = sum(s["overlength_records"] for s in stats_payload["splits"].values())
        errors = []
        if overlength_count and overlength_policy == "error":
            errors.append(
                f"{overlength_count} CLM samples exceed max_length={effective_max_length}. "
                "No sequences were truncated. Increase --max-length/--max_length or explicitly "
                "choose overlength_policy='drop' to discard whole samples."
            )
        for split_name, stats in stats_payload["splits"].items():
            if stats["within_limit_records"] == 0:
                errors.append(
                    f"CLM split={split_name} has no complete samples within the length limit"
                )
            stats["saved_records"] = 0 if errors else stats["within_limit_records"]
            stats["dropped_overlength_records"] = (
                stats["overlength_records"] if overlength_policy == "drop" else 0
            )
        if errors:
            stats_payload["status"] = "rejected"
            for stats in stats_payload["splits"].values():
                stats["saved_records"] = 0
        report_path = os.path.join(dataset_path, "tokenization_stats.json")
        with open(report_path, "w", encoding="utf-8") as file_obj:
            json.dump(stats_payload, file_obj, indent=2)
            file_obj.write("\n")
        if errors:
            raise ValueError(" ".join(errors) + f" See {report_path}.")
        if overlength_count:
            logger.warning("Dropping %d whole CLM sequences; see %s", overlength_count, report_path)
            tokenized_datasets = tokenized_datasets.filter(
                lambda length: length <= effective_max_length,
                input_columns=["original_length"],
                desc="Dropping whole overlength CLM sequences",
            )
        for split_name, dataset in tokenized_datasets.items():
            identity_columns = [
                name for name in ("workload_key", "source_file") if name in dataset.column_names
            ]
            tokenized_datasets[split_name] = dataset.remove_columns(identity_columns)
            validate_clm_dataset_integrity(tokenized_datasets[split_name], split_name)
    tokenized_datasets.save_to_disk(dataset_path)
    return stats_payload


def make_dataset_test(file, dataset_path, tokenizer_path, for_clm_or_mlm, max_length=None):
    data_files = {}
    data_files["train"] = file
    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, keep_in_memory=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    effective_max_length = _resolve_max_length(tokenizer, max_length=max_length)

    if for_clm_or_mlm == "clm":

        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=effective_max_length,
            )
            input_ids = output["input_ids"]
            for inp in input_ids:
                if len(inp) > 0:
                    del inp[-1]
            attention_mask = output["attention_mask"]
            for mask in attention_mask:
                if len(mask) > 0:
                    del mask[-1]
            output["length"] = [len(ids) for ids in output["input_ids"]]
            output.pop("token_type_ids", None)
            return output

    elif for_clm_or_mlm == "mlm":

        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=effective_max_length,
            )
            output["length"] = [len(ids) for ids in output["input_ids"]]
            output.pop("token_type_ids", None)
            return output

    else:
        assert False

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
        keep_in_memory=True,
    )

    tokenized_datasets.save_to_disk(dataset_path)
