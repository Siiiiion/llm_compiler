#!/usr/bin/env python3
"""Re-encode existing raw schedule JSONL without truncating decisions or changing splits."""

import argparse
import json
from pathlib import Path

from LLM.data.make_dataset_utils import DEFAULT_CLM_MAX_LENGTH, make_dataset
from LLM.core.io import atomic_json, file_sha256


def verify_build_audit(report_path, train_file, validation_file, allow_partial=False):
    from LLM.data.prepare_audited_clm import load_manifest

    report = json.loads(Path(report_path).read_text())
    manifest = load_manifest(report["audit_dir"], allow_partial)
    if (
        report["manifest_id"] != manifest["manifest_id"]
        or report["compiler_fingerprint"] != manifest["compiler_fingerprint"]
    ):
        raise ValueError("Raw provenance differs from the completed build audit")
    for split, filename in (("train", train_file), ("validation", validation_file)):
        if not filename or file_sha256(filename) != report["outputs"][split]:
            raise ValueError(f"Raw {split} file differs from the audited export")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, help="Existing train_raw.json")
    parser.add_argument(
        "--validation-file", help="Existing validation_raw.json; preserves the split"
    )
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True, help="New, empty dataset directory")
    parser.add_argument("--max-length", type=int, default=DEFAULT_CLM_MAX_LENGTH)
    parser.add_argument("--overlength-policy", choices=("error", "drop"), default="error")
    parser.add_argument("--ppt-marker", default="PPT")
    parser.add_argument(
        "--build-audit-manifest", help="build_audit_manifest.json from audited raw export"
    )
    parser.add_argument("--allow-partial-audit", action="store_true", help="Diagnostic runs only")
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    for path in (args.train_file, args.validation_file):
        if path and not Path(path).is_file():
            parser.error(f"Raw input file does not exist: {path}")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(
            "--output-dir must be a new or empty directory; old data will not be overwritten"
        )

    audit_report = None
    if args.build_audit_manifest:
        audit_report = verify_build_audit(
            args.build_audit_manifest,
            args.train_file,
            args.validation_file,
            args.allow_partial_audit,
        )
    stats = make_dataset(
        args.train_file,
        str(output),
        args.tokenizer_path,
        "clm",
        valid_percentage=0,
        max_length=args.max_length,
        validation_file=args.validation_file,
        ppt_marker=args.ppt_marker,
        overlength_policy=args.overlength_policy,
    )
    if audit_report is not None:
        verify_build_audit(
            args.build_audit_manifest,
            args.train_file,
            args.validation_file,
            args.allow_partial_audit,
        )
        atomic_json(output / "build_audit_manifest.json", audit_report)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "max_length": stats["max_length"],
                "overlength_policy": stats["overlength_policy"],
                "splits": {
                    name: {
                        key: values[key]
                        for key in ("input_records", "saved_records", "overlength_records")
                    }
                    for name, values in stats["splits"].items()
                },
                "stats_path": str(output / "tokenization_stats.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
