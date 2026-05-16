#!/usr/bin/env python3
"""Validate the prepared Head CT RSNA benchmark files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


QUESTION_NAMES = [
    "Is intra-cranial hemorrhage (ICH) present?",
    "Is intra-parenchymal hemorrhage (IPH) present?",
    "Is intra-ventricular hemorrhage (IVH) present?",
    "Is subarachnoid hemorrhage (SAH) present?",
    "Is subdural hemorrhage (SDH) present?",
    "Is epidural hemorrhage (EDH) present?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmarks/headct_rsna/data"),
        help="Directory containing train/valid/test JSONL, manifest.csv, and labels.json.",
    )
    parser.add_argument(
        "--check-all-paths",
        action="store_true",
        help="Check every manifest path for volume.mkv and metadata.json.",
    )
    parser.add_argument(
        "--sample-path-checks",
        type=int,
        default=50,
        help="Number of manifest paths to check when --check-all-paths is not set.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSONL row: {exc}") from exc
    return rows


def load_manifest(path: Path) -> dict[str, Path]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_name", "image_cache_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        return {row["sample_name"]: Path(row["image_cache_path"]) for row in reader}


def answer_to_int(answer: str) -> int:
    value = answer.lower()
    if value == "yes":
        return 1
    if value == "no":
        return 0
    raise ValueError(f"unexpected answer: {answer!r}")


def label_for_question(label_row: dict, question: str) -> int:
    qa_results = label_row.get("qa_results", {})
    for qa_list in qa_results.values():
        if not isinstance(qa_list, list):
            continue
        for item in qa_list:
            if isinstance(item, dict) and question in item:
                return answer_to_int(item[question])
    raise KeyError(question)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir

    required_files = [
        "train.json",
        "valid.json",
        "test.json",
        "manifest.csv",
        "labels.json",
    ]
    for filename in required_files:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

    split_rows = {
        "train": load_jsonl(data_dir / "train.json"),
        "valid": load_jsonl(data_dir / "valid.json"),
        "test": load_jsonl(data_dir / "test.json"),
    }
    manifest = load_manifest(data_dir / "manifest.csv")
    labels = json.loads((data_dir / "labels.json").read_text())

    all_samples = []
    for split, rows in split_rows.items():
        names = [row.get("sample_name") for row in rows]
        if len(names) != len(set(names)):
            duplicates = [name for name, count in Counter(names).items() if count > 1]
            raise ValueError(f"{split} has duplicate sample_name values: {duplicates[:10]}")
        all_samples.extend(names)

    if len(all_samples) != len(set(all_samples)):
        duplicates = [name for name, count in Counter(all_samples).items() if count > 1]
        raise ValueError(f"samples overlap across splits: {duplicates[:10]}")

    missing_manifest = sorted(set(all_samples) - set(manifest))
    if missing_manifest:
        raise ValueError(f"{len(missing_manifest)} samples missing from manifest: {missing_manifest[:10]}")

    missing_labels = sorted(set(all_samples) - set(labels))
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} samples missing from labels.json: {missing_labels[:10]}")

    paths_to_check = list(manifest.items())
    if not args.check_all_paths:
        paths_to_check = paths_to_check[: args.sample_path_checks]

    for sample_name, volume_dir in paths_to_check:
        if not volume_dir.exists():
            raise FileNotFoundError(f"{sample_name}: missing volume directory {volume_dir}")
        if not (volume_dir / "volume.mkv").exists():
            raise FileNotFoundError(f"{sample_name}: missing {volume_dir / 'volume.mkv'}")
        if not (volume_dir / "metadata.json").exists():
            raise FileNotFoundError(f"{sample_name}: missing {volume_dir / 'metadata.json'}")

    print(f"validated benchmark files under {data_dir}")
    print(f"manifest rows: {len(manifest)}")
    print(f"label rows: {len(labels)}")
    if args.check_all_paths:
        print(f"checked all {len(paths_to_check)} volume paths")
    else:
        print(f"checked first {len(paths_to_check)} volume paths")

    for split, rows in split_rows.items():
        print(f"{split}: {len(rows)} studies")
        for question in QUESTION_NAMES:
            positives = sum(label_for_question(labels[row["sample_name"]], question) for row in rows)
            negatives = len(rows) - positives
            short_name = question.split("(")[-1].split(")")[0] if "(" in question else question
            print(f"  {short_name}: {positives} positive, {negatives} negative")


if __name__ == "__main__":
    main()

