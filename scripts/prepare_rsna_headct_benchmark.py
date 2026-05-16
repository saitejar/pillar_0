#!/usr/bin/env python3
"""Prepare public RSNA Head CT data for Pillar0-HeadCT benchmarking.

Inputs:
  - Raw RSNA DICOM slice directory from Kaggle.
  - Kaggle slice label CSV (`stage_2_train.csv` style).

Outputs:
  - RAVE `series_path` CSV.
  - RATE-Evals train/valid/test JSONL files.
  - RATE-Evals `labels.json` with study-level hemorrhage labels.

The script groups DICOM slices by StudyInstanceUID and SeriesInstanceUID,
keeps the largest series per study, and aggregates slice labels with `any`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


QUESTIONS = {
    "any": "Is intra-cranial hemorrhage (ICH) present?",
    "intraparenchymal": "Is intra-parenchymal hemorrhage (IPH) present?",
    "intraventricular": "Is intra-ventricular hemorrhage (IVH) present?",
    "subarachnoid": "Is subarachnoid hemorrhage (SAH) present?",
    "subdural": "Is subdural hemorrhage (SDH) present?",
    "epidural": "Is epidural hemorrhage (EDH) present?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-dir", required=True, type=Path)
    parser.add_argument("--labels-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--series-link-dir",
        type=Path,
        default=None,
        help="Optional directory where per-series symlink folders are created for RAVE.",
    )
    parser.add_argument("--valid-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-studies",
        type=int,
        default=None,
        help="Optional cap for a fast benchmark smoke run.",
    )
    return parser.parse_args()


def require_pydicom():
    try:
        import pydicom  # type: ignore
    except ImportError:
        print(
            "pydicom is required. Install it in your environment with `uv add pydicom` "
            "or `pip install pydicom`.",
            file=sys.stderr,
        )
        raise
    return pydicom


def load_slice_labels(path: Path) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    pattern = re.compile(r"^(ID_[0-9a-fA-F]+)_(.+)$")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "ID" not in reader.fieldnames or "Label" not in reader.fieldnames:
            raise ValueError("labels CSV must contain ID and Label columns")

        for row in reader:
            match = pattern.match(row["ID"])
            if not match:
                continue
            slice_id, label_name = match.groups()
            labels[slice_id][label_name] = int(float(row["Label"]))

    return labels


def dicom_slice_id(path: Path) -> str:
    return path.stem


def scan_dicom_series(dicom_dir: Path, slice_labels: dict[str, dict[str, int]]):
    pydicom = require_pydicom()
    studies: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    study_series_labels: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {name: 0 for name in QUESTIONS}
    )

    dicom_paths = sorted(dicom_dir.rglob("*.dcm"))
    if not dicom_paths:
        raise FileNotFoundError(f"No .dcm files found under {dicom_dir}")

    for idx, path in enumerate(dicom_paths, start=1):
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                specific_tags=["StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber"],
            )
        except Exception as exc:
            print(f"warning: failed to read DICOM metadata from {path}: {exc}", file=sys.stderr)
            continue

        study_uid = str(getattr(ds, "StudyInstanceUID", ""))
        series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if not study_uid or not series_uid:
            print(f"warning: missing study/series UID for {path}", file=sys.stderr)
            continue

        studies[study_uid][series_uid].append(path)

        labels = slice_labels.get(dicom_slice_id(path), {})
        aggregate = study_series_labels[(study_uid, series_uid)]
        for label_name in QUESTIONS:
            aggregate[label_name] = max(aggregate[label_name], int(labels.get(label_name, 0)))

        if idx % 50000 == 0:
            print(f"scanned {idx} DICOM slices")

    selected = []
    for study_uid, series_map in studies.items():
        series_uid, paths = max(series_map.items(), key=lambda item: len(item[1]))
        selected.append(
            {
                "sample_name": safe_sample_name(study_uid),
                "study_uid": study_uid,
                "series_uid": series_uid,
                "paths": sorted(paths, key=lambda p: p.name),
                "labels": study_series_labels[(study_uid, series_uid)],
            }
        )
    return selected


def safe_sample_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def create_symlink_series(selected: list[dict], link_root: Path) -> None:
    link_root.mkdir(parents=True, exist_ok=True)
    for item in selected:
        series_dir = link_root / item["sample_name"]
        series_dir.mkdir(parents=True, exist_ok=True)
        for path in item["paths"]:
            target = series_dir / path.name
            if target.exists():
                continue
            os.symlink(path.resolve(), target)
        item["series_path"] = str(series_dir.resolve())


def assign_direct_series_paths(selected: list[dict]) -> None:
    for item in selected:
        common_parent = Path(os.path.commonpath([str(p.parent) for p in item["paths"]]))
        item["series_path"] = str(common_parent.resolve())


def stratified_split(
    selected: list[dict], valid_frac: float, test_frac: float, seed: int
) -> dict[str, list[dict]]:
    positives = [item for item in selected if item["labels"].get("any", 0) == 1]
    negatives = [item for item in selected if item["labels"].get("any", 0) == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_group(items: list[dict]):
        n = len(items)
        n_test = int(round(n * test_frac))
        n_valid = int(round(n * valid_frac))
        test = items[:n_test]
        valid = items[n_test : n_test + n_valid]
        train = items[n_test + n_valid :]
        return train, valid, test

    pos_train, pos_valid, pos_test = split_group(positives)
    neg_train, neg_valid, neg_test = split_group(negatives)

    splits = {
        "train": pos_train + neg_train,
        "valid": pos_valid + neg_valid,
        "test": pos_test + neg_test,
    }
    for items in splits.values():
        rng.shuffle(items)
    return splits


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w") as f:
        for item in items:
            f.write(
                json.dumps(
                    {
                        "sample_name": item["sample_name"],
                        "nii_path": None,
                        "report_metadata": "",
                        "study_uid": item["study_uid"],
                        "series_uid": item["series_uid"],
                    }
                )
                + "\n"
            )


def write_labels(path: Path, selected: list[dict]) -> None:
    labels_json = {}
    for item in selected:
        qa_pairs = []
        for label_name, question in QUESTIONS.items():
            qa_pairs.append({question: "yes" if item["labels"].get(label_name, 0) == 1 else "no"})

        labels_json[item["sample_name"]] = {
            "qa_results": {
                "Hemorrhage": qa_pairs,
            },
            "study_uid": item["study_uid"],
            "series_uid": item["series_uid"],
        }

    path.write_text(json.dumps(labels_json, indent=2, sort_keys=True) + "\n")


def write_rve_series(path: Path, selected: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_name", "accession", "series_number", "series_path"]
        )
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "sample_name": item["sample_name"],
                    "accession": item["sample_name"],
                    "series_number": 1,
                    "series_path": item["series_path"],
                }
            )


def write_summary(path: Path, splits: dict[str, list[dict]]) -> None:
    rows = []
    for split, items in splits.items():
        total = len(items)
        positives = sum(item["labels"].get("any", 0) for item in items)
        rows.append(
            {
                "split": split,
                "num_studies": total,
                "ich_positive": positives,
                "ich_negative": total - positives,
            }
        )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    slice_labels = load_slice_labels(args.labels_csv)
    selected = scan_dicom_series(args.dicom_dir, slice_labels)

    if args.max_studies is not None:
        selected = selected[: args.max_studies]

    if args.series_link_dir is not None:
        create_symlink_series(selected, args.series_link_dir)
    else:
        assign_direct_series_paths(selected)

    splits = stratified_split(selected, args.valid_frac, args.test_frac, args.seed)

    for split, items in splits.items():
        write_jsonl(args.output_dir / f"{split}.json", items)

    write_labels(args.output_dir / "labels.json", selected)
    write_rve_series(args.output_dir / "rve_series.csv", selected)
    write_summary(args.output_dir / "summary.csv", splits)

    print(f"prepared {len(selected)} studies in {args.output_dir}")
    for split, items in splits.items():
        positives = sum(item["labels"].get("any", 0) for item in items)
        print(f"{split}: {len(items)} studies, {positives} ICH-positive")


if __name__ == "__main__":
    main()
