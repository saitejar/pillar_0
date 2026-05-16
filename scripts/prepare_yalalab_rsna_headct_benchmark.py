#!/usr/bin/env python3
"""Prepare YalaLab's preprocessed RSNA Head CT dataset for RATE-Evals.

This consumes the archive layout from `YalaLab/rsna_0.5_0.5_1.25`:

  rsna/
    metadata/{train,val,test}.csv
    data_0.5_0.5_1.25/<volume_dir>/{volume.mkv,metadata.json}

and writes:

  benchmarks/headct_rsna/data/{train,valid,test}.json
  benchmarks/headct_rsna/data/labels.json
  benchmarks/headct_rsna/data/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument(
        "--rsna-root",
        required=True,
        type=Path,
        help="Path to extracted `rsna` directory from the YalaLab dataset archive.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def local_volume_path(rsna_root: Path, row: dict[str, str]) -> Path:
    output_path = row["output_path"]
    volume_dir_name = Path(output_path).name
    return (rsna_root / "data_0.5_0.5_1.25" / volume_dir_name).resolve()


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w") as f:
        for row in rows:
            sample_name = row["SeriesInstanceUID"]
            f.write(
                json.dumps(
                    {
                        "sample_name": sample_name,
                        "nii_path": None,
                        "report_metadata": "",
                        "series_uid": row.get("series_uid", sample_name),
                        "patient_id": row.get("PatientID", ""),
                    }
                )
                + "\n"
            )


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    labels_json = {}
    for row in rows:
        sample_name = row["SeriesInstanceUID"]
        qa_pairs = []
        for label_name, question in QUESTIONS.items():
            value = int(float(row[label_name]))
            qa_pairs.append({question: "yes" if value == 1 else "no"})

        labels_json[sample_name] = {
            "qa_results": {"Hemorrhage": qa_pairs},
            "series_uid": row.get("series_uid", sample_name),
            "patient_id": row.get("PatientID", ""),
        }

    path.write_text(json.dumps(labels_json, indent=2, sort_keys=True) + "\n")


def write_manifest(path: Path, rsna_root: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_name", "image_cache_path"])
        writer.writeheader()
        for row in rows:
            volume_path = local_volume_path(rsna_root, row)
            if not volume_path.exists():
                raise FileNotFoundError(f"Missing volume directory: {volume_path}")
            writer.writerow(
                {
                    "sample_name": row["SeriesInstanceUID"],
                    "image_cache_path": str(volume_path),
                }
            )


def write_summary(path: Path, split_rows: dict[str, list[dict[str, str]]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "num_studies", "ich_positive", "ich_negative"]
        )
        writer.writeheader()
        for split, rows in split_rows.items():
            positives = sum(int(float(row["any"])) for row in rows)
            total = len(rows)
            writer.writerow(
                {
                    "split": split,
                    "num_studies": total,
                    "ich_positive": positives,
                    "ich_negative": total - positives,
                }
            )


def main() -> None:
    args = parse_args()
    metadata_dir = args.rsna_root / "metadata"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = {
        "train": read_csv(metadata_dir / "train.csv"),
        "valid": read_csv(metadata_dir / "val.csv"),
        "test": read_csv(metadata_dir / "test.csv"),
    }

    all_rows = []
    for split, rows in split_rows.items():
        write_jsonl(args.output_dir / f"{split}.json", rows)
        all_rows.extend(rows)

    write_labels(args.output_dir / "labels.json", all_rows)
    write_manifest(args.output_dir / "manifest.csv", args.rsna_root, all_rows)
    write_summary(args.output_dir / "summary.csv", split_rows)

    print(f"prepared RATE-Evals files in {args.output_dir}")
    for split, rows in split_rows.items():
        positives = sum(int(float(row["any"])) for row in rows)
        print(f"{split}: {len(rows)} studies, {positives} ICH-positive")


if __name__ == "__main__":
    main()

