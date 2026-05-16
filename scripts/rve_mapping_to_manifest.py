#!/usr/bin/env python3
"""Convert RAVE mapping.csv into a RATE-Evals manifest.

RATE-Evals expects:

  sample_name,image_cache_path

RAVE writes a mapping keyed by `source_path`. This script joins RAVE's mapping
back to the benchmark `rve_series.csv` produced by prepare_rsna_headct_benchmark.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rve-series-csv", required=True, type=Path)
    parser.add_argument("--rave-mapping-csv", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()

    series_rows = read_rows(args.rve_series_csv)
    mapping_rows = read_rows(args.rave_mapping_csv)

    sample_by_series = {
        str(Path(row["series_path"]).resolve()): row["sample_name"] for row in series_rows
    }

    output_rows = []
    missing = []
    for row in mapping_rows:
        source_column = "source_path" if "source_path" in row else "series_path"
        series_path = str(Path(row[source_column]).resolve())
        sample_name = sample_by_series.get(series_path)
        if sample_name is None:
            missing.append(series_path)
            continue

        output_path = row.get("output_path") or row.get("processed_path")
        if not output_path:
            raise ValueError("RAVE mapping CSV must contain output_path or processed_path")

        output_rows.append(
            {
                "sample_name": sample_name,
                "image_cache_path": str(Path(output_path).resolve()),
            }
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_name", "image_cache_path"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"wrote {len(output_rows)} manifest rows to {args.output_manifest}")
    if missing:
        print(f"warning: {len(missing)} RAVE rows did not match rve_series.csv")


if __name__ == "__main__":
    main()
