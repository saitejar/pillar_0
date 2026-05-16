#!/usr/bin/env python3
"""Print progress for the 100/100/100 Core ML int8 agreement gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "pillar0_headct_coreml"


def main() -> int:
    files = sorted(BASE.glob("int8_split_agreement_100_*_cpu.jsonl"))
    if not files:
        print(f"No agreement JSONL files found in {BASE}")
        return 0

    summary = {}
    for path in files:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        summary[path.name] = {
            "rows": len(rows),
            "failures": sum(not row.get("passed_thresholds", False) for row in rows),
            "min_cosine": min([row["cosine_similarity"] for row in rows], default=None),
            "max_abs_diff": max([row["max_abs_diff"] for row in rows], default=None),
        }

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

