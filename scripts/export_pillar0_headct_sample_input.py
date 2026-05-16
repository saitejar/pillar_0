#!/usr/bin/env python3
"""Export one real Pillar0-HeadCT windowed input tensor from the RSNA benchmark.

This mirrors the RATE/Pillar path for `rve_brain_ct`:

1. Load one RVE cached volume from `manifest.csv`.
2. Pad/crop it to `1 x 128 x 256 x 256`.
3. Flip depth like `RVEDataset`.
4. Apply `ct_window_type=all`, yielding `1 x 11 x 128 x 256 x 256`.
5. Save the windowed tensor as `.npy` for Core ML conversion/prediction tests.

The exported tensor is the model input for `Pillar0HeadCTVision.mlpackage`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_MANIFEST = "benchmarks/headct_rsna/data/manifest.csv"
DEFAULT_OUTPUT = "artifacts/pillar0_headct_coreml/sample_windowed_headct.npy"
DEFAULT_SHAPE = (128, 256, 256)


def discover_repo_root(start: Path) -> Path:
    """Find the local Pillar-0 replication root from a script/cwd path."""
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (
            (candidate / "rate-evals").exists()
            and (candidate / "rave").exists()
            and (candidate / "benchmarks").exists()
        ):
            return candidate
    return start


def resolve_from_root(path: str, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--target-d", type=int, default=DEFAULT_SHAPE[0])
    parser.add_argument("--target-h", type=int, default=DEFAULT_SHAPE[1])
    parser.add_argument("--target-w", type=int, default=DEFAULT_SHAPE[2])
    parser.add_argument("--pad-value", type=float, default=-1.0)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--repo-root",
        default=None,
        help=(
            "repo root used to add local rate-evals/rave packages to sys.path; "
            "auto-detected when omitted"
        ),
    )
    return parser


def add_local_packages(repo_root: Path) -> None:
    for rel in ("rate-evals", "rave"):
        path = repo_root / rel
        if path.exists():
            sys.path.insert(0, str(path))


def load_manifest_row(path: Path, sample_name: str | None, index: int) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows found in manifest: {path}")
    if sample_name is not None:
        for row in rows:
            if row.get("sample_name") == sample_name:
                return row
        raise SystemExit(f"sample_name {sample_name!r} not found in {path}")
    if index < 0 or index >= len(rows):
        raise SystemExit(f"index {index} out of range for manifest with {len(rows)} rows")
    return rows[index]


def pad_or_crop_4d(torch, volume, target_d: int, target_h: int, target_w: int, pad_value: float):
    """Match RATE's center pad/crop behavior for C,D,H,W tensors."""
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
    if volume.dim() != 4:
        raise ValueError(f"Expected RVE tensor with shape D,H,W or C,D,H,W; got {tuple(volume.shape)}")

    _, depth, height, width = volume.shape

    if depth < target_d:
        diff = target_d - depth
        before = diff // 2
        after = diff - before
        volume = torch.nn.functional.pad(volume, (0, 0, 0, 0, before, after), value=pad_value)
    elif depth > target_d:
        start = (depth - target_d) // 2
        volume = volume[:, start : start + target_d]

    height = volume.shape[2]
    if height < target_h:
        diff = target_h - height
        before = diff // 2
        after = diff - before
        volume = torch.nn.functional.pad(volume, (0, 0, before, after, 0, 0), value=pad_value)
    elif height > target_h:
        start = (height - target_h) // 2
        volume = volume[:, :, start : start + target_h, :]

    width = volume.shape[3]
    if width < target_w:
        diff = target_w - width
        before = diff // 2
        after = diff - before
        volume = torch.nn.functional.pad(volume, (before, after, 0, 0, 0, 0), value=pad_value)
    elif width > target_w:
        start = (width - target_w) // 2
        volume = volume[:, :, :, start : start + target_w]

    return volume


def main() -> int:
    args = make_argparser().parse_args()
    script_root = Path(__file__).resolve().parents[1]
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root is not None
        else discover_repo_root(script_root)
    )
    add_local_packages(repo_root)

    try:
        import numpy as np
        import torch
        import rve
        from rate_eval.models.common import batch_apply_ct_windowing
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run inside the rate-evals environment, for example:\n"
            "  cd rate-evals && uv run --no-sync python ../scripts/export_pillar0_headct_sample_input.py"
        ) from exc

    manifest_path = resolve_from_root(args.manifest, repo_root)
    row = load_manifest_row(manifest_path, args.sample_name, args.index)
    sample_name = row["sample_name"]
    cache_path = Path(row["image_cache_path"])
    if not cache_path.exists():
        raise SystemExit(f"RVE cache path does not exist for {sample_name}: {cache_path}")

    volume = rve.load_sample(str(cache_path), use_hardware_acceleration=False)
    volume = volume.to(torch.float32)
    volume = pad_or_crop_4d(torch, volume, args.target_d, args.target_h, args.target_w, args.pad_value)
    volume = torch.flip(volume, dims=[1])
    batch = volume.unsqueeze(0)
    windowed = batch_apply_ct_windowing(batch, ct_window_type="all", modality="CT", per_sample=True)

    array = windowed.cpu().numpy()
    array = array.astype(np.float16 if args.dtype == "float16" else np.float32)

    output_path = resolve_from_root(args.output, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)

    metadata = {
        "sample_name": sample_name,
        "cache_path": str(cache_path),
        "output": str(output_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "ct_window_type": "all",
        "modality": "brain_ct",
        "target_dhw": [args.target_d, args.target_h, args.target_w],
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
