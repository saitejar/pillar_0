#!/usr/bin/env python3
"""Validate a Pillar0 HeadCT Core ML package against TorchScript over data splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path


DEFAULT_DATA_DIR = "benchmarks/headct_rsna/data"
DEFAULT_TRACE = "artifacts/pillar0_headct_coreml/pillar0_headct_vision_coreml.pt"
DEFAULT_PACKAGE = "artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_int8.mlpackage"
DEFAULT_OUTPUT = "artifacts/pillar0_headct_coreml/int8_split_agreement.json"


def discover_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "benchmarks").exists() and (candidate / "rate-evals").exists():
            return candidate
    return start


def resolve_from_root(path: str, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved


def add_local_packages(repo_root: Path) -> None:
    import sys

    for rel in ("rate-evals", "rave"):
        path = repo_root / rel
        if path.exists():
            sys.path.insert(0, str(path))


def load_manifest(path: Path) -> dict[str, Path]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return {row["sample_name"]: Path(row["image_cache_path"]) for row in rows}


def load_split(path: Path) -> list[str]:
    names: list[str] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                names.append(json.loads(line)["sample_name"])
    return names


def choose_samples(names: list[str], count: int, seed: int, mode: str) -> list[str]:
    if count <= 0 or count >= len(names):
        return list(names)
    if mode == "head":
        return names[:count]
    rng = random.Random(seed)
    selected = list(names)
    rng.shuffle(selected)
    return selected[:count]


def pad_or_crop_4d(torch, volume, target_d: int, target_h: int, target_w: int, pad_value: float):
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
    if volume.dim() != 4:
        raise ValueError(f"Expected D,H,W or C,D,H,W; got {tuple(volume.shape)}")

    _, depth, height, width = volume.shape

    if depth < target_d:
        diff = target_d - depth
        before = diff // 2
        volume = torch.nn.functional.pad(
            volume, (0, 0, 0, 0, before, diff - before), value=pad_value
        )
    elif depth > target_d:
        start = (depth - target_d) // 2
        volume = volume[:, start : start + target_d]

    height = volume.shape[2]
    if height < target_h:
        diff = target_h - height
        before = diff // 2
        volume = torch.nn.functional.pad(
            volume, (0, 0, before, diff - before, 0, 0), value=pad_value
        )
    elif height > target_h:
        start = (height - target_h) // 2
        volume = volume[:, :, start : start + target_h, :]

    width = volume.shape[3]
    if width < target_w:
        diff = target_w - width
        before = diff // 2
        volume = torch.nn.functional.pad(
            volume, (before, diff - before, 0, 0, 0, 0), value=pad_value
        )
    elif width > target_w:
        start = (width - target_w) // 2
        volume = volume[:, :, :, start : start + target_w]

    return volume


def make_windowed_input(
    torch,
    np,
    rve,
    batch_apply_ct_windowing,
    cache_path: Path,
    target_d: int,
    target_h: int,
    target_w: int,
    pad_value: float,
    input_dtype: str,
):
    volume = rve.load_sample(str(cache_path), use_hardware_acceleration=False)
    volume = volume.to(torch.float32)
    volume = pad_or_crop_4d(torch, volume, target_d, target_h, target_w, pad_value)
    volume = torch.flip(volume, dims=[1])
    batch = volume.unsqueeze(0)
    windowed = batch_apply_ct_windowing(
        batch,
        ct_window_type="all",
        modality="CT",
        per_sample=True,
    )
    array = windowed.cpu().numpy()
    return array.astype(np.float16 if input_dtype == "float16" else np.float32)


def cosine_similarity(np, lhs, rhs) -> float:
    denom = np.linalg.norm(lhs) * np.linalg.norm(rhs)
    if denom == 0:
        return 0.0
    return float((lhs * rhs).sum() / denom)


def summarize_split(rows: list[dict], np) -> dict:
    if not rows:
        return {"count": 0}
    cosines = np.array([row["cosine_similarity"] for row in rows], dtype=np.float64)
    max_diffs = np.array([row["max_abs_diff"] for row in rows], dtype=np.float64)
    mean_diffs = np.array([row["mean_abs_diff"] for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "cosine_min": float(cosines.min()),
        "cosine_mean": float(cosines.mean()),
        "max_abs_diff_max": float(max_diffs.max()),
        "max_abs_diff_mean": float(max_diffs.mean()),
        "mean_abs_diff_mean": float(mean_diffs.mean()),
        "all_finite": bool(all(row["finite"] for row in rows)),
        "all_passed_thresholds": bool(all(row["passed_thresholds"] for row in rows)),
    }


def load_progress(path: Path) -> dict[tuple[str, str, str], dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["split"], row["compute_unit"], row["sample_name"])
            rows[key] = row
    return rows


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest-name", default="manifest.csv")
    parser.add_argument("--torchscript", default=DEFAULT_TRACE)
    parser.add_argument("--mlpackage", default=DEFAULT_PACKAGE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--progress-jsonl",
        default=None,
        help="append one JSON row per completed sample; defaults to output path with .jsonl",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse completed rows from --progress-jsonl",
    )
    parser.add_argument("--input-name", default="windowed_headct")
    parser.add_argument("--input-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--samples-per-split", type=int, default=3)
    parser.add_argument("--sample-mode", choices=("random", "head"), default="random")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--compute-units",
        nargs="+",
        default=["CPU_ONLY"],
        choices=["ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"],
    )
    parser.add_argument("--target-d", type=int, default=128)
    parser.add_argument("--target-h", type=int, default=256)
    parser.add_argument("--target-w", type=int, default=256)
    parser.add_argument("--pad-value", type=float, default=-1.0)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-abs-diff", type=float, default=0.01)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--fail-on-threshold", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    script_root = Path(__file__).resolve().parents[1]
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root is not None
        else discover_repo_root(script_root)
    )
    add_local_packages(repo_root)

    import coremltools as ct
    import numpy as np
    import torch
    import rve
    from rate_eval.models.common import batch_apply_ct_windowing

    data_dir = resolve_from_root(args.data_dir, repo_root)
    manifest = load_manifest(data_dir / args.manifest_name)
    trace_path = resolve_from_root(args.torchscript, repo_root)
    package_path = resolve_from_root(args.mlpackage, repo_root)
    output_path = resolve_from_root(args.output, repo_root)
    progress_path = (
        resolve_from_root(args.progress_jsonl, repo_root)
        if args.progress_jsonl is not None
        else output_path.with_suffix(".jsonl")
    )
    completed_rows = load_progress(progress_path) if args.resume else {}

    torch.set_grad_enabled(False)
    torch_model = torch.jit.load(str(trace_path), map_location="cpu").eval()
    coreml_models = {
        unit_name: ct.models.MLModel(
            str(package_path),
            compute_units=getattr(ct.ComputeUnit, unit_name),
        )
        for unit_name in args.compute_units
    }

    results = {
        "repo_root": str(repo_root),
        "data_dir": str(data_dir),
        "torchscript": str(trace_path),
        "mlpackage": str(package_path),
        "input_dtype": args.input_dtype,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_abs_diff": args.max_abs_diff,
        },
        "progress_jsonl": str(progress_path),
        "splits": {},
    }

    overall_passed = True
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_f = progress_path.open("a", buffering=1)
    for split in args.splits:
        split_names = load_split(data_dir / f"{split}.json")
        selected = choose_samples(
            split_names,
            args.samples_per_split,
            args.seed + sum(ord(ch) for ch in split),
            args.sample_mode,
        )
        split_result = {"selected_samples": selected, "compute_units": {}}

        for unit_name, coreml_model in coreml_models.items():
            unit_rows = []
            for sample_name in selected:
                if sample_name not in manifest:
                    raise SystemExit(f"{sample_name} from {split} missing in manifest")

                key = (split, unit_name, sample_name)
                if key in completed_rows:
                    row = completed_rows[key]
                    unit_rows.append(row)
                    overall_passed = overall_passed and row["passed_thresholds"]
                    print(json.dumps({"resumed": True, **row}))
                    continue

                started = time.time()
                coreml_input = make_windowed_input(
                    torch,
                    np,
                    rve,
                    batch_apply_ct_windowing,
                    manifest[sample_name],
                    args.target_d,
                    args.target_h,
                    args.target_w,
                    args.pad_value,
                    args.input_dtype,
                )
                preprocessing_sec = time.time() - started

                started = time.time()
                torch_output = torch_model(
                    torch.from_numpy(coreml_input.astype(np.float32))
                ).detach().cpu().numpy()
                torch_sec = time.time() - started

                started = time.time()
                output_dict = coreml_model.predict({args.input_name: coreml_input})
                coreml_output = next(iter(output_dict.values()))
                coreml_sec = time.time() - started

                diff = np.abs(torch_output - coreml_output)
                cosine = cosine_similarity(np, torch_output, coreml_output)
                passed = (
                    bool(np.isfinite(coreml_output).all())
                    and cosine >= args.min_cosine
                    and float(diff.max()) <= args.max_abs_diff
                )
                overall_passed = overall_passed and passed

                row = {
                    "split": split,
                    "compute_unit": unit_name,
                    "sample_name": sample_name,
                    "finite": bool(np.isfinite(coreml_output).all()),
                    "cosine_similarity": cosine,
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "torch_norm": float(np.linalg.norm(torch_output)),
                    "coreml_norm": float(np.linalg.norm(coreml_output)),
                    "preprocessing_sec": round(preprocessing_sec, 3),
                    "torchscript_sec": round(torch_sec, 3),
                    "coreml_sec": round(coreml_sec, 3),
                    "passed_thresholds": passed,
                }
                unit_rows.append(row)
                row_json = json.dumps(row)
                progress_f.write(row_json + "\n")
                print(row_json)

            split_result["compute_units"][unit_name] = {
                "summary": summarize_split(unit_rows, np),
                "samples": unit_rows,
            }

        results["splits"][split] = split_result

    results["overall_passed_thresholds"] = overall_passed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    progress_f.close()
    print(json.dumps({"output": str(output_path), "overall_passed": overall_passed}, indent=2))

    if args.fail_on_threshold and not overall_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
