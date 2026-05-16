#!/usr/bin/env python3
"""Compare a Pillar0 HeadCT Core ML package against the TorchScript trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SAMPLE = "artifacts/pillar0_headct_coreml/sample_windowed_headct.npy"
DEFAULT_TRACE = "artifacts/pillar0_headct_coreml/pillar0_headct_vision_coreml.pt"
DEFAULT_PACKAGE = "artifacts/pillar0_headct_coreml/Pillar0HeadCTVision_int8.mlpackage"


def discover_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "artifacts").exists() and (candidate / "scripts").exists():
            return candidate
    return start


def resolve_from_root(path: str, repo_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-input", default=DEFAULT_SAMPLE)
    parser.add_argument("--torchscript", default=DEFAULT_TRACE)
    parser.add_argument("--mlpackage", default=DEFAULT_PACKAGE)
    parser.add_argument("--input-name", default="windowed_headct")
    parser.add_argument("--input-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--compute-units",
        nargs="+",
        default=["CPU_ONLY", "CPU_AND_NE"],
        choices=["ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"],
    )
    parser.add_argument("--repo-root", default=None)
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    script_root = Path(__file__).resolve().parents[1]
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root is not None
        else discover_repo_root(script_root)
    )

    import coremltools as ct
    import numpy as np
    import torch

    sample_path = resolve_from_root(args.sample_input, repo_root)
    trace_path = resolve_from_root(args.torchscript, repo_root)
    package_path = resolve_from_root(args.mlpackage, repo_root)

    sample = np.load(sample_path)
    coreml_input = sample.astype(np.float16 if args.input_dtype == "float16" else np.float32)
    torch_input = torch.from_numpy(sample.astype(np.float32))

    torch.set_grad_enabled(False)
    torch_model = torch.jit.load(str(trace_path), map_location="cpu").eval()
    torch_output = torch_model(torch_input).detach().cpu().numpy()

    results = {
        "sample_input": str(sample_path),
        "torchscript": str(trace_path),
        "mlpackage": str(package_path),
        "input_shape": list(coreml_input.shape),
        "input_dtype": str(coreml_input.dtype),
        "compute_units": {},
    }

    for unit_name in args.compute_units:
        unit = getattr(ct.ComputeUnit, unit_name)
        mlmodel = ct.models.MLModel(str(package_path), compute_units=unit)
        output_dict = mlmodel.predict({args.input_name: coreml_input})
        coreml_output = next(iter(output_dict.values()))

        diff = np.abs(torch_output - coreml_output)
        cosine = float(
            (torch_output * coreml_output).sum()
            / (np.linalg.norm(torch_output) * np.linalg.norm(coreml_output))
        )
        results["compute_units"][unit_name] = {
            "output_shape": list(coreml_output.shape),
            "output_dtype": str(coreml_output.dtype),
            "finite": bool(np.isfinite(coreml_output).all()),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "cosine_similarity": cosine,
            "output_norm": float(np.linalg.norm(coreml_output)),
        }

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
