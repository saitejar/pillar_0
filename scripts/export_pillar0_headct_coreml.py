#!/usr/bin/env python3
"""Attempt to export Pillar0-HeadCT teacher vision inference to Core ML.

This is intentionally a feasibility harness for the teacher model, not a
student/distillation script. It preserves the Pillar-0 HeadCT vision input
contract used by RATE after CT windowing:

    B x 11 x 128 x 256 x 256

The raw/RVE CT volume -> 11-window tensor step is kept outside the Core ML graph
because the released RATE/RVE windowing path is Python/RVE driven and is not a
stable TorchScript/Core ML conversion target. The Core ML model produced here
therefore accepts the same windowed tensor that the HF Pillar-0 vision encoder
expects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_SHAPE = (1, 11, 128, 256, 256)


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


def parse_shape(value: str) -> tuple[int, ...]:
    parts = value.lower().replace("x", ",").split(",")
    shape = tuple(int(part.strip()) for part in parts if part.strip())
    if len(shape) != 5:
        raise argparse.ArgumentTypeError(
            "shape must have 5 dimensions: B,C,D,H,W, e.g. 1,11,128,256,256"
        )
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("all shape dimensions must be positive")
    return shape


def sizeof_shape(shape: Iterable[int], bytes_per_value: int) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total * bytes_per_value


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def resolve_torch_dtype(torch, dtype: str):
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def get_coreml_target(ct, name: str):
    try:
        return getattr(ct.target, name)
    except AttributeError as exc:
        available = [item for item in dir(ct.target) if item.startswith("iOS")]
        raise SystemExit(
            f"coremltools target {name!r} not found. Available iOS targets: {available}"
        ) from exc


def tensor_description(tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def summarize_output(output) -> object:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(output, torch.Tensor):
        return tensor_description(output)
    if isinstance(output, dict):
        return {key: summarize_output(value) for key, value in output.items()}
    if isinstance(output, (list, tuple)):
        return [summarize_output(value) for value in output]
    return repr(output)


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trace and optionally convert Pillar0-HeadCT teacher vision features "
            "to Core ML while preserving the 11-window volumetric input shape."
        )
    )
    parser.add_argument("--model-repo-id", default="YalaLab/Pillar0-HeadCT")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--modality", default="brain_ct")
    parser.add_argument(
        "--input-shape",
        type=parse_shape,
        default=DEFAULT_INPUT_SHAPE,
        help="B,C,D,H,W shape for the windowed Pillar input tensor",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
        help="dtype for loading/tracing the PyTorch model",
    )
    parser.add_argument(
        "--trace-device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="device used for the PyTorch trace/dry run",
    )
    parser.add_argument("--output-dir", default="artifacts/pillar0_headct_coreml")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="project root for resolving relative artifact paths; auto-detected when omitted",
    )
    parser.add_argument("--trace-name", default="pillar0_headct_vision.pt")
    parser.add_argument("--coreml-name", default="Pillar0HeadCTVision.mlpackage")
    parser.add_argument(
        "--sample-input",
        default=None,
        help="optional .npy tensor to use for dry-run/tracing instead of zeros",
    )
    parser.add_argument(
        "--input-name",
        default="windowed_headct",
        help="Core ML model input name",
    )
    parser.add_argument(
        "--minimum-deployment-target",
        default="iOS16",
        help="coremltools deployment target, e.g. iOS16 or iOS17",
    )
    parser.add_argument(
        "--coreml-input-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--coreml-compute-precision",
        choices=("float16", "float32"),
        default="float16",
        help="Core ML ML Program compute precision",
    )
    parser.add_argument(
        "--skip-dry-run",
        action="store_true",
        help="skip a PyTorch forward pass before tracing",
    )
    parser.add_argument(
        "--load-model-only",
        action="store_true",
        help="load the Hugging Face model and exit before allocating input/tracing",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="run one PyTorch forward pass and exit before tracing",
    )
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="do not generate TorchScript; useful for input/memory preflight only",
    )
    parser.add_argument(
        "--check-trace",
        action="store_true",
        help="enable TorchScript trace checking; off by default to avoid extra full-volume forwards",
    )
    parser.add_argument(
        "--coreml-friendly-repeat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "replace repeat_interleave in Atlas multiscale attention with an "
            "equivalent expand/reshape path before tracing"
        ),
    )
    parser.add_argument(
        "--convert-coreml",
        action="store_true",
        help="convert the traced TorchScript model to Core ML",
    )
    parser.add_argument(
        "--quantize-coreml-weights",
        action="store_true",
        help="apply Core ML post-training int8 weight quantization after conversion",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass trust_remote_code to AutoModel.from_pretrained",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="avoid network calls; requires model files already cached",
    )
    return parser


def repeat_batch_dim_for_coreml(tensor, num_repeats: int):
    repeats = int(num_repeats)
    if repeats == 1:
        return tensor
    shape = tensor.shape
    return (
        tensor.unsqueeze(1)
        .expand(shape[0], repeats, *shape[1:])
        .reshape(shape[0] * repeats, *shape[1:])
    )


def coreml_friendly_all2all_sattn(self, x_S, S):
    import torch

    q_S, k_S, v_S = self.get_qkv(x_S, S)

    k_Sp1, v_Sp1 = [k_S], [v_S]
    if len(self.out_scales) > 0:
        for T, out_t in self.out_scales.items():
            x_t = out_t["tokens"]
            num_repeats = x_S.shape[0] // x_t.shape[0]
            k_t, v_t = self.get_qkv(x_t, T, keys=["kv"])
            k_t = repeat_batch_dim_for_coreml(k_t, num_repeats)
            v_t = repeat_batch_dim_for_coreml(v_t, num_repeats)

            k_Sp1.append(k_t)
            v_Sp1.append(v_t)

    k_Sp1 = torch.cat(k_Sp1, dim=2)
    v_Sp1 = torch.cat(v_Sp1, dim=2)

    x_S = self.blocks[S].skip_with_drop(
        x_S, self.blocks[S].xattn_qkv(q_S, k_Sp1, v_Sp1)
    )
    x_S = self.blocks[S].mlp_residual(x_S)

    return x_S


def patch_coreml_friendly_repeat(model) -> int:
    patched = 0
    for module in model.modules():
        if hasattr(module, "_process__sequential__all2all_sattn"):
            module._process__sequential__all2all_sattn = types.MethodType(
                coreml_friendly_all2all_sattn,
                module,
            )
            patched += 1
    return patched


def convert_trace_to_coreml(args, traced, shape: tuple[int, ...], output_dir: Path) -> None:
    try:
        import numpy as np
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit(
            "Missing coremltools/numpy. Install coremltools on macOS before converting."
        ) from exc

    coreml_dtype = np.float16 if args.coreml_input_dtype == "float16" else np.float32
    compute_precision = (
        ct.precision.FLOAT16
        if args.coreml_compute_precision == "float16"
        else ct.precision.FLOAT32
    )
    target = get_coreml_target(ct, args.minimum_deployment_target)
    package_path = output_dir / args.coreml_name

    print(f"Converting TorchScript to Core ML package: {package_path}")
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=compute_precision,
        inputs=[ct.TensorType(name=args.input_name, shape=shape, dtype=coreml_dtype)],
    )
    mlmodel.save(str(package_path))

    if args.quantize_coreml_weights:
        print("Applying Core ML int8 linear weight quantization...")
        import coremltools.optimize as cto

        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
            granularity="per_channel",
            weight_threshold=2048,
        )
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        quantized = cto.coreml.linear_quantize_weights(mlmodel, config=config)
        quantized_path = package_path.with_name(package_path.stem + "_int8.mlpackage")
        quantized.save(str(quantized_path))
        print(f"Saved quantized package: {quantized_path}")


class Pillar0VisionWrapper:
    """Small nn.Module wrapper created lazily after torch import."""

    @staticmethod
    def build(torch, model, modality: str):
        class _Wrapper(torch.nn.Module):
            def __init__(self, inner_model, inner_modality: str):
                super().__init__()
                self.inner_model = inner_model
                self.inner_modality = inner_modality

            def forward(self, windowed_volume):
                out = self.inner_model.extract_vision_feats(
                    {self.inner_modality: windowed_volume}
                )
                if isinstance(out, dict):
                    for key in ("features", "image_features", "vision_feats", "embeddings"):
                        if key in out:
                            out = out[key]
                            break
                    else:
                        raise RuntimeError(
                            "extract_vision_feats returned a dict without a known tensor key: "
                            f"{sorted(out.keys())}"
                        )
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out

        return _Wrapper(model, modality)


def main() -> int:
    args = make_argparser().parse_args()
    script_root = Path(__file__).resolve().parents[1]
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root is not None
        else discover_repo_root(script_root)
    )
    output_dir = resolve_from_root(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    shape = tuple(args.input_shape)
    print("Pillar0 HeadCT teacher-on-iPhone export attempt")
    print(json.dumps(
        {
            "model_repo_id": args.model_repo_id,
            "revision": args.revision,
            "modality": args.modality,
            "input_shape_BCDHW": shape,
            "input_values": sizeof_shape(shape, 1),
            "input_size_int8": human_bytes(sizeof_shape(shape, 1)),
            "input_size_fp16": human_bytes(sizeof_shape(shape, 2)),
            "input_size_fp32": human_bytes(sizeof_shape(shape, 4)),
            "sample_input": args.sample_input,
            "repo_root": str(repo_root),
            "output_dir": str(output_dir),
        },
        indent=2,
    ))

    if args.skip_dry_run and args.dry_run_only:
        raise SystemExit("--dry-run-only cannot be combined with --skip-dry-run.")

    if (
        args.skip_trace
        and not args.convert_coreml
        and not args.load_model_only
        and not args.dry_run_only
    ):
        print("Preflight only: --skip-trace set and --convert-coreml not requested.")
        return 0

    if args.skip_trace and args.convert_coreml:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing PyTorch. Run this inside the rate-evals or Pillar-0 environment, "
                "e.g. `uv run --no-sync python ...`."
            ) from exc

        trace_path = output_dir / args.trace_name
        if not trace_path.exists():
            raise SystemExit(f"TorchScript trace does not exist: {trace_path}")
        print(f"Loading existing TorchScript trace: {trace_path}")
        traced = torch.jit.load(str(trace_path), map_location="cpu").eval()
        convert_trace_to_coreml(args, traced, shape, output_dir)
        print("Done.")
        return 0

    try:
        import torch
        from transformers import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "Missing PyTorch/transformers. Run this inside the rate-evals or "
            "Pillar-0 environment, e.g. `uv run --no-sync python ...`."
        ) from exc

    if args.trace_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")
    if args.trace_device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but torch.backends.mps.is_available() is false.")

    device = torch.device(args.trace_device)
    torch_dtype = resolve_torch_dtype(torch, args.torch_dtype)

    print("Loading Hugging Face model...")
    model = AutoModel.from_pretrained(
        args.model_repo_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
    )
    model.eval().to(device)
    print("Model loaded.")

    if args.load_model_only:
        print("Model access/load probe only: --load-model-only set.")
        return 0

    if args.coreml_friendly_repeat:
        patched = patch_coreml_friendly_repeat(model)
        print(f"Core ML friendly repeat patch applied to {patched} module(s).")

    wrapper = Pillar0VisionWrapper.build(torch, model, args.modality).eval().to(device)
    if args.sample_input is not None:
        import numpy as np

        sample_path = resolve_from_root(args.sample_input, repo_root)
        print(f"Loading sample input: {sample_path}")
        sample_array = np.load(sample_path)
        if tuple(sample_array.shape) != shape:
            raise SystemExit(
                f"sample shape {tuple(sample_array.shape)} does not match --input-shape {shape}"
            )
        sample = torch.from_numpy(sample_array).to(dtype=torch_dtype, device=device)
    else:
        sample = torch.zeros(shape, dtype=torch_dtype, device=device)

    if not args.skip_dry_run:
        print("Running PyTorch dry run...")
        with torch.no_grad():
            output = wrapper(sample)
        print("Dry run output:")
        print(json.dumps(summarize_output(output), indent=2))

    if args.dry_run_only:
        print("Dry run probe only: --dry-run-only set.")
        return 0

    trace_path = output_dir / args.trace_name
    if not args.skip_trace:
        print(f"Tracing to TorchScript: {trace_path}")
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                sample,
                strict=False,
                check_trace=args.check_trace,
            )
            traced = torch.jit.freeze(traced.eval())
        traced.save(str(trace_path))
    else:
        traced = torch.jit.load(str(trace_path), map_location=device)

    if not args.convert_coreml:
        print("Core ML conversion skipped. Pass --convert-coreml to convert.")
        return 0

    convert_trace_to_coreml(args, traced, shape, output_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
