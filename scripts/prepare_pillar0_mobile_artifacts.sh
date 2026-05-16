#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT/artifacts/pillar0_headct_coreml"
IOS_RESOURCES="$ROOT/ios/test_pillar_0/test_pillar_0/Pillar0Resources"

MODEL_PACKAGE="$ARTIFACT_DIR/Pillar0HeadCTVision_int8.mlpackage"
MODEL_COMPILED="$ARTIFACT_DIR/Pillar0HeadCTVision_int8.mlmodelc"
SAMPLE_NPY="$ARTIFACT_DIR/sample_windowed_headct.npy"
TRACE="$ARTIFACT_DIR/pillar0_headct_vision_coreml.pt"
INPUT_RAW="$ARTIFACT_DIR/simulator_sample_windowed_headct_f16.raw"
EXPECTED_RAW="$ARTIFACT_DIR/simulator_expected_torchscript_f32.raw"
METADATA_JSON="$ARTIFACT_DIR/simulator_validation_inputs.json"

mkdir -p "$ARTIFACT_DIR" "$IOS_RESOURCES"

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing required artifact: $path" >&2
    echo "$hint" >&2
    exit 1
  fi
}

if [[ ! -d "$MODEL_COMPILED" ]]; then
  require_file "$MODEL_PACKAGE" "Run ./scripts/build_pillar0_mobile_artifacts.sh first, or copy Pillar0HeadCTVision_int8.mlpackage into $ARTIFACT_DIR."
  echo "Compiling Core ML package for iOS simulator/device bundle..."
  rm -rf "$MODEL_COMPILED"
  xcrun coremlcompiler compile "$MODEL_PACKAGE" "$ARTIFACT_DIR"
fi

if [[ ! -f "$INPUT_RAW" || ! -f "$EXPECTED_RAW" ]]; then
  require_file "$SAMPLE_NPY" "Run ./scripts/build_pillar0_mobile_artifacts.sh first, or place sample_windowed_headct.npy in $ARTIFACT_DIR."
  require_file "$TRACE" "Run ./scripts/build_pillar0_mobile_artifacts.sh first, or place pillar0_headct_vision_coreml.pt in $ARTIFACT_DIR."
  "$ROOT/scripts/apply_rate_evals_mobile_patch.sh"
  echo "Preparing raw fp16 input and expected fp32 TorchScript output for iOS validation..."
  (
    cd "$ROOT/rate-evals"
    uv run ${PILLAR0_UV_ARGS:---no-sync} python - "$ROOT" <<'PY'
from pathlib import Path
import json
import sys

import numpy as np
import torch

root = Path(sys.argv[1])
artifact_dir = root / "artifacts" / "pillar0_headct_coreml"
sample_path = artifact_dir / "sample_windowed_headct.npy"
trace_path = artifact_dir / "pillar0_headct_vision_coreml.pt"
input_raw = artifact_dir / "simulator_sample_windowed_headct_f16.raw"
expected_raw = artifact_dir / "simulator_expected_torchscript_f32.raw"
metadata_path = artifact_dir / "simulator_validation_inputs.json"

sample = np.load(sample_path)
if tuple(sample.shape) != (1, 11, 128, 256, 256):
    raise SystemExit(f"Unexpected sample shape: {sample.shape}")

sample.astype(np.float16).tofile(input_raw)

torch.set_grad_enabled(False)
model = torch.jit.load(str(trace_path), map_location="cpu").eval()
expected = model(torch.from_numpy(sample.astype(np.float32))).detach().cpu().numpy()
expected.astype(np.float32).tofile(expected_raw)

metadata = {
    "input_shape": list(sample.shape),
    "input_raw": str(input_raw),
    "expected_output_shape": list(expected.shape),
    "expected_raw": str(expected_raw),
    "torchscript": str(trace_path),
}
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
PY
  )
fi

echo "Copying iOS runtime resources..."
rm -rf "$IOS_RESOURCES/Pillar0HeadCTVision_int8.mlmodelc"
cp -R "$MODEL_COMPILED" "$IOS_RESOURCES/Pillar0HeadCTVision_int8.mlmodelc"
cp "$INPUT_RAW" "$IOS_RESOURCES/simulator_sample_windowed_headct_f16.raw"
cp "$EXPECTED_RAW" "$IOS_RESOURCES/simulator_expected_torchscript_f32.raw"
if command -v xattr >/dev/null 2>&1; then
  xattr -cr "$IOS_RESOURCES"
fi

echo "Prepared:"
du -sh "$IOS_RESOURCES"/Pillar0HeadCTVision_int8.mlmodelc \
  "$IOS_RESOURCES"/simulator_sample_windowed_headct_f16.raw \
  "$IOS_RESOURCES"/simulator_expected_torchscript_f32.raw
