#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT/artifacts/pillar0_headct_coreml"
UV_ARGS="${PILLAR0_UV_ARGS:---no-sync}"

mkdir -p "$ARTIFACT_DIR"
"$ROOT/scripts/apply_rate_evals_mobile_patch.sh"

echo "Checking Hugging Face auth and Pillar0-HeadCT access..."
(
  cd "$ROOT/rate-evals"
  uv run $UV_ARGS hf auth whoami
  uv run $UV_ARGS python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download("YalaLab/Pillar0-HeadCT", "config.json")
print(f"HF config accessible: {path}")
PY
)

if [[ ! -f "$ARTIFACT_DIR/sample_windowed_headct.npy" ]]; then
  echo "Exporting one real windowed HeadCT sample from the prepared benchmark..."
  (
    cd "$ROOT/rate-evals"
    uv run $UV_ARGS python ../scripts/export_pillar0_headct_sample_input.py
  )
fi

if [[ ! -f "$ARTIFACT_DIR/pillar0_headct_vision_coreml.pt" ]]; then
  echo "Tracing Core ML friendly Pillar-0 HeadCT vision encoder..."
  (
    cd "$ROOT/rate-evals"
    uv run $UV_ARGS python -u ../scripts/export_pillar0_headct_coreml.py \
      --skip-dry-run \
      --sample-input artifacts/pillar0_headct_coreml/sample_windowed_headct.npy \
      --trace-device cpu \
      --torch-dtype float32 \
      --trace-name pillar0_headct_vision_coreml.pt
  )
fi

if [[ ! -d "$ARTIFACT_DIR/Pillar0HeadCTVision_int8.mlpackage" ]]; then
  echo "Converting to Core ML and applying int8 weight quantization..."
  (
    cd "$ROOT/rate-evals"
    uv run $UV_ARGS python -u ../scripts/export_pillar0_headct_coreml.py \
      --skip-trace \
      --convert-coreml \
      --trace-name pillar0_headct_vision_coreml.pt \
      --coreml-name Pillar0HeadCTVision.mlpackage \
      --coreml-input-dtype float16 \
      --coreml-compute-precision float16 \
      --minimum-deployment-target iOS16 \
      --quantize-coreml-weights
  )
fi

"$ROOT/scripts/prepare_pillar0_mobile_artifacts.sh"

echo "Mobile artifacts are ready in $ARTIFACT_DIR"
