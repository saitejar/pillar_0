#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT/artifacts/pillar0_headct_coreml"
OUT="$ARTIFACT_DIR/mac_inference_latest.json"
UV_ARGS="${PILLAR0_UV_ARGS:---no-sync}"

"$ROOT/scripts/prepare_pillar0_mobile_artifacts.sh" >/dev/null
"$ROOT/scripts/apply_rate_evals_mobile_patch.sh" >/dev/null

echo "Running Pillar-0 HeadCT Core ML int8 inference on Mac..."
echo "Using CPU_ONLY and CPU_AND_NE. GPU/ALL is intentionally skipped because it was numerically wrong on this Mac."
(
  cd "$ROOT/rate-evals"
  uv run $UV_ARGS python ../scripts/validate_pillar0_coreml.py \
    --mlpackage "$ARTIFACT_DIR/Pillar0HeadCTVision_int8.mlpackage" \
    --torchscript "$ARTIFACT_DIR/pillar0_headct_vision_coreml.pt" \
    --sample-input "$ARTIFACT_DIR/sample_windowed_headct.npy" \
    --input-dtype float16 \
    --compute-units CPU_ONLY CPU_AND_NE
) | tee "$OUT"

echo "Saved Mac inference report: $OUT"
