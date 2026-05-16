#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-smoke}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

cd "$ROOT"

echo "Workspace: $ROOT"
echo "Mode: $MODE"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script must run on a CUDA/NVIDIA host."
  exit 1
fi

nvidia-smi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv --version

if [[ ! -d "$ROOT/rate-evals" ]]; then
  echo "Missing rate-evals directory. Sync the full workspace to this GPU host first."
  exit 1
fi

if [[ ! -f "$ROOT/benchmarks/headct_rsna/data/manifest.csv" ]]; then
  echo "Missing prepared benchmark data. Sync data/ and benchmarks/ to this GPU host first."
  exit 1
fi

python3 "$ROOT/scripts/validate_headct_rsna_benchmark.py" \
  --data-dir "$ROOT/benchmarks/headct_rsna/data" \
  --sample-path-checks 100

case "$MODE" in
  dry-run)
    DRY_RUN=1 MAX_SAMPLES=3 NUM_GPUS="$NUM_GPUS" SKIP_EVAL=1 BATCH_SIZE=1 NUM_WORKERS=0 \
      bash "$ROOT/scripts/run_headct_pillar0_benchmark.sh"
    ;;
  smoke)
    MAX_SAMPLES="${MAX_SAMPLES:-3}" NUM_GPUS="$NUM_GPUS" SKIP_EVAL=1 BATCH_SIZE=1 NUM_WORKERS=0 \
      bash "$ROOT/scripts/run_headct_pillar0_benchmark.sh"
    ;;
  extract)
    NUM_GPUS="$NUM_GPUS" SKIP_EVAL=1 BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
      bash "$ROOT/scripts/run_headct_pillar0_benchmark.sh"
    ;;
  full)
    NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
      bash "$ROOT/scripts/run_headct_pillar0_benchmark.sh"
    ;;
  eval)
    cd "$ROOT/rate-evals"
    uv run rate-evaluate \
      --model pillar0 \
      --checkpoint-dir "$ROOT/benchmarks/headct_rsna/cache/pillar0_headct_rsna" \
      --dataset-name rve_brain_ct \
      --labels-json "$ROOT/benchmarks/headct_rsna/data/labels.json" \
      --output-dir "$ROOT/benchmarks/headct_rsna/results/pillar0_headct_rsna" \
      evaluation.use_wandb=false \
      evaluation.use_pytorch=false
    ;;
  *)
    echo "Unknown MODE=$MODE. Use dry-run, smoke, extract, full, or eval."
    exit 1
    ;;
esac

