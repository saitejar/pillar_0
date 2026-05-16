#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/benchmarks/headct_rsna/data"
CACHE_DIR="$ROOT/benchmarks/headct_rsna/cache/pillar0_headct_rsna"
RESULTS_DIR="$ROOT/benchmarks/headct_rsna/results/pillar0_headct_rsna"

DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
ALLOW_CPU="${ALLOW_CPU:-0}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_GPUS="${NUM_GPUS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PYTHON="${PYTHON:-python3}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

cd "$ROOT/rate-evals"

if [[ ! -f "$DATA_DIR/train.json" || ! -f "$DATA_DIR/valid.json" || ! -f "$DATA_DIR/test.json" || ! -f "$DATA_DIR/manifest.csv" || ! -f "$DATA_DIR/labels.json" ]]; then
  echo "Missing benchmark files under $DATA_DIR"
  echo "Read benchmarks/headct_rsna/README.md first."
  exit 1
fi

run "$PYTHON" "$ROOT/scripts/validate_headct_rsna_benchmark.py" \
  --data-dir "$DATA_DIR" \
  --sample-path-checks 100

if [[ "$DRY_RUN" != "1" ]]; then
  device_lower="$(printf '%s' "$DEVICE" | tr '[:upper:]' '[:lower:]')"
  case "$device_lower" in
    cpu)
      if [[ "$ALLOW_CPU" != "1" ]]; then
        echo "CPU benchmarking is disabled by default. Set ALLOW_CPU=1 for a tiny smoke test."
        exit 1
      fi
      ;;
    cuda*)
      if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "CUDA requested but nvidia-smi was not found. Use DEVICE=mps on macOS Metal, or ALLOW_CPU=1 DEVICE=cpu for a tiny smoke test."
        exit 1
      fi
      nvidia-smi
      ;;
    auto|"")
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
      elif [[ "$(uname -s)" == "Darwin" ]]; then
        echo "No nvidia-smi found; using PyTorch auto device selection (Metal/MPS when available)."
      elif [[ "$ALLOW_CPU" != "1" ]]; then
        echo "No CUDA GPU detected. Set DEVICE=mps on macOS Metal, or ALLOW_CPU=1 DEVICE=cpu for a tiny smoke test."
        exit 1
      fi
      ;;
    mps*|metal*)
      if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "Warning: DEVICE=$DEVICE requested outside macOS; PyTorch will validate availability."
      fi
      ;;
  esac
fi

if [[ "$DRY_RUN" != "1" ]]; then
  run uv sync
fi

extract_cmd=(
  uv run rate-extract
  --model pillar0 \
  --dataset rve_brain_ct \
  --all-splits \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --model-repo-id YalaLab/Pillar0-HeadCT \
  --ct-window-type all \
  --modality brain_ct \
  --output-dir "$CACHE_DIR" \
  data.train_json="$DATA_DIR/train.json" \
  data.valid_json="$DATA_DIR/valid.json" \
  data.test_json="$DATA_DIR/test.json" \
  data.cache_manifest="$DATA_DIR/manifest.csv"
)

if [[ -n "$NUM_GPUS" ]]; then
  extract_cmd+=(--num-gpus "$NUM_GPUS")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  extract_cmd+=(--max-samples "$MAX_SAMPLES")
fi

run "${extract_cmd[@]}"

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "Skipping evaluation because SKIP_EVAL=1"
  exit 0
fi

run uv run rate-evaluate \
  --model pillar0 \
  --checkpoint-dir "$CACHE_DIR" \
  --dataset-name rve_brain_ct \
  --labels-json "$DATA_DIR/labels.json" \
  --output-dir "$RESULTS_DIR" \
  hardware.device="$DEVICE" \
  evaluation.use_wandb=false \
  evaluation.use_pytorch=false
