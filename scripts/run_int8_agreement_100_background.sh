#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RATE_EVALS="$ROOT/rate-evals"
ARTIFACT_DIR="$ROOT/artifacts/pillar0_headct_coreml"
mkdir -p "$ARTIFACT_DIR"

start_split() {
  local split="$1"
  local output="$ARTIFACT_DIR/int8_split_agreement_100_${split}_cpu.json"
  local progress="$ARTIFACT_DIR/int8_split_agreement_100_${split}_cpu.jsonl"
  local log="$ARTIFACT_DIR/int8_split_agreement_100_${split}_cpu.log"
  local pidfile="$ARTIFACT_DIR/int8_split_agreement_100_${split}_cpu.pid"

  local session="pillar0_int8_${split}"
  if screen -list | grep -q "[.]${session}[[:space:]]"; then
    echo "$split already running screen=$session"
    return
  fi

  screen -dmS "$session" bash -lc "
    cd '$RATE_EVALS'
    env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
      uv run --no-sync python -u ../scripts/validate_pillar0_coreml_splits.py \
        --splits '$split' \
        --samples-per-split 100 \
        --compute-units CPU_ONLY \
        --input-dtype float16 \
        --output '$output' \
        --progress-jsonl '$progress'
  " >"$log" 2>&1

  echo "$session" >"$pidfile"
  echo "$split started screen=$session log=$log"
}

start_split train
start_split valid
start_split test
