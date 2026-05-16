#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Workspace: $ROOT"
echo

for repo in rate-evals rave rate pillar-finetune pillar-pretrain; do
  if [[ -d "$repo/.git" ]]; then
    commit="$(git -C "$repo" rev-parse --short HEAD)"
    echo "ok: $repo cloned at $commit"
  else
    echo "missing: $repo"
    exit 1
  fi
done

echo
if command -v uv >/dev/null 2>&1; then
  uv --version
else
  echo "missing: uv"
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "notice: nvidia-smi not found; paper-scale runs need a CUDA/NVIDIA host"
fi

echo
echo "Hugging Face access check:"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli whoami || true
else
  echo "huggingface-cli is not installed until the uv envs are synced"
fi

echo
echo "Next: read PILLAR0_REPLICATION.md and start with the Merlin abdomen CT RATE-Eval path."
