#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="$ROOT/patches/rate-evals-mobile.patch"
RATE_EVALS="$ROOT/rate-evals"

if [[ ! -d "$RATE_EVALS/.git" ]]; then
  echo "Missing rate-evals submodule. Run: git submodule update --init --recursive" >&2
  exit 1
fi

if [[ ! -f "$PATCH" ]]; then
  echo "Missing patch file: $PATCH" >&2
  exit 1
fi

if git -C "$RATE_EVALS" apply --check "$PATCH" >/dev/null 2>&1; then
  echo "Applying Mac/mobile patch to rate-evals..."
  git -C "$RATE_EVALS" apply "$PATCH"
elif git -C "$RATE_EVALS" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "rate-evals Mac/mobile patch already applied."
else
  echo "rate-evals has local changes that do not match the packaged patch." >&2
  echo "Inspect with: git -C rate-evals status --short" >&2
  exit 1
fi

