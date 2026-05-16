#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT/artifacts/pillar0_headct_coreml"
DEVICE_NAME="${DEVICE_NAME:-Pillar0-iPhone13}"
DEVICE_TYPE="${DEVICE_TYPE:-com.apple.CoreSimulator.SimDeviceType.iPhone-13}"
RUNTIME="${RUNTIME:-com.apple.CoreSimulator.SimRuntime.iOS-17-5}"
COMPUTE_UNITS="${COMPUTE_UNITS:-cpuOnly}"

MODEL_PACKAGE="$ARTIFACT_DIR/Pillar0HeadCTVision_int8.mlpackage"
MODEL_COMPILED="$ARTIFACT_DIR/Pillar0HeadCTVision_int8.mlmodelc"
RUNNER="$ARTIFACT_DIR/Pillar0SimulatorValidate"
INPUT_RAW="$ARTIFACT_DIR/simulator_sample_windowed_headct_f16.raw"
EXPECTED_RAW="$ARTIFACT_DIR/simulator_expected_torchscript_f32.raw"

if [[ ! -d "$MODEL_COMPILED" ]]; then
  xcrun coremlcompiler compile "$MODEL_PACKAGE" "$ARTIFACT_DIR"
fi

SDK="$(xcrun --sdk iphonesimulator --show-sdk-path)"
xcrun swiftc \
  -sdk "$SDK" \
  -target arm64-apple-ios17.5-simulator \
  "$ROOT/ios/Pillar0SimulatorValidate.swift" \
  -o "$RUNNER"

UDID="$(
  xcrun simctl list devices available |
    awk -v name="$DEVICE_NAME" '$0 ~ name {gsub(/[()]/, "", $2); print $2; exit}'
)"

if [[ -z "$UDID" ]]; then
  UDID="$(xcrun simctl create "$DEVICE_NAME" "$DEVICE_TYPE" "$RUNTIME")"
fi

xcrun simctl boot "$UDID" || true
xcrun simctl bootstatus "$UDID" -b
xcrun simctl spawn "$UDID" "$RUNNER" "$MODEL_COMPILED" "$INPUT_RAW" "$EXPECTED_RAW" "$COMPUTE_UNITS"
