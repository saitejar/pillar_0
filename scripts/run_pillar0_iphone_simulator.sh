#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="$ROOT/ios/test_pillar_0"
PROJECT="$APP_ROOT/test_pillar_0.xcodeproj"
SCHEME="${PILLAR0_IOS_SCHEME:-test_pillar_0}"
BUNDLE_ID="${PILLAR0_IOS_BUNDLE_ID:-test-pillar-0.test-pillar-0}"
DEVICE_NAME="${PILLAR0_SIM_DEVICE_NAME:-iPhone 15}"
DERIVED_DATA="${PILLAR0_DERIVED_DATA:-${TMPDIR:-/tmp}/pillar0_test_pillar_0_derived_data}"
SCREENSHOT="$ROOT/artifacts/pillar0_headct_coreml/iphone_simulator_latest.png"
WAIT_SECONDS="${PILLAR0_IOS_WAIT_SECONDS:-80}"
XCODEBUILD_QUIET_ARGS=()
if [[ "${PILLAR0_XCODEBUILD_VERBOSE:-0}" != "1" ]]; then
  XCODEBUILD_QUIET_ARGS=(-quiet)
fi

"$ROOT/scripts/prepare_pillar0_mobile_artifacts.sh"

find_booted_udid() {
  xcrun simctl list devices -j | python3 -c 'import json, sys
devices = json.load(sys.stdin).get("devices", {})
for runtime_devices in devices.values():
    for device in runtime_devices:
        if device.get("state") == "Booted" and device.get("isAvailable", True):
            print(device["udid"])
            raise SystemExit(0)'
}

find_named_udid() {
  local name="$1"
  xcrun simctl list devices -j | python3 -c 'import json, sys
name = sys.argv[1]
devices = json.load(sys.stdin).get("devices", {})
for runtime_devices in devices.values():
    for device in runtime_devices:
        if device.get("name") == name and device.get("state") == "Shutdown" and device.get("isAvailable", True):
            print(device["udid"])
            raise SystemExit(0)' "$name"
}

UDID="${PILLAR0_SIM_UDID:-$(find_booted_udid)}"
if [[ -z "$UDID" ]]; then
  UDID="$(find_named_udid "$DEVICE_NAME")"
  if [[ -z "$UDID" ]]; then
    echo "No booted simulator found and no available '$DEVICE_NAME' simulator found." >&2
    echo "Boot a simulator manually, or set PILLAR0_SIM_UDID=<udid>." >&2
    exit 1
  fi
  echo "Booting simulator $DEVICE_NAME ($UDID)..."
  xcrun simctl boot "$UDID" || true
  xcrun simctl bootstatus "$UDID" -b
fi

echo "Building iOS app for simulator $UDID..."
rm -rf "$DERIVED_DATA/Build/Products/Debug-iphonesimulator/test_pillar_0.app"
xcodebuild \
  "${XCODEBUILD_QUIET_ARGS[@]}" \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination "platform=iOS Simulator,id=$UDID" \
  -configuration Debug \
  -derivedDataPath "$DERIVED_DATA" \
  build

APP="$DERIVED_DATA/Build/Products/Debug-iphonesimulator/test_pillar_0.app"
echo "Installing $APP..."
xcrun simctl terminate "$UDID" "$BUNDLE_ID" >/dev/null 2>&1 || true
xcrun simctl install "$UDID" "$APP"

echo "Launching $BUNDLE_ID..."
xcrun simctl launch "$UDID" "$BUNDLE_ID"

if [[ "${PILLAR0_IOS_NO_WAIT:-0}" != "1" ]]; then
  echo "Waiting ${WAIT_SECONDS}s for full-volume inference to finish..."
  sleep "$WAIT_SECONDS"
  mkdir -p "$(dirname "$SCREENSHOT")"
  xcrun simctl io "$UDID" screenshot "$SCREENSHOT"
  echo "Saved simulator screenshot: $SCREENSHOT"
fi
