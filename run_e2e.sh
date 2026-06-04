#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
AVD="${AVD:-Pixel_8a}"
APK="${ROOT}/example/testing_app/build/app/outputs/flutter-apk/app-debug.apk"
APPIUM_URL="${APPIUM_URL:-http://127.0.0.1:4723}"
APP_PACKAGE=""
started_emulator=false

export ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"

emulator_bin="$(command -v emulator || true)"
if [[ -z "$emulator_bin" || "$emulator_bin" == *"/sdk/tools/emulator" ]]; then
  emulator_bin="$ANDROID_HOME/emulator/emulator"
fi

find_aapt() {
  find "$ANDROID_HOME/build-tools" -name aapt 2>/dev/null | sort -V | tail -1
}

apk_package_id() {
  local apk="$1" aapt
  if [[ ! -f "$apk" ]]; then
    echo "APK not found: $apk" >&2
    exit 1
  fi
  aapt="$(find_aapt)"
  if [[ -z "$aapt" ]]; then
    echo "aapt not found under $ANDROID_HOME/build-tools" >&2
    exit 1
  fi
  "$aapt" dump badging "$apk" | awk -F"'" '/package: name=/{print $2; exit}'
}

wait_for_boot() {
  adb wait-for-device
  until [[ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; do
    sleep 2
  done
}

device_ready() {
  adb devices | awk 'NR>1 && $2=="device" { found=1 } END { exit !found }'
}

start_emulator() {
  if device_ready; then
    echo "Emulator already running"
    return
  fi
  started_emulator=true
  echo "Starting emulator: $AVD"
  "$emulator_bin" -avd "$AVD" -no-snapshot-load >/dev/null 2>&1 &
  wait_for_boot
}

uninstall_app() {
  if [[ -z "$APP_PACKAGE" ]]; then
    return
  fi
  if adb shell pm path "$APP_PACKAGE" >/dev/null 2>&1; then
    echo "Uninstalling $APP_PACKAGE"
    adb uninstall "$APP_PACKAGE" >/dev/null
  fi
}

stop_emulator() {
  if [[ "$started_emulator" != true ]]; then
    return
  fi
  echo "Stopping emulator"
  adb emu kill >/dev/null 2>&1 || true
}

cleanup() {
  uninstall_app
  stop_emulator
}

trap cleanup EXIT

build_apk() {
  echo "Building debug APK"
  (cd "$ROOT/example/testing_app" && flutter build apk --debug)
}

install_apk() {
  echo "Installing APK"
  adb install -r "$APK"
}

appium_ready() {
  curl -sf "$APPIUM_URL/status" >/dev/null 2>&1
}

start_appium() {
  if appium_ready; then
    echo "Appium already running at $APPIUM_URL"
    return
  fi
  echo "Starting Appium"
  appium >/tmp/ai_e2e_appium.log 2>&1 &
  for _ in $(seq 1 30); do
    if appium_ready; then
      return
    fi
    sleep 1
  done
  echo "Appium failed to start. See /tmp/ai_e2e_appium.log"
  exit 1
}

launch_app() {
  adb shell monkey -p "$APP_PACKAGE" -c android.intent.category.LAUNCHER 1 >/dev/null
}

main() {
  start_emulator
  build_apk
  APP_PACKAGE="$(apk_package_id "$APK")"
  echo "APK package: $APP_PACKAGE"
  install_apk
  start_appium
  launch_app
  echo "Running E2E test"
  (cd "$ROOT" && APP_PACKAGE="$APP_PACKAGE" uv run python main.py)
  echo "E2E test finished"
}

main "$@"
