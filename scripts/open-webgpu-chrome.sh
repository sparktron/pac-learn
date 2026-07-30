#!/usr/bin/env bash
# WHAT: Launch Google Chrome with WebGPU-capable GPU flags and open a URL
#       (default: the local CNN/WebGPU benchmark page).
# WHY:  On Linux, Chrome with only --use-gl=angle exposes navigator.gpu but
#       requestAdapter() returns null. WebGPU needs --enable-unsafe-webgpu plus
#       Vulkan. Chromium also ignores new flags if an existing Chrome process
#       is already running, so this script refuses to attach to a live session.
# USAGE:
#   ./scripts/open-webgpu-chrome.sh
#   ./scripts/open-webgpu-chrome.sh 'http://127.0.0.1:5173/?cnnWebglBenchmark=1'
#   DEV_URL=http://127.0.0.1:5173 ./scripts/open-webgpu-chrome.sh
# Privileges: none. Requires google-chrome-stable on PATH.
# REVERT / UNDO: Close the Chrome window this script starts. Desktop-wide flags
# live in ~/.local/share/applications/google-chrome.desktop (see AI_readme).
set -euo pipefail

CHROME_BIN="${CHROME_BIN:-google-chrome-stable}"
DEV_URL="${DEV_URL:-http://127.0.0.1:5173}"
DEFAULT_PATH='/?cnnWebglBenchmark=1'
TARGET_URL="${1:-${DEV_URL}${DEFAULT_PATH}}"

# These match the host desktop launcher. Keep WebGL ANGLE + enable Vulkan WebGPU.
CHROME_GPU_FLAGS=(
  --ignore-gpu-blocklist
  --enable-unsafe-webgpu
  --enable-features=Vulkan,UseSkiaRenderer
  --use-gl=angle
  --disable-gpu-sandbox
)

if ! command -v "$CHROME_BIN" >/dev/null 2>&1; then
  echo "[abort] ${CHROME_BIN} not found on PATH" >&2
  exit 1
fi

# Existing Chrome reuses the running browser and drops CLI GPU flags.
if pgrep -f '/opt/google/chrome/chrome|/usr/bin/google-chrome' >/dev/null 2>&1; then
  echo "[abort] Chrome is already running." >&2
  echo "        Quit every Chrome window (so no chrome process remains), then rerun." >&2
  echo "        New tabs in a live session will not pick up WebGPU flags." >&2
  exit 2
fi

echo "[info] launching Chrome with WebGPU/Vulkan flags -> ${TARGET_URL}"
exec "$CHROME_BIN" "${CHROME_GPU_FLAGS[@]}" "$TARGET_URL"
