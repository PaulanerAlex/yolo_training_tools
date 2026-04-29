#!/usr/bin/env bash
set -euo pipefail

# Start the server in a detached GNU screen session.
# -S: session name
# -dm: start detached
SESSION_NAME="yolo-training-server"

echo "Starting screen session (${SESSION_NAME})..."

# Create (or replace) a detached session running this script's server command.
# If an old session exists, kill it first to avoid 'There is a screen on' errors.
if command -v screen >/dev/null 2>&1; then
  if screen -list | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "Existing screen session found. Stopping it..."
    screen -S "${SESSION_NAME}" -X quit || true
  fi

  echo "Syncing uv environment..."
  uv sync

  echo "Starting Nextcloud YOLO Training Server in screen."
  echo "Attach with: screen -r ${SESSION_NAME}"
  echo "Detach from screen with: Ctrl-A then D"
  screen -S "${SESSION_NAME}" -dm uv run src/yolo_training_tools/server.py
else
  echo "Error: 'screen' is not installed or not in PATH. Install it (e.g. 'sudo apt install screen') or remove the screen wrapper." >&2
  exit 1
fi
