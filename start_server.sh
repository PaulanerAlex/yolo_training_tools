#!/usr/bin/env bash
set -euo pipefail

PID_FILE="server.pid"
LOG_FILE="server.log"

echo "Preparing to start YOLO Training Server..."

# If an old process exists, kill it first
if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE")
  if ps -p "$OLD_PID" > /dev/null 2>&1; then
    echo "Existing server found (PID: $OLD_PID). Stopping it..."
    kill "$OLD_PID" || true
    # Wait for the process to exit
    while ps -p "$OLD_PID" > /dev/null 2>&1; do sleep 0.5; done
  fi
  rm -f "$PID_FILE"
fi

echo "Syncing uv environment..."
uv sync

# We ask for the password *before* backgrounding and pass it to the
# server process via an environment variable.
# server.py must read this env var (e.g. os.environ["NC_APP_PASSWORD"]).
if [[ -z "${NC_APP_PASSWORD:-}" ]]; then
  read -rsp "Password (NC_APP_PASSWORD): " NC_APP_PASSWORD
  echo
fi

echo "Starting YOLO Training Server in the background using nohup."
echo "Logs will be written to ${LOG_FILE}."

nohup env NC_APP_PASSWORD="${NC_APP_PASSWORD}" uv run src/yolo_training_tools/server.py > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"

echo "Server started successfully (PID: $NEW_PID)."
echo "To view logs: tail -f $LOG_FILE"
echo "To stop the server, run this script again or run: kill $NEW_PID"
