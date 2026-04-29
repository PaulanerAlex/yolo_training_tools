#!/bin/bash

echo "Syncing uv environment..."
uv sync

echo "Starting Nextcloud YOLO Training Server..."
uv run src/yolo_training_tools/server.py
