#!/bin/bash

echo "Starting screen session..."
echo | <screen>



echo "Syncing uv environment..."
uv sync

echo "Starting Nextcloud YOLO Training Server. To exit, type strg + A and then D ..."
uv run src/yolo_training_tools/server.py
