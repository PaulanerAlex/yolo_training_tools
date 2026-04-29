# YOLO Training Tools

A complete toolkit for interactive and automated YOLO model training, including an automated Nextcloud-based WebDAV training server.

## Features
- **YoloTrainer & YoloTester**: High-level API for interactive and config-driven YOLO training/testing.
- **Bounding Box Tools**: Utilities for interactive bounding box annotations.
- **Nextcloud Training Server**: A daemon that fetches `.zip` datasets from a Nextcloud share, trains YOLO models automatically, and uploads the results (or error logs) back to Nextcloud.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd yolo_training_tools
   ```

2. **Initialize the Environment**:
   This project uses `uv` for dependency management. Install your dependencies and create the virtual environment:
   ```bash
   uv sync
   ```

## Deploying the Automated Server

The setup is extremely streamlined. You don't need to answer a bunch of questions every time. You use a configuration file for non-sensitive data and only need the app password.

1. **Run the Server for the First Time**:
   ```bash
   ./start_server.sh
   ```
   This will auto-generate a `server_config.json` file in your directory and exit. 

2. **Edit `server_config.json`**:
   Open the file and place your Nextcloud URL, username, input folder, and output folder there. Those parameters are not sensitive and do not need encryption.

3. **Start the Server in the Background**:
   Since the server needs to run constantly, we recommend using `screen` to keep it running even if you close your terminal.
   
   Start a new screen session:
   ```bash
   screen -S yolo-server
   ```
   
   Run the startup script:
   ```bash
   ./start_server.sh
   ```
   
   You will only be prompted strictly for the **Nextcloud App Password** (which is hidden). After entering it, the server will start watching for datasets.
   
   To **detach** from the screen session (leaving it running in the background), press: `Ctrl+A` then `D`.
   
   To **re-attach** to the running server later (e.g., to view logs or stop it):
   ```bash
   screen -r yolo-server
   ```

## How the Server Works

1. **Upload Dataset**: Zip your dataset folder (including a `config.yml` or `config.json` if desired) and upload it to the watched Nextcloud input folder.
2. **Auto-Training**: The server detects the `.zip` file, downloads it, unzips it, and immediately begins training the YOLO model according to your config.
3. **Finish & Cleanup**: 
   - **On Success**: The resulting `best.pt` weights file is uploaded to your Nextcloud output folder.
   - **On Failure**: A detailed `_error.txt` log is uploaded to the output folder.
   - The remote `.zip` dataset is then deleted to prevent duplicate runs, and local temporary files are cleaned up.
