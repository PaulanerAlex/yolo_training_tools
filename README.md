# Anleitung für Arne:

1. clonen `git clone https://github.com/PaulanerAlex/blenderproc_yolo_db_generator.git && cd blenderproc_yolo_db_generator`
2. `uv sync` ausführen
3. server_config.json einfügen
4. `./start_server.sh -f` ausführen. -f für die Ausführung in diesem Prozess (ohne einen neuen zu starten)



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

3. **Start the Server**:
   The start script will automatically run the server in the background using `nohup`.
   
   Start the server:
   ```bash
   ./start_server.sh
   ```
   
   You will be prompted for your **Nextcloud App Password** (which is hidden). After entering it, the server will start watching for datasets in the background.

   Logs will be written to `server.log`. You can view them with:
   ```bash
   tail -f server.log
   ```

   To stop the background server, you can run the script again, or use the `kill` command with the PID found in `server.pid`.

   If you prefer to run the server in the current process (foreground), you can use the `-f` or `--foreground` flag:
   ```bash
   ./start_server.sh -f
   ```

### Server Configuration

The auto-generated `server_config.json` file contains several parameters you can adjust:

- `nc_url`: Nextcloud WebDAV URL.
- `nc_user`: Nextcloud username.
- `input_path`: Nextcloud folder where zip datasets are uploaded.
- `output_path`: Nextcloud folder where trained weights and error logs will be uploaded.
- `idle_timeout`: Time in seconds the server will wait for a new dataset before exiting automatically. Set to `0` to disable the timeout and run the server indefinitely.

## Local Training

If you want to train on a dataset stored locally in `datasets/`, use the interactive trainer:

```bash
uv run python main.py
```

The program lists the folders in `datasets/`, asks which dataset to use, auto-detects the dataset YAML inside that folder, and then prompts for the base model, epochs, and image size. It uses the existing `YoloTrainer` class, so the same training behavior is reused for both local and server-based runs.

## How the Server Works

1. **Upload Dataset**: Zip your dataset folder (including a `config.yml` or `config.json` if desired) and upload it to the watched Nextcloud input folder.
2. **Auto-Training**: The server detects the `.zip` file, downloads it, unzips it, and immediately begins training the YOLO model according to your config.
3. **Finish & Cleanup**: 
   - **On Success**: The resulting `best.pt` weights file is uploaded to your Nextcloud output folder.
   - **On Failure**: A detailed `_error.txt` log is uploaded to the output folder.
   - The remote `.zip` dataset is then deleted to prevent duplicate runs, and local temporary files are cleaned up.
