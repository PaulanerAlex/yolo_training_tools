import os
import time
import shutil
import traceback
import getpass
import zipfile
import pathlib
import json
import sys
from webdav3.client import Client

# Add the src directory to sys.path if running directly to resolve imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from yolo_training_tools.training_tools import YoloTrainer

def run_server():
    print("--- NEXTCLOUD YOLO TRAINING SERVER INIT ---")
    
    config_path = pathlib.Path("server_config.json")
    if not config_path.exists():
        default_config = {
            "nc_url": "https://cloud.example.com/remote.php/webdav/",
            "nc_user": "your_username",
            "input_path": "/yolo_datasets_in/",
            "output_path": "/yolo_models_out/",
            "idle_timeout": 0
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)
        print(f"Created default configuration at {config_path.absolute()}")
        print("Please edit it with your Nextcloud URL, username, and paths, then run the script again.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
        
    nc_url = config.get("nc_url")
    nc_user = config.get("nc_user")
    input_path = config.get("input_path")
    output_path = config.get("output_path")
    idle_timeout = int(config.get("idle_timeout", 0))
    
    if not all([nc_url, nc_user, input_path, output_path]):
        print("Invalid server_config.json. Please ensure nc_url, nc_user, input_path, and output_path are set.")
        return

    print(f"Loaded config for user '{nc_user}' at '{nc_url}'")
    print(f"Watching input path: {input_path}")
    print(f"Outputting to: {output_path}")
    if idle_timeout > 0:
        print(f"Idle timeout set to {idle_timeout} seconds.")
    else:
        print("Idle timeout not set. Server will run indefinitely.")

    nc_pass = os.getenv("NC_APP_PASSWORD")
    if not nc_pass:
        nc_pass = getpass.getpass("Nextcloud App Password: ")
    
    # Configure WebDAV client
    options = {
        'webdav_hostname': nc_url,
        'webdav_login': nc_user,
        'webdav_password': nc_pass
    }
    client = Client(options)

    # Ensure input and output directories exist (won't crash if they already do)
    try:
        client.mkdir(input_path)
    except:
        pass

    try:
        client.mkdir(output_path)
    except:
        pass

    local_dataset_dir = pathlib.Path("dataset_server_temp")
    local_dataset_dir.mkdir(exist_ok=True)
    
    print("\nServer running. Waiting for new .zip datasets...")
    idle_start_time = None

    while True:
        found_dataset = False
        try:
            # Check for generic zip datasets
            files = client.list(input_path)

            for file_name in files:
                if not file_name.endswith('.zip'):
                    continue
                    
                found_dataset = True
                remote_file_path = f"{input_path.rstrip('/')}/{file_name}"
                local_zip_path = local_dataset_dir / file_name
                dataset_name = file_name.replace('.zip', '')
                
                extract_dir = local_dataset_dir / dataset_name
                try:
                    print(f"New dataset detected: {file_name}. Downloading...")
                    client.download_sync(remote_file_path, str(local_zip_path))
                    
                    # Unzip dataset
                    print(f"Unzipping {file_name}...")
                    if extract_dir.exists():
                        shutil.rmtree(extract_dir)
                    
                    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

                    # Look for configuration recursively in case it's in a subdirectory
                    config_path = None
                    for root_dir, _, unzipped_files in os.walk(extract_dir):
                        for f in unzipped_files:
                            if f.lower() in ["config.yml", "config.yaml", "config.json"]:
                                config_path = str(pathlib.Path(root_dir) / f)
                                break
                        if config_path:
                            break
                    
                    # Look for the actual YOLO dataset YAML recursively (excluding configs)
                    dataset_yaml_path = None
                    dataset_actual_name = dataset_name
                    for root_dir, _, unzipped_files in os.walk(extract_dir):
                        for f in unzipped_files:
                            if (f.endswith(".yml") or f.endswith(".yaml")) and f.lower() not in ["config.yml", "config.yaml"]:
                                dataset_yaml_path = pathlib.Path(root_dir) / f
                                dataset_actual_name = dataset_yaml_path.stem
                                break
                        if dataset_yaml_path:
                            break
                    
                    # Find the parent folder of the dataset folder 
                    # (since trainer appends /dataset_name/dataset_name.yml)
                    if dataset_yaml_path:
                        base_dataset_dir = str(dataset_yaml_path.parent.parent)
                        
                        # Read the dataset YAML to dynamically update absolute paths if needed
                        try:
                            import yaml
                            with open(dataset_yaml_path, 'r') as yf:
                                dataset_yaml_content = yaml.safe_load(yf)
                            
                            modified = False
                            if 'path' in dataset_yaml_content or 'train' in dataset_yaml_content:
                                dataset_yaml_content['path'] = str(dataset_yaml_path.parent.absolute())
                                modified = True
                                
                            if 'train' in dataset_yaml_content and str(dataset_yaml_content['train']).startswith('/'):
                                 dataset_yaml_content['train'] = os.path.basename(dataset_yaml_content['train'])
                                 modified = True
                            if 'val' in dataset_yaml_content and str(dataset_yaml_content['val']).startswith('/'):
                                 dataset_yaml_content['val'] = os.path.basename(dataset_yaml_content['val'])
                                 modified = True

                            if modified:
                                with open(dataset_yaml_path, 'w') as yf:
                                    yaml.dump(dataset_yaml_content, yf)
                                print(f"Dynamically updated YAML path configurations inside {dataset_yaml_path.name}")
                        except Exception as yaml_err:
                            print(f"Failed parsing dataset YAML for path overrides: {yaml_err}")

                    else:
                        base_dataset_dir = str(local_dataset_dir)

                    # Initialize train process
                    print(f"Starting training on {dataset_actual_name}..." + (" (With found config)" if config_path else ""))
                    trainer = YoloTrainer(dataset_dir=base_dataset_dir, config_path=config_path)
                    # In a server environment, we cannot ask for interactive input.
                    if not trainer.config:
                        trainer.config = {}
                    
                    # Overwrite the dataset property to point directly to the located dataset name
                    trainer.config["dataset"] = dataset_actual_name
                    if dataset_yaml_path:
                        trainer.config["dataset_yaml_path"] = str(dataset_yaml_path)
                    
                    # Ensure a model is selected without interactive prompt
                    if "model" not in trainer.config:
                        trainer.config["model"] = "yolo11s.pt"
                        print("No model specified in config, defaulting to yolo11s.pt")
                    
                    results, resulting_model_name, _ = trainer.train()
                    
                    # Compress the entire run folder and upload
                    if hasattr(results, 'save_dir') and results.save_dir:
                        run_dir = str(results.save_dir)
                    else:
                        run_dir = "runs/detect/train"

                    if os.path.exists(run_dir):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        local_zip_output_base = os.path.join(str(local_dataset_dir), f"{dataset_name}_run_{timestamp}")
                        
                        print(f"Compressing run directory {run_dir}...")
                        shutil.make_archive(local_zip_output_base, 'zip', run_dir)
                        
                        local_zip_output = f"{local_zip_output_base}.zip"
                        remote_model_dest = f"{output_path.rstrip('/')}/{dataset_name}_run_{timestamp}.zip"
                        
                        print(f"Uploading trained run to {remote_model_dest}...")
                        client.upload_sync(remote_path=remote_model_dest, local_path=local_zip_output)
                        
                        # Clean up the generated zip file after upload
                        if os.path.exists(local_zip_output):
                            os.remove(local_zip_output)
                    
                    # Delete remote dataset
                    print(f"Cleaning up: deleting original dataset on remote: {remote_file_path}")
                    client.clean(remote_file_path)
                    
                except Exception as e:
                    # Catch processing/training exceptions
                    error_msg = traceback.format_exc()
                    print(f"Dataset processing/training failed. Uploading log. Error: {e}")
                    log_file = f"{dataset_name}_error.txt"
                    with open(log_file, "w") as f:
                        f.write(error_msg)
                        
                    remote_err_dest = f"{output_path.rstrip('/')}/{log_file}"
                    try:
                        client.upload_sync(remote_path=remote_err_dest, local_path=log_file)
                    except Exception as upload_err:
                        print(f"Failed to upload error log to remote: {upload_err}")
                    if os.path.exists(log_file):
                        os.remove(log_file)
                    
                    # Clean up the broken remote file to avoid an infinite loop
                    try:
                        print(f"Cleaning up: deleting broken dataset on remote: {remote_file_path}")
                        client.clean(remote_file_path)
                    except Exception as clean_err:
                        print(f"Failed to delete broken dataset on remote: {clean_err}")
                    
                finally:
                    # Clean up local files
                    if local_zip_path.exists():
                        os.remove(local_zip_path)
                    if extract_dir.exists():
                        shutil.rmtree(extract_dir)
        
        except Exception as e:
            # Exception with client itself (e.g. connection lost) -> Don't crash, just log and try again later
            print(f"Connection or loop error: {e}")

        if found_dataset:
            # We processed a dataset. Reset the idle timer.
            idle_start_time = None
        else:
            # No dataset was found this iteration.
            if idle_start_time is None:
                # Start the timer when we confirm no dataset is processing
                idle_start_time = time.time()
            elif idle_timeout > 0 and (time.time() - idle_start_time) > idle_timeout:
                print(f"No new datasets found for {idle_timeout} seconds. Exiting...")
                break

        # Sleep for a bit before checking again to avoid rate limits
        poll_interval = int(config.get("poll_interval", 30))
        time.sleep(poll_interval)

if __name__ == "__main__":
    run_server()