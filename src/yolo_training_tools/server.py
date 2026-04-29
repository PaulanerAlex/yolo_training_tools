import os
import time
import shutil
import traceback
import getpass
import zipfile
import pathlib
import json
from webdav3.client import Client
from .training_tools import YoloTrainer

def run_server():
    print("--- NEXTCLOUD YOLO TRAINING SERVER INIT ---")
    
    config_path = pathlib.Path("server_config.json")
    if not config_path.exists():
        default_config = {
            "nc_url": "https://cloud.example.com/remote.php/webdav/",
            "nc_user": "your_username",
            "input_path": "/yolo_datasets_in/",
            "output_path": "/yolo_models_out/"
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
    
    if not all([nc_url, nc_user, input_path, output_path]):
        print("Invalid server_config.json. Please ensure nc_url, nc_user, input_path, and output_path are set.")
        return

    print(f"Loaded config for user '{nc_user}' at '{nc_url}'")
    print(f"Watching input path: {input_path}")
    print(f"Outputting to: {output_path}")

    nc_pass = getpass.getpass("Nextcloud App Password: ")
    
    # Configure WebDAV client
    options = {
        'webdav_hostname': nc_url,
        'webdav_login': nc_user,
        'webdav_password': nc_pass
    }
    client = Client(options)

    # Ensure output directory exists (won't crash if it already does)
    try:
        client.mkdir(output_path)
    except:
        pass

    local_dataset_dir = pathlib.Path("dataset_server_temp")
    local_dataset_dir.mkdir(exist_ok=True)
    
    print("\nServer running. Waiting for new .zip datasets...")

    while True:
        try:
            # Check for generic zip datasets
            files = client.list(input_path)

            for file_name in files:
                if not file_name.endswith('.zip'):
                    continue
                    
                remote_file_path = f"{input_path.rstrip('/')}/{file_name}"
                local_zip_path = local_dataset_dir / file_name
                dataset_name = file_name.replace('.zip', '')
                
                print(f"New dataset detected: {file_name}. Downloading...")
                client.download_sync(remote_file_path, str(local_zip_path))
                
                # Unzip dataset
                print(f"Unzipping {file_name}...")
                extract_dir = local_dataset_dir / dataset_name
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Look for configuration (assuming the unzipped content has a YAML or JSON named config)
                config_path = None
                for conf in ["config.yml", "config.yaml", "config.json"]:
                    potential_conf = extract_dir / conf
                    if potential_conf.exists():
                        config_path = str(potential_conf)
                        break
                
                # Initialize train process
                print(f"Starting training on {dataset_name}..." + (" (With found config)" if config_path else ""))
                
                # Pass the unzipped dir as dataset_dir
                trainer = YoloTrainer(dataset_dir=str(local_dataset_dir), config_path=config_path)
                try:
                    # Overwrite the dataset property to point directly to the new unzipped dataset
                    if trainer.config:
                        trainer.config["dataset"] = dataset_name
                    
                    results, resulting_model_name, _ = trainer.train()
                    
                    # Upload successful trained weights (Usually Ultralytics puts it at runs/detect/train/weights/best.pt)
                    # For a robust approach, we need to locate the best.pt from the training results.
                    # As a shortcut, we're assuming ultralytics default output runs/detect/train/weights/best.pt.
                    
                    if hasattr(results, 'save_dir'):
                        best_weights_path = os.path.join(results.save_dir, "weights", "best.pt")
                    else:
                        best_weights_path = "runs/detect/train/weights/best.pt"

                    if os.path.exists(best_weights_path):
                        remote_model_dest = f"{output_path.rstrip('/')}/{dataset_name}_best.pt"
                        print(f"Uploading trained model to {remote_model_dest}...")
                        client.upload_sync(remote_path=remote_model_dest, local_path=best_weights_path)
                    
                    # Delete remote dataset
                    print(f"Cleaning up: deleting original dataset on remote: {remote_file_path}")
                    client.clean(remote_file_path)
                    
                except Exception as e:
                    # Catch training exceptions
                    error_msg = traceback.format_exc()
                    print(f"Training failed. Uploading log. Error: {e}")
                    log_file = f"{dataset_name}_error.txt"
                    with open(log_file, "w") as f:
                        f.write(error_msg)
                        
                    remote_err_dest = f"{output_path.rstrip('/')}/{log_file}"
                    client.upload_sync(remote_path=remote_err_dest, local_path=log_file)
                    
                finally:
                    # Clean up local files
                    if local_zip_path.exists():
                        os.remove(local_zip_path)
                    if extract_dir.exists():
                        shutil.rmtree(extract_dir)
        
        except Exception as e:
            # Exception with client itself (e.g. connection lost) -> Don't crash, just log and try again later
            print(f"Connection or loop error: {e}")

        # Sleep for a bit before checking again
        time.sleep(30)

if __name__ == "__main__":
    run_server()