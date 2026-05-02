import os
import pathlib
import random
from typing import Optional, List, Tuple
from ultralytics import YOLO
from .bb_tools import BoundingBoxVisualizer

class YoloTrainer:
    """Class to manage interactive or config-based YOLO model training."""
    
    def __init__(self, dataset_dir: str = "dataset", models_avail: Optional[List[str]] = None, config_path: Optional[str] = None, config: Optional[dict] = None):
        self.dataset_dir = pathlib.Path(dataset_dir)
        if models_avail is None:
            self.models_avail = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        else:
            self.models_avail = models_avail
            
        self.config = self._load_config(config_path) if config_path else {}
        if config:
            self.config.update(config)

    def _load_config(self, path: str) -> dict:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found at {path}")
        if p.suffix.lower() == '.json':
            import json
            with open(p, 'r') as f:
                return json.load(f)
        elif p.suffix.lower() in ('.yml', '.yaml'):
            import yaml
            with open(p, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .json or .yaml")

    def _select_model(self, default_model: str = "yolo11s.pt") -> str:
        print("--- MODEL SELECTION ---")
        print("Available models:")
        for i, name in enumerate(self.models_avail):
            print(f"{i+1}: {name}")
            
        inp = input(f"Which model do you want to use? (type in number, press Enter for default {default_model}): ")
        
        if inp.strip().isdigit() and 1 <= int(inp.strip()) <= len(self.models_avail):
            model_name = self.models_avail[int(inp.strip()) - 1]
        else:
            model_name = default_model
            
        print(f"Using model: {model_name}\n")
        return model_name

    def _select_dataset(self, default_dataset: str = "ellipse_recognition") -> str:
        print("--- DATASET SELECTION ---")
        
        if not self.dataset_dir.exists():
            raise ValueError(f"No datasets found in the '{self.dataset_dir}' directory.")
            
        dataset_avail = [d.name for d in self.dataset_dir.iterdir() if d.is_dir()]
        if not dataset_avail:
            raise ValueError(f"No datset folders found in the '{self.dataset_dir}' directory.")

        print("Available datasets:")
        for i, name in enumerate(dataset_avail):
            print(f"{i+1}: {name}")

        inp = input(f"Which dataset do you want to use? (type in number, press Enter for default {default_dataset}): ")

        if inp.strip().isdigit() and 1 <= int(inp.strip()) <= len(dataset_avail):
            dataset_name = dataset_avail[int(inp.strip()) - 1]
        else:
            dataset_name = default_dataset

        print(f"Using dataset: {dataset_name}\n")
        return dataset_name

    def train(self, epochs: int = 250, imgsz: int = 640, default_model: str = "yolo11s.pt", default_dataset: str = "ellipse_recognition") -> Tuple[object, str, str]:
        """Runs the training process, using config values if available, else interactive."""
        print("--- WELCOME TO YOLO MODEL TRAINER ---\n")
        
        if "model" not in self.config:
            model_name = self._select_model(default_model)
        else:
            model_name = self.config["model"]
            print(f"Using model selected from config: {model_name}")
            
        if "dataset" not in self.config:
            dataset_name = self._select_dataset(default_dataset)
        else:
            dataset_name = self.config["dataset"]
            print(f"Using dataset selected from config: {dataset_name}")
            
        epochs = self.config.get("epochs", epochs)
        imgsz = self.config.get("imgsz", imgsz)
        train_args = self.config.get("train_args", {})
        
        if "dataset_yaml_path" in self.config:
            dataset_yml_path = pathlib.Path(self.config["dataset_yaml_path"])
            if not dataset_yml_path.is_absolute():
                candidate_path = self.dataset_dir / dataset_name / dataset_yml_path
                if candidate_path.exists():
                    dataset_yml_path = candidate_path
                else:
                    candidate_path = self.dataset_dir / dataset_yml_path
                    if candidate_path.exists():
                        dataset_yml_path = candidate_path
        else:
            dataset_yml_path = self.dataset_dir / dataset_name / f"{dataset_name}.yml"

        # Normalize and patch dataset YAML contents when paths inside the YAML
        # point to absolute locations or otherwise won't resolve in this run.
        try:
            import yaml
            if dataset_yml_path and dataset_yml_path.exists():
                with open(dataset_yml_path, 'r') as yf:
                    dataset_yaml_content = yaml.safe_load(yf)

                modified = False
                if isinstance(dataset_yaml_content, dict):
                    # Ensure 'path' points to the dataset YAML parent (absolute)
                    if 'path' in dataset_yaml_content or 'train' in dataset_yaml_content:
                        dataset_yaml_content['path'] = str(dataset_yml_path.parent.absolute())
                        modified = True

                    # If train/val are absolute paths from another machine, convert them
                    if 'train' in dataset_yaml_content and isinstance(dataset_yaml_content['train'], str) and dataset_yaml_content['train'].startswith('/'):
                        dataset_yaml_content['train'] = os.path.basename(dataset_yaml_content['train'])
                        modified = True
                    if 'val' in dataset_yaml_content and isinstance(dataset_yaml_content['val'], str) and dataset_yaml_content['val'].startswith('/'):
                        dataset_yaml_content['val'] = os.path.basename(dataset_yaml_content['val'])
                        modified = True

                if modified:
                    with open(dataset_yml_path, 'w') as yf:
                        yaml.dump(dataset_yaml_content, yf)
                    print(f"Dynamically updated YAML path configurations inside {dataset_yml_path.name}")
        except Exception:
            # If YAML parsing/patching fails, continue and let the existence check raise a clear error
            pass

        if not dataset_yml_path.exists():
            raise FileNotFoundError(f"Dataset YAML file not found at {dataset_yml_path}")

        print("--- CONFIG SELECTION ---")
        print("Using default config settings. You can modify the training parameters in the script if needed.\n")

        model = YOLO(model_name)
        results = model.train(data=str(dataset_yml_path), epochs=epochs, imgsz=imgsz, **train_args)
        return results, model_name, dataset_name


class YoloTester:
    """Class to manage YOLO model testing and evaluation."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.visualizer = BoundingBoxVisualizer()
        
    def test(self, project_name: str, path_to_model: str = "runs/detect/train/weights/best.pt"):
        """Evaluates a model randomly on a validation image."""
        model = YOLO(path_to_model) 

        val_path = self.dataset_dir / project_name / "images" / "val"
        if not val_path.exists():
            # For OBB logic matching original
            val_path = self.dataset_dir / project_name / "images" / "val"
            
        if not val_path.exists():
             raise FileNotFoundError(f"Validation path not found: {val_path}")
             
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = [f for f in val_path.iterdir() if f.suffix.lower() in valid_extensions]
        
        if not images:
            raise ValueError(f"No images found in {val_path}")

        img_path = random.choice(images)
        print(f"Testing model on image: {img_path}")
        
        results = model(str(img_path), show=True)

        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                orig_xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
                confs = result.obb.conf.cpu().numpy()
                
                print(f"Bounding box corners of {len(orig_xyxyxyxy)} bounding boxes: {orig_xyxyxyxy}")
                print(f"Confidence scores: {confs}")
                
                # To list format for plot
                corners_list = []
                for box in orig_xyxyxyxy:
                    # box should be a array of shape (4, 2)
                    pts = [(pt[0], pt[1]) for pt in box]
                    corners_list.append(pts)
                
                self.visualizer.show(img_path, corners_list, multiple=True)
            elif hasattr(result, 'boxes') and result.boxes is not None:
                # Add basic standard bounding box support
                orig_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                print(f"Bounding box corners of {len(orig_xyxy)} bounding boxes: {orig_xyxy}")
                print(f"Confidence scores: {confs}")
                
                 # Convert to [top-left, top-right, bottom-right, bottom-left] format
                corners_list = []
                for box in orig_xyxy:
                    x1, y1, x2, y2 = box
                    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    corners_list.append(corners)
                    
                self.visualizer.show(str(img_path), corners_list, multiple=True)
            else:
                print("No bounding boxes detected.")

if __name__ == "__main__":
    trainer = YoloTrainer(dataset_dir="rt_26_dataset_v1", config_path="rt_26_dataset_v1/config.yaml")
    results, model_name, dataset_name = trainer.train()
    
    tester = YoloTester()
    tester.test(project_name=dataset_name, path_to_model=f"runs/detect/train/weights/best.pt")
