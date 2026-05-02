from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from yolo_training_tools.training_tools import YoloTrainer


def _prompt_choice(prompt: str, options: list[str], default_index: int = 0) -> str:
    print(prompt)
    for index, option in enumerate(options, start=1):
        print(f"{index}: {option}")

    raw_value = input(f"Choose a number (press Enter for {default_index + 1}): ").strip()
    if raw_value.isdigit():
        choice = int(raw_value) - 1
        if 0 <= choice < len(options):
            return options[choice]

    return options[default_index]


def _prompt_int(prompt: str, default_value: int) -> int:
    raw_value = input(f"{prompt} [{default_value}]: ").strip()
    if not raw_value:
        return default_value
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def _find_dataset_yaml(dataset_path: Path) -> Path:
    candidates = sorted(
        path for path in dataset_path.rglob("*.yml")
        if path.is_file()
    ) + sorted(
        path for path in dataset_path.rglob("*.yaml")
        if path.is_file()
    )

    if not candidates:
        raise FileNotFoundError(f"No dataset YAML file found inside {dataset_path}")

    if len(candidates) == 1:
        return candidates[0]

    print("Found multiple dataset YAML files:")
    selected = _prompt_choice("Select the YAML file to use:", [str(path.relative_to(dataset_path)) for path in candidates])
    return dataset_path / selected


def _find_dataset_config(dataset_path: Path) -> Path | None:
    config_candidates = [
        dataset_path / "config.yaml",
        dataset_path / "config.yml",
        dataset_path / "config.json",
    ]

    for candidate in config_candidates:
        if candidate.exists():
            return candidate

    return None


def _prompt_model() -> str:
    model_options = [
        "yolo11n.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo11x.pt",
        "yolo11n-obb.pt",
        "yolo11s-obb.pt",
        "yolo26n.pt",
        "yolo26n-obb.pt",
        "Enter a custom model name or local .pt path",
    ]

    print("Select a base model:")
    for index, option in enumerate(model_options, start=1):
        print(f"{index}: {option}")

    raw_value = input("Choose a number (press Enter for 2): ").strip()
    if raw_value.isdigit():
        choice = int(raw_value) - 1
        if 0 <= choice < len(model_options):
            if choice == len(model_options) - 1:
                custom_model = input("Enter the model name or path: ").strip()
                if custom_model:
                    return custom_model
                return "yolo11s.pt"
            return model_options[choice]

    return "yolo11s.pt"


def run_local_trainer() -> None:
    datasets_root = Path("datasets")
    if not datasets_root.exists():
        raise FileNotFoundError("The local datasets/ directory does not exist.")

    dataset_dirs = sorted(path for path in datasets_root.iterdir() if path.is_dir())
    if not dataset_dirs:
        raise FileNotFoundError("No dataset folders found in datasets/.")

    print("--- LOCAL YOLO TRAINER ---")
    dataset_folder_name = _prompt_choice(
        "Select a dataset folder:",
        [path.name for path in dataset_dirs],
    )
    selected_dataset = datasets_root / dataset_folder_name

    dataset_yaml_path = _find_dataset_yaml(selected_dataset)
    print(f"Using dataset YAML: {dataset_yaml_path}")

    config_path = _find_dataset_config(selected_dataset)
    if config_path:
        print(f"Using config file: {config_path}")
        trainer = YoloTrainer(
            dataset_dir=str(datasets_root),
            config_path=str(config_path),
        )
        trainer.config.setdefault("dataset_yaml_path", str(dataset_yaml_path))
        results, model_name, dataset_name = trainer.train()
    else:
        print("No config file found in the dataset folder. Falling back to prompts.")
        trainer = YoloTrainer(
            dataset_dir=str(datasets_root),
            config={
                "dataset": dataset_folder_name,
                "dataset_yaml_path": str(dataset_yaml_path),
                "model": _prompt_model(),
                "epochs": _prompt_int("Epochs", 100),
                "imgsz": _prompt_int("Image size", 640),
            },
        )
        results, model_name, dataset_name = trainer.train()
    print(f"Training complete for {dataset_name} with {model_name}")
    if hasattr(results, "save_dir"):
        print(f"Results saved in: {results.save_dir}")


if __name__ == "__main__":
    run_local_trainer()