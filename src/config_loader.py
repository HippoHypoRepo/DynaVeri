import yaml
from pathlib import Path

def load_config(config_path: str = "configs/workstation.yaml") -> dict:
    """Loads and returns the YAML configuration dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)