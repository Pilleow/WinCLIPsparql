import json
import os
from pathlib import Path

_DEFAULT_PATH = Path("data/config.json")


def save_config(class_name: str, threshold: float, path: Path | str = _DEFAULT_PATH) -> None:
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    data: dict[str, dict[str, float]] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data[class_name] = {
        "anomaly_thr": threshold,
        "mask_mod": 0.2
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_config(
    class_name: str,
    path: Path | str = _DEFAULT_PATH
) -> float:
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if class_name in data:
            return data[class_name]
        else:
            raise KeyError(f"Unknown config - run train.py first: {class_name}")
    raise KeyError(
        f"No calibrated threshold for class '{class_name}' in {path}. "
        f"Run batch_eval first."
    )
