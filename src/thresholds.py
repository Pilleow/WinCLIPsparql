import json
import os
from pathlib import Path

_DEFAULT_PATH = Path("data/thresholds.json")


def save_threshold(class_name: str, threshold: float, path: Path | str = _DEFAULT_PATH) -> None:
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    data: dict[str, float] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data[class_name] = threshold
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_threshold(
    class_name: str,
    path: Path | str = _DEFAULT_PATH,
    default: float | None = None,
) -> float:
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if class_name in data:
            return float(data[class_name])
    if default is not None:
        return default
    raise KeyError(
        f"No calibrated threshold for class '{class_name}' in {path}. "
        f"Run batch_eval first."
    )
