import csv
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

from vision.winclip_adapter import WinCLIPAdapter
from src.config import save_config


def list_image_files(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def evaluate_class_folders(
    class_name: str,
    base_input_dir: str = "data/input",
    output_csv_path: Optional[str] = None,
    decimals: int = 6,
    score_field: str = "fused_score",
    use_fewshot: bool = True,
    threshold_k: float = 2.0,
    debug: bool = False,
) -> dict:
    """
    For one object class:
      1. Scores all images in the 'good' folder to calibrate the detection threshold
         (threshold = mean + threshold_k * std of good scores).
      2. Scores all images in each anomaly folder.
      3. Saves a CSV table (rows = image index, columns = folder name, values = score).

    Returns:
        {
            "csv_path":    path to the saved CSV,
            "threshold":   calibrated anomaly threshold for this class,
            "good_scores": list of scores on good images,
        }

    Expected directory structure:
        data/input/{class_name}/good/
        data/input/{class_name}/{anomaly_folder}/
    """
    class_dir = Path(base_input_dir) / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    good_dir = class_dir / "good"
    if not good_dir.exists():
        raise FileNotFoundError(f"'good' folder not found: {good_dir}")

    adapter = WinCLIPAdapter(
        repo_path="external/WinClip",
        dataset="mvtec",
        class_name=class_name,
        image_threshold=0.0,   # overridden after calibration
        debug=debug,
    )

    # Build the few-shot gallery from all good images once up front.
    good_files = list_image_files(good_dir)
    if use_fewshot:
        adapter.build_fewshot_gallery([str(p) for p in good_files])

    # ── 1. Calibrate threshold from good images ───────────────────────────────
    good_scores: list[float] = []

    for image_path in tqdm(good_files, desc="calibrating (good)", leave=False):
        result = adapter.predict(str(image_path))
        good_scores.append(float(result[score_field]))

    good_arr = np.array(good_scores)
    threshold = float(good_arr.mean() + threshold_k * good_arr.std())
    adapter.image_threshold = threshold

    if debug:
        print(f"Good scores: mean={good_arr.mean():.4f} std={good_arr.std():.4f}")
    print(f"Calibrated threshold (mean + {threshold_k}σ): {threshold:.6f}")
    save_config(class_name, threshold)

    # ── 2. Score anomaly folders ──────────────────────────────────────────────
    anomaly_folders = sorted([
        p for p in class_dir.iterdir()
        if p.is_dir() and p.name != "good"
    ])

    if not anomaly_folders:
        raise ValueError(f"No anomaly folders found in {class_dir}")

    # Determine max row count across all folders for the CSV.
    max_images = max(len(list_image_files(f)) for f in anomaly_folders)
    table_rows: list[dict[str, str]] = [{"index": str(i)} for i in range(1, max_images + 1)]
    folder_names: list[str] = []

    for folder in anomaly_folders:
        folder_name = folder.name
        folder_names.append(folder_name)
        image_files = list_image_files(folder)

        for row_idx, image_path in enumerate(
            tqdm(image_files, desc=folder_name, leave=False)
        ):
            result = adapter.predict(str(image_path))

            if score_field not in result:
                raise KeyError(
                    f"score_field='{score_field}' not in result. "
                    f"Available: {list(result.keys())}"
                )

            value = round(float(result[score_field]), decimals)
            table_rows[row_idx][folder_name] = f"{value:.{decimals}f}"

        for row_idx in range(len(image_files), max_images):
            table_rows[row_idx][folder_name] = ""

    # ── 3. Save CSV ───────────────────────────────────────────────────────────
    if output_csv_path is None:
        output_csv_path = str(Path("data/output") / f"{class_name}_{score_field}_table.csv")

    os.makedirs(Path(output_csv_path).parent, exist_ok=True)

    fieldnames = ["index"] + folder_names
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    return {
        "csv_path": output_csv_path,
        "threshold": threshold,
        "good_scores": good_scores,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate threshold and evaluate anomaly folders.")
    parser.add_argument("class_name", help="Object class to evaluate (e.g. grid, bottle)")
    parser.add_argument("--input-dir", default="data/input", help="Base input directory (default: data/input)")
    parser.add_argument("--score-field", default="fused_score", choices=["fused_score", "textual_score", "visual_score"])
    parser.add_argument("--threshold-k", type=float, default=2.0, help="Threshold = mean + k*std of good scores (default: 2.0)")
    parser.add_argument("--no-fewshot", action="store_true", help="Disable few-shot gallery")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    result = evaluate_class_folders(
        class_name=args.class_name,
        base_input_dir=args.input_dir,
        score_field=args.score_field,
        use_fewshot=not args.no_fewshot,
        threshold_k=args.threshold_k,
        debug=args.debug,
    )

    print(f"Saved CSV to: {result['csv_path']}")
    print(f"Threshold:    {result['threshold']:.6f}")
    print(f"Good scores:  {[round(s, 4) for s in result['good_scores']]}")


if __name__ == "__main__":
    main()
