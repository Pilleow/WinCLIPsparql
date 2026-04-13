import csv
import os
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from vision.winclip_adapter import WinCLIPAdapter


def list_image_files(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def pick_good_images(good_folder: Path, k: Optional[int] = None) -> list[str]:
    good_files = list_image_files(good_folder)
    if k is not None:
        good_files = good_files[:k]
    return [str(p) for p in good_files]


def evaluate_class_folders(
    class_name: str,
    base_input_dir: str = "data/input",
    output_csv_path: Optional[str] = None,
    n_first: int = 5,
    decimals: int = 6,
    score_field: str = "fused_score",
    use_fewshot: bool = True,
    fewshot_k: Optional[int] = 5,
    debug: bool = False,
) -> str:
    """
    For one object class, evaluate the first N images in each anomaly folder
    and save results as a CSV table:
      - rows: 1..N
      - columns: anomaly folders
      - values: selected score_field rounded to `decimals`

    Expected structure:
      data/input/{class_name}/good/
      data/input/{class_name}/{anomaly_folder}/

    Example:
      data/input/bottle/good/
      data/input/bottle/broken_small/
      data/input/bottle/contamination/
    """
    class_dir = Path(base_input_dir) / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    good_dir = class_dir / "good"
    if not good_dir.exists():
        raise FileNotFoundError(f"'good' folder not found: {good_dir}")

    good_images = pick_good_images(good_dir, k=fewshot_k) if use_fewshot else []

    adapter = WinCLIPAdapter(
        repo_path="external/WinClip",
        dataset="mvtec",
        class_name=class_name,
        image_threshold=0.01,
        mask_percentile=90.0,
        debug=debug,
    )

    folders = sorted(
        [
            p for p in class_dir.iterdir()
            if p.is_dir()
        ]
    )

    if not folders:
        raise ValueError(f"No anomaly folders found in {class_dir}")

    # Table: row index -> {folder_name: value}
    table_rows: list[dict[str, str]] = [
        {"index": str(i)} for i in range(1, n_first + 1)
    ]

    folder_names: list[str] = []

    for folder in folders:
        folder_name = folder.name
        folder_names.append(folder_name)

        image_files = list_image_files(folder)[:n_first]

        for row_idx, image_path in enumerate(
                tqdm(image_files, desc=f"{folder_name}", leave=False)
        ):
            result = adapter.predict(
                image_path=str(image_path),
                good_image_paths=good_images,
            )

            if score_field not in result:
                raise KeyError(
                    f"Requested score_field='{score_field}' not found in result. "
                    f"Available keys: {list(result.keys())}"
                )

            value = round(float(result[score_field]), decimals)
            table_rows[row_idx][folder_name] = f"{value:.{decimals}f}"

        for row_idx in range(len(image_files), n_first):
            table_rows[row_idx][folder_name] = ""

    if output_csv_path is None:
        output_csv_path = str(Path("data/output") / f"{class_name}_{score_field}_table.csv")

    os.makedirs(Path(output_csv_path).parent, exist_ok=True)

    fieldnames = ["index"] + folder_names
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    return output_csv_path


def main():
    class_name = "grid"
    n_first = 5
    decimals = 3
    score_field = "fused_score"   # can also be "textual_score" or "visual_score"

    csv_path = evaluate_class_folders(
        class_name=class_name,
        base_input_dir="data/input",
        output_csv_path=f"data/output/{class_name}_{score_field}_table.csv",
        n_first=n_first,
        decimals=decimals,
        score_field=score_field,
        use_fewshot=True,
        fewshot_k=5,
        debug=False,
    )

    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
