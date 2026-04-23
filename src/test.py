import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

from vision.winclip_adapter import WinCLIPAdapter
from src.config import load_config


def list_image_files(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def resolve_images(patterns: list[str], class_name: str, input_dir: str) -> list[str]:
    """
    Resolve one or more image paths/glob patterns to a list of files.
    For each token:
      1. Try as-is (handles absolute paths and shell-expanded paths).
      2. Try relative to {input_dir}/{class_name}/.
    """
    base = Path(input_dir) / class_name
    results: list[str] = []

    for pattern in patterns:
        # Shell already expanded it, or it's an absolute/cwd-relative path
        matches = glob.glob(pattern)
        if matches:
            results.extend(matches)
            continue

        # Try relative to the class input directory
        matches = glob.glob(str(base / pattern))
        if matches:
            results.extend(matches)

    return sorted(set(results))


def output_folder_name(image_path: str, class_name: str, input_dir: str) -> str:
    """
    Derive a folder name from the image path.
    e.g. data/input/bottle/good/000.png  →  good-000
         broken_small/002.png            →  broken_small-002
    """
    p = Path(image_path)
    # Strip the base input dir + class prefix if present
    try:
        rel = p.relative_to(Path(input_dir) / class_name)
    except ValueError:
        rel = p
    # Drop extension, replace path separators with '-'
    parts = list(rel.with_suffix("").parts)
    return "-".join(parts)


def process_image(adapter, image_path: str, class_name: str,
                  input_dir: str, output_dir: str) -> dict:
    folder_name = output_folder_name(image_path, class_name, input_dir)
    image_output_dir = str(Path(output_dir) / folder_name)

    result = adapter.predict(image_path=image_path)

    heatmap_path, mask_path = adapter.save_outputs(
        image_path=image_path,
        heatmap=result["heatmap"],
        output_dir=image_output_dir,
        is_anomalous=result["is_anomalous"],
        label=result["defect_type_label"]
    )

    final_output = {
        "image": os.path.basename(image_path),
        "class_name": class_name,
        "textual_score": round(result["textual_score"], 6),
        "visual_score": round(result["visual_score"], 6),
        "fused_score": round(result["fused_score"], 6),
        "is_anomalous": result["is_anomalous"],
        "defect_type_iri": result["defect_type_iri"],
        "defect_type_label": result["defect_type_label"],
        "defect_type_scores": {
            iri.split("#")[-1]: score
            for iri, score in sorted(
                result["defect_type_scores"].items(), key=lambda x: x[1], reverse=True
            )
        },
        "heatmap_path": heatmap_path,
        "mask_path": mask_path,
    }

    os.makedirs(image_output_dir, exist_ok=True)
    json_path = os.path.join(image_output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    return final_output


def main():
    parser = argparse.ArgumentParser(description="Run WinCLIP inference on one or more images.")
    parser.add_argument(
        "class_name",
        help="Object class to inspect (e.g. grid, bottle).",
    )
    parser.add_argument(
        "image",
        nargs="+",
        help=(
            "One or more image paths or glob patterns. "
            "Relative paths are resolved against --input-dir/<class_name>/. "
            "Examples: broken_small/002.png  |  'good/*.png'  |  'broken_small/*.png' 'good/*.png'"
        ),
    )
    parser.add_argument(
        "--input-dir",
        default="data/input",
        help="Root directory containing per-class subfolders (default: data/input).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Root directory for results; each image gets its own subfolder, e.g. good-000/ (default: data/output).",
    )
    parser.add_argument(
        "--no-fewshot",
        action="store_true",
        help="Skip building the few-shot gallery from good/ images (zero-shot mode).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostic output (tensor shapes, scores, device info).",
    )
    args = parser.parse_args()

    class_name = args.class_name
    output_dir = args.output_dir

    image_paths = resolve_images(args.image, class_name, args.input_dir)  # args.image is now a list
    if not image_paths:
        parser.error(f"No images found for: {args.image}")

    good_images: list[str] = []
    if not args.no_fewshot:
        good_dir = Path(args.input_dir) / class_name / "good"
        if not good_dir.exists():
            parser.error(f"Good folder not found: {good_dir}")
        good_images = [str(p) for p in list_image_files(good_dir)]

    conf = load_config(class_name)

    adapter = WinCLIPAdapter(
        repo_path="external/WinClip",
        dataset="mvtec",
        class_name=class_name,
        image_threshold=conf['anomaly_thr'],
        mask_mod=conf['mask_mod'],
        debug=args.debug,
    )

    if good_images:
        adapter.build_fewshot_gallery(good_images)

    total = len(image_paths)
    start = time.time()

    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{total}] {image_path}")
        result = process_image(adapter, image_path, class_name, args.input_dir, output_dir)
        status = f"anomaly: {result['defect_type_label']}" if result["is_anomalous"] else "no anomaly"
        print(f"        → {status}")
        print(f"           textual={result['textual_score']:.4f}  visual={result['visual_score']:.4f}  fused={result['fused_score']:.4f}")

    print(f"\nDone. {total} image(s) in {time.time() - start:.1f}s")
    print(f"Results in: {output_dir}/")


if __name__ == "__main__":
    main()
