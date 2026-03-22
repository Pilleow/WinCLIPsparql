import json
import os

from vision.winclip_adapter import WinCLIPAdapter


def main():
    image_path = "data/input/bottle/broken_large/003.png"
    output_dir = "data/output"
    class_name = "a top-down picture of a bottle with a small piece of neck broken off"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # zero-shot
    # good_images = []

    # few-shot
    good_images = [
        # "data/input/bottle/good/000.png",
        # "data/input/bottle/good/001.png",
        # "data/input/bottle/good/002.png",
        # "data/input/bottle/good/003.png",
        # "data/input/bottle/good/004.png",
    ]

    adapter = WinCLIPAdapter(
        repo_path="external/WinClip",
        dataset="mvtec",
        class_name=class_name,
        image_size=240,
        resolution=400,
        use_cpu=False,
    )

    print("Running WinCLIP inference...")
    result = adapter.predict(
        image_path=image_path,
        good_image_paths=good_images,
    )

    heatmap_path, mask_path = adapter.save_outputs(
        image_path=image_path,
        heatmap=result["heatmap"],
        output_dir=output_dir,
    )

    final_output = {
        "image": os.path.basename(image_path),
        "class_name": class_name,
        "anomaly_score": round(result["anomaly_score"], 4),
        "is_anomalous": result["is_anomalous"],
        "heatmap_path": heatmap_path,
        "mask_path": mask_path,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(image_path))[0]}_result.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()