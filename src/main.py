import json
import os

from vision.winclip_adapter import WinCLIPAdapter


def main():
    class_name = "transistor"
    image_path = f"data/input/{class_name}/misplaced/002.png"
    output_dir = "data/output"

    good_images = [
        f"data/input/{class_name}/good/000.png",
        f"data/input/{class_name}/good/001.png",
        f"data/input/{class_name}/good/002.png",
        f"data/input/{class_name}/good/003.png",
        f"data/input/{class_name}/good/004.png",
    ]

    adapter = WinCLIPAdapter(
        repo_path="external/WinClip",
        dataset="mvtec",
        class_name=class_name,
        image_threshold=0.01,
        mask_percentile=90.0,
        debug=True,
    )

    print("Running WinCLIP inference...\n")
    result = adapter.predict(
        image_path=image_path,
        good_image_paths=good_images,
    )

    if result['is_anomalous']:
        heatmap_path, mask_path = adapter.save_outputs(
            image_path=image_path,
            heatmap=result["heatmap"],
            output_dir=output_dir,
        )
    else:
        heatmap_path = None
        mask_path = None

    print("\nInspecting prompts...")
    prompt_info = adapter.inspect_prompts(image_path=image_path, top_k=5)

    final_output = {
        "image": os.path.basename(image_path),
        "class_name": class_name,
        "anomaly_score": round(result["anomaly_score"], 6),
        "is_anomalous": result["is_anomalous"],
        "heatmap_path": heatmap_path,
        "mask_path": mask_path,
        "top_normal_prompts": [
            {"prompt": prompt, "score": float(score)}
            for prompt, score in prompt_info["top_normal_prompts"]
        ],
        "top_abnormal_prompts": [
            {"prompt": prompt, "score": float(score)}
            for prompt, score in prompt_info["top_abnormal_prompts"]
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(
        output_dir,
        f"result.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
