import json
import os

from vision.winclip_adapter import WinCLIPAdapter


def main():
    image_path = "data/input/bottle/broken_large/003.png"
    output_dir = "data/output"
    class_name = "bottle"

    good_images = [
        "data/input/bottle/good/000.png",
        "data/input/bottle/good/001.png",
        "data/input/bottle/good/002.png",
        "data/input/bottle/good/003.png",
        "data/input/bottle/good/004.png",
    ]

    adapter = WinCLIPAdapter(
        repo_path="external/WinClip",
        dataset="mvtec",
        class_name=class_name,
        image_threshold=0.01,
        mask_percentile=90.0,
        debug=True,
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

    print("\nInspecting prompts...")
    prompt_info = adapter.inspect_prompts(image_path=image_path, top_k=5)

    print("\nTop normal prompts:")
    for prompt, score in prompt_info["top_normal_prompts"]:
        print(f"{score:.6f} - {prompt}")

    print("\nTop abnormal prompts:")
    for prompt, score in prompt_info["top_abnormal_prompts"]:
        print(f"{score:.6f} - {prompt}")

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
        f"{os.path.splitext(os.path.basename(image_path))[0]}_result.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()