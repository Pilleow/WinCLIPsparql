import json
import os
import time

from vision.winclip_adapter import WinCLIPAdapter


def main():
    class_name = "bottle"
    image_path = f"data/input/{class_name}/broken_small/002.png"
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
        # device="cpu",
    )

    print("Running WinCLIP inference...\n")
    start = time.time()
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

    print("Inspecting prompts...\n")
    prompt_info = adapter.inspect_prompts(image_path=image_path, top_k=5)

    final_output = {
        "image": os.path.basename(image_path),
        "class_name": class_name,
        "textual_score": round(result["textual_score"], 6),
        "visual_score": round(result["visual_score"], 6),
        "fused_score": round(result["fused_score"], 6),
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

    print(f"Decision:",
        final_output["top_normal_prompts"][0]["prompt"]
          if not final_output["is_anomalous"]
          else final_output["top_abnormal_prompts"][0]["prompt"]
    )
    print("Time taken: ", time.time() - start)


if __name__ == "__main__":
    main()
