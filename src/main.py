import json
import os

from vision.winclip_adapter import WinCLIPAdapter


def main():
    image_path = "data/input/cable_bent_001.png"
    output_dir = "data/output"
    class_name = "cable"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    adapter = WinCLIPAdapter(threshold=0.12, mask_threshold=0.6)

    print("Running vision component...")
    result = adapter.predict(image_path)

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

    json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
