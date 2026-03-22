import os
from pathlib import Path

import numpy as np
from PIL import Image


class WinCLIPAdapter:
    """
    Minimalny adapter vision na etap MVP.
    Na razie nie używa prawdziwego WinCLIP - tylko zwraca wynik w docelowym formacie.
    """

    def __init__(self, threshold: float = 0.5, mask_threshold: float = 0.6):
        self.threshold = threshold
        self.mask_threshold = mask_threshold

    def predict(self, image_path: str) -> dict:
        """
        Zwraca:
        - anomaly_score
        - is_anomalous
        - heatmap (numpy array 2D)
        """
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        image_np = np.asarray(image).astype(np.float32) / 255.0

        # Prosty placeholder:
        # odchylenie od średniej jasności traktujemy jako "anomalię"
        gray = image_np.mean(axis=2)
        mean_val = gray.mean()
        heatmap = np.abs(gray - mean_val)

        # normalizacja 0..1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else:
            heatmap = np.zeros((height, width), dtype=np.float32)

        anomaly_score = float(heatmap.mean())
        is_anomalous = anomaly_score >= self.threshold

        return {
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
            "heatmap": heatmap,
        }

    def save_outputs(
        self,
        image_path: str,
        heatmap: np.ndarray,
        output_dir: str
    ) -> tuple[str, str]:
        """
        Zapisuje:
        - heatmapę nałożoną na obraz
        - binarną maskę
        """
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image).astype(np.uint8)

        if heatmap.shape[:2] != image_np.shape[:2]:
            raise ValueError("Heatmap size must match image size.")

        stem = Path(image_path).stem
        heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)

        # Prosty czerwony overlay
        overlay = image_np.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], heatmap_uint8)

        heatmap_path = os.path.join(output_dir, f"{stem}_heatmap_overlay.png")
        Image.fromarray(overlay).save(heatmap_path)

        mask = (heatmap >= self.mask_threshold).astype(np.uint8) * 255
        mask_path = os.path.join(output_dir, f"{stem}_mask.png")
        Image.fromarray(mask).save(mask_path)

        return heatmap_path, mask_path
