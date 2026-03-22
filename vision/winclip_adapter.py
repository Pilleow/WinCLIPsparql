import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image


class WinCLIPAdapter:
    def __init__(
        self,
        repo_path: str = "external/WinClip",
        dataset: str = "mvtec",
        class_name: str = "bottle",
        scales: Sequence[int] = (2, 3),
        image_size: int = 240,
        resolution: int = 400,
        backbone: str = "ViT-B-16-plus-240",
        pretrained_dataset: str = "laion400m_e32",
        image_threshold: float = 0.01,
        mask_percentile: float = 90.0,
        device: Optional[str] = None,
        use_cpu: bool = False,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.dataset = dataset
        self.class_name = class_name
        self.scales = tuple(scales)
        self.image_size = image_size
        self.resolution = resolution
        self.backbone = backbone
        self.pretrained_dataset = pretrained_dataset
        self.image_threshold = image_threshold
        self.mask_percentile = mask_percentile

        if device is None:
            if torch.cuda.is_available() and not use_cpu:
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._transform = None

        self._import_winclip_code()
        self._build_model()

    def _import_winclip_code(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"WinClip repo not found at: {self.repo_path}"
            )

        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from WinCLIP import WinClipAD  # type: ignore
        self.WinClipAD = WinClipAD

    def _build_model(self) -> None:
        kwargs = {
            "dataset": self.dataset,
            "class_name": self.class_name,
            "img_resize": self.image_size,
            "img_cropsize": self.image_size,
            "resolution": self.resolution,
            "batch_size": 1,
            "vis": False,
            "root_dir": "./result_winclip_adapter",
            "load_memory": True,
            "cal_pro": False,
            "experiment_indx": 0,
            "gpu_id": 0,
            "pure_test": False,
            "k_shot": 0,
            "scales": self.scales,
            "backbone": self.backbone,
            "pretrained_dataset": self.pretrained_dataset,
            "use_cpu": 1 if self.device == "cpu" else 0,
            "device": self.device,
            "out_size_h": self.resolution,
            "out_size_w": self.resolution,
        }

        self._model = self.WinClipAD(**kwargs).to(self.device)
        self._model.eval_mode()
        self._transform = self._model.transform
        self._model.build_text_feature_gallery(self.class_name)

    def _load_and_transform_rgb(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self._transform(image)

    def build_fewshot_gallery(self, good_image_paths: Sequence[str]) -> None:
        if not good_image_paths:
            return

        batch = []
        for path in good_image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Few-shot image not found: {path}")
            batch.append(self._load_and_transform_rgb(path))

        data = torch.stack(batch, dim=0).to(self.device)

        with torch.no_grad():
            self._model.build_image_feature_gallery(data)

    def predict(
        self,
        image_path: str,
        good_image_paths: Optional[Sequence[str]] = None,
    ) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        if good_image_paths:
            self.build_fewshot_gallery(good_image_paths)

        tensor = self._load_and_transform_rgb(image_path)
        data = torch.stack([tensor], dim=0).to(self.device)

        with torch.no_grad():
            raw_score = self._model(data)

        print("raw_score type:", type(raw_score))
        if isinstance(raw_score, list):
            print("raw_score list len:", len(raw_score))
            if len(raw_score) > 0:
                first = raw_score[0]
                if torch.is_tensor(first):
                    print("first tensor shape:", tuple(first.shape))
                else:
                    print("first item type:", type(first))
        elif torch.is_tensor(raw_score):
            print("raw_score tensor shape:", tuple(raw_score.shape))

        raw_heatmap, norm_heatmap = self._extract_heatmap(raw_score)

        anomaly_score = float(raw_heatmap.max())
        is_anomalous = anomaly_score >= self.image_threshold

        return {
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
            "raw_heatmap": raw_heatmap,
            "heatmap": norm_heatmap,
        }

    def _extract_heatmap(self, raw_score) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(raw_score, list):
            if len(raw_score) == 0:
                raise ValueError("Model returned an empty list.")
            raw_score = raw_score[0]

        if torch.is_tensor(raw_score):
            arr = raw_score.detach().float().cpu().numpy()
        else:
            arr = np.asarray(raw_score, dtype=np.float32)

        print("arr shape before squeeze:", np.asarray(arr).shape)
        print("arr min/mean/max:", float(arr.min()), float(arr.mean()), float(arr.max()))

        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(
                f"Unexpected WinCLIP output shape after squeeze: {arr.shape}"
            )

        raw_arr = arr.astype(np.float32)

        if raw_arr.max() > raw_arr.min():
            norm_arr = (raw_arr - raw_arr.min()) / (raw_arr.max() - raw_arr.min())
        else:
            norm_arr = np.zeros_like(raw_arr, dtype=np.float32)

        return raw_arr, norm_arr.astype(np.float32)

    def save_outputs(
        self,
        image_path: str,
        heatmap: np.ndarray,
        output_dir: str,
    ) -> tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((heatmap.shape[1], heatmap.shape[0]))
        image_np = np.asarray(image).astype(np.uint8)

        heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)

        overlay = image_np.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], heatmap_uint8)

        stem = Path(image_path).stem

        heatmap_path = os.path.join(output_dir, f"{stem}_heatmap_overlay.png")
        Image.fromarray(overlay).save(heatmap_path)

        thr = np.percentile(heatmap, self.mask_percentile)
        mask = (heatmap >= thr).astype(np.uint8) * 255

        mask_path = os.path.join(output_dir, f"{stem}_mask.png")
        Image.fromarray(mask).save(mask_path)

        return heatmap_path, mask_path
