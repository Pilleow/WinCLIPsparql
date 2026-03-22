import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image


class WinCLIPAdapter:
    """
    Thin wrapper around the original caoyunkang/WinClip implementation.
    """

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
        debug: bool = True,
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
        self.debug = debug

        print("CUDA Available: ", torch.cuda.is_available())
        if device is None:
            if torch.cuda.is_available() and not use_cpu:
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._transform = None
        self.WinClipAD = None

        self._import_winclip_code()
        self._build_model()

    def _log(self, *args) -> None:
        if self.debug:
            print(*args)

    def _import_winclip_code(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"WinClip repo not found at: {self.repo_path}. "
                f"Clone https://github.com/caoyunkang/WinClip.git into external/WinClip"
            )

        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from WinCLIP import WinClipAD  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import WinClipAD from external/WinClip. "
                "Check that the original repo is cloned correctly and its dependencies are installed."
            ) from e

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

        self._log("WinCLIP model built.")
        self._log(f"class_name: {self.class_name}")
        self._log(f"dataset: {self.dataset}")
        self._log(f"device: {self.device}")

    def _load_and_transform_rgb(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self._transform(image)

    def build_fewshot_gallery(self, good_image_paths: Sequence[str]) -> None:
        """
        Build the few-shot visual gallery from provided 'good' images.
        Uses all provided images.
        """
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

        self._log(f"Built few-shot gallery from {len(good_image_paths)} good image(s).")

    def predict(
        self,
        image_path: str,
        good_image_paths: Optional[Sequence[str]] = None,
    ) -> dict:
        """
        Run inference on a single image.

        Returns:
        - anomaly_score: image-level scalar derived from raw map max
        - is_anomalous: thresholded image-level decision
        - raw_heatmap: raw anomaly map from WinCLIP
        - heatmap: normalized heatmap for visualization
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        if good_image_paths:
            self.build_fewshot_gallery(good_image_paths)

        tensor = self._load_and_transform_rgb(image_path)
        data = torch.stack([tensor], dim=0).to(self.device)

        with torch.no_grad():
            raw_score = self._model(data)

        self._debug_raw_score(raw_score)

        raw_heatmap, norm_heatmap = self._extract_heatmap(raw_score)

        anomaly_score = float(raw_heatmap.max())
        is_anomalous = anomaly_score >= self.image_threshold

        return {
            "anomaly_score": anomaly_score,
            "is_anomalous": is_anomalous,
            "raw_heatmap": raw_heatmap,
            "heatmap": norm_heatmap,
        }

    def _debug_raw_score(self, raw_score) -> None:
        self._log("raw_score type:", type(raw_score))

        if isinstance(raw_score, list):
            self._log("raw_score list len:", len(raw_score))
            if len(raw_score) > 0:
                first = raw_score[0]
                if torch.is_tensor(first):
                    self._log("first tensor shape:", tuple(first.shape))
                else:
                    self._log("first item type:", type(first))
        elif torch.is_tensor(raw_score):
            self._log("raw_score tensor shape:", tuple(raw_score.shape))

    def _extract_heatmap(self, raw_score) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract raw and normalized 2D anomaly maps from model output.
        """
        if isinstance(raw_score, list):
            if len(raw_score) == 0:
                raise ValueError("Model returned an empty list.")
            raw_score = raw_score[0]

        if torch.is_tensor(raw_score):
            arr = raw_score.detach().float().cpu().numpy()
        else:
            arr = np.asarray(raw_score, dtype=np.float32)

        self._log("arr shape before squeeze:", np.asarray(arr).shape)
        self._log(
            "arr min/mean/max:",
            float(arr.min()),
            float(arr.mean()),
            float(arr.max()),
        )

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

    def make_mask(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Create a binary mask from the normalized heatmap using percentile thresholding.
        """
        thr = np.percentile(heatmap, self.mask_percentile)
        mask = (heatmap >= thr).astype(np.uint8) * 255
        return mask

    def save_outputs(
            self,
            image_path: str,
            heatmap: np.ndarray,
            output_dir: str,
    ) -> tuple[str, str]:
        """
        Save:
        - border overlay based on the binary mask
        - binary mask
        """
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((heatmap.shape[1], heatmap.shape[0]))
        image_np = np.asarray(image).astype(np.uint8)

        stem = Path(image_path).stem

        mask = self.make_mask(heatmap)
        mask_bool = mask > 0

        border = np.zeros_like(mask_bool, dtype=bool)

        h, w = mask_bool.shape
        eroded = np.zeros_like(mask_bool, dtype=bool)

        if h >= 3 and w >= 3:
            eroded[1:-1, 1:-1] = (
                    mask_bool[0:-2, 0:-2] &
                    mask_bool[0:-2, 1:-1] &
                    mask_bool[0:-2, 2:] &
                    mask_bool[1:-1, 0:-2] &
                    mask_bool[1:-1, 1:-1] &
                    mask_bool[1:-1, 2:] &
                    mask_bool[2:, 0:-2] &
                    mask_bool[2:, 1:-1] &
                    mask_bool[2:, 2:]
            )

        border = mask_bool & (~eroded)

        thick_border = border.copy()
        thick_border[:-1, :] |= border[1:, :]
        thick_border[1:, :] |= border[:-1, :]
        thick_border[:, :-1] |= border[:, 1:]
        thick_border[:, 1:] |= border[:, :-1]

        overlay = image_np.copy()

        overlay[thick_border] = [255, 0, 0]

        heatmap_path = os.path.join(output_dir, f"heatmap_overlay.png")
        Image.fromarray(overlay).save(heatmap_path)

        mask_path = os.path.join(output_dir, f"mask.png")
        Image.fromarray(mask).save(mask_path)

        return heatmap_path, mask_path

    def inspect_prompts(
            self,
            image_path: str,
            top_k: int = 5,
    ) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        tensor = self._load_and_transform_rgb(image_path)
        data = torch.stack([tensor], dim=0).to(self.device)

        with torch.no_grad():
            ranked = self._model.rank_prompts_for_image(data, top_k=top_k)

        return ranked[0]