import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_otsu

from knowledge.graph import load_defect_types


class WinCLIPAdapter:
    """
    Thin wrapper around the original caoyunkang/WinClip implementation.
    Defect type classification is driven by the RDF knowledge graph:
    each candidate defect type maps to an IRI from the ontology.
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
        mask_mod: float = 0.2,
        mask_percentile: float = 90.0,
        kg_path: str = "knowledge/ontology.ttl",
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
        self.mask_mod = mask_mod
        self.mask_percentile = mask_percentile
        self.kg_path = Path(kg_path)
        self.debug = debug

        print("CUDA Available:", torch.cuda.is_available())
        print("MPS Built:", torch.backends.mps.is_built())
        print("MPS Available:", torch.backends.mps.is_available())
        self.device = self._get_best_device(device, use_cpu)

        self._model = None
        self._transform = None
        self.WinClipAD = None
        # IRI → human-readable label, populated from the KG
        self._iri_to_label: dict[str, str] = {}

        self._import_winclip_code()
        self._build_model()

    def _get_best_device(self, device: Optional[str], use_cpu: bool) -> str:
        if device is not None:
            return device
        if use_cpu:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

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

        # Load defect type nodes from the RDF knowledge graph.
        # Keys in defect_types are IRI strings; prompts use '{}' as class placeholder.
        defect_types = load_defect_types(self.kg_path, class_name=self.class_name)
        self._iri_to_label = {info["iri"]: info["label"] for info in defect_types.values()}

        # Substitute the class name into prompt templates before encoding.
        defect_prompts = {
            iri: [p.format(self.class_name) for p in info["prompts"]]
            for iri, info in defect_types.items()
        }
        self._model.build_defect_type_feature_gallery(defect_prompts)

        self._log("WinCLIP model built.")
        self._log(f"class_name: {self.class_name}")
        self._log(f"dataset: {self.dataset}")
        self._log(f"device: {self.device}")
        self._log(f"defect types loaded from KG: {list(self._iri_to_label.values())}")

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

        Returns a dict with:
          - textual_score, visual_score, fused_score: image-level anomaly scores
          - is_anomalous: bool
          - defect_type_iri:   IRI of the best-matching ex:DefectType node in the KG
          - defect_type_label: human-readable label of that node
          - defect_type_scores: {iri: probability} for all defect type nodes
          - raw_heatmap, heatmap: pixel-level anomaly maps
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        if good_image_paths:
            self.build_fewshot_gallery(good_image_paths)

        tensor = self._load_and_transform_rgb(image_path)
        data = torch.stack([tensor], dim=0).to(self.device)

        with torch.no_grad():
            raw_score = self._model(data, return_details=True)

        raw_heatmap, norm_heatmap = self._extract_heatmap(raw_score["fused_map"])

        textual_score = float(raw_score["zero_shot_score"][0])
        visual_score = float(raw_score["visual_max_score"][0])
        fused_score = float(raw_score["fused_image_score"][0])
        is_anomalous = fused_score >= self.image_threshold

        defect_type_scores: dict[str, float] = raw_score.get("defect_type_scores", [{}])[0]
        if is_anomalous and defect_type_scores:
            best_iri = max(defect_type_scores, key=defect_type_scores.get)
            best_label = self._iri_to_label.get(best_iri, best_iri.split("#")[-1])
        else:
            best_iri = None
            best_label = None

        return {
            "textual_score": textual_score,
            "visual_score": visual_score,
            "fused_score": fused_score,
            "is_anomalous": is_anomalous,
            "defect_type_iri": best_iri,
            "defect_type_label": best_label,
            "defect_type_scores": defect_type_scores,
            "raw_heatmap": raw_heatmap,
            "heatmap": norm_heatmap,
        }

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

    def make_mask(self, heatmap: np.ndarray, k: float = 2) -> np.ndarray:
        """
        Create a binary mask using Otsu's thresholding, which automatically
        finds the optimal split between background and anomalous pixels.
        Works for both small anomalies (few bright pixels) and large anomalies
        (large bright region) without manual tuning.
        Falls back to mean + k*std if Otsu fails (e.g. flat heatmap).
        """
        try:
            thr = threshold_otsu(heatmap)
            thr = thr + self.mask_mod * (1 - thr)
            print("treshold: ", thr)
        except ValueError:
            thr = heatmap.mean() + k * heatmap.std()
        mask = (heatmap >= thr).astype(np.uint8) * 255
        return mask

    def save_outputs(
            self,
            image_path: str,
            heatmap: np.ndarray,
            output_dir: str,
            is_anomalous: bool = True,
            label: str = ""
    ) -> tuple[str, str]:
        """
        Save border overlay and binary mask for the given heatmap.
        If is_anomalous=False the mask is saved as all-black (no anomaly).
        """
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((heatmap.shape[1], heatmap.shape[0]))
        image_np = np.asarray(image).astype(np.uint8)

        if not is_anomalous:
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
        else:
            mask = self.make_mask(heatmap)
        mask_bool = mask > 0

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

        overlay_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_img)

        text = label if is_anomalous else "normal"
        color = (255, 0, 0) if is_anomalous else (0, 255, 0)

        font = ImageFont.load_default(30)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = 10
        x = overlay_img.width - text_width - padding
        y = overlay_img.height - text_height - padding

        draw.text((x, y), text, fill=color, font=font)

        heatmap_path = os.path.join(output_dir, "heatmap_overlay.png")
        overlay_img.save(heatmap_path)

        return heatmap_path, ""
