# local_services/sam_local.py
from __future__ import annotations
# ------------------------------------------------------------------ #
# ③  Reduce CUDA fragmentation: split big allocations into ≤64 MB
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
# ------------------------------------------------------------------ #
import pathlib
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator,
)


# ------------------------------------------------------------------ #
# helper functions (private)
# ------------------------------------------------------------------ #
def _flush_gpu() -> None:
    """④  Free any cached / unreferenced GPU memory early."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _release_predictor_features(predictor: SamPredictor) -> None:
    """⑦  Drop image‑encoder features to free VRAM."""
    try:
        predictor.features = None
    except AttributeError:
        pass
    _flush_gpu()


# ------------------------------------------------------------------ #
class Segmentor:  # empty superclass for typing parity
    pass


class SAM(Segmentor):
    """
    Local wrapper around Meta‑AI’s Segment‑Anything‑Model (SAM).
    (Public docstring unchanged.)
    """

    # -------------------------------------------------------------- #
    # constructor
    # -------------------------------------------------------------- #
    def __init__(
        self,
        checkpoint: str | pathlib.Path = (
            "/home/oguz/Desktop/attempt_three/segment-anything/weights/"
            "sam_vit_b_01ec64.pth"
        ),
        model_type: str = "vit_b",
        device: str | None = None,
        generator_kwargs: Dict[str, Any] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- load backbone & helpers ----------------------------------
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(self.device).eval()  # inference‑only

        self.predictor = SamPredictor(sam)

        default_gen_args = dict(
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=0,
        )
        if generator_kwargs:
            default_gen_args.update(generator_kwargs)

        # AutomaticMaskGenerator can be big → build lazily on first use
        self._gen_args = default_gen_args
        self._mask_generator: SamAutomaticMaskGenerator | None = None

    # -------------------------------------------------------------- #
    # Lazy getter so we don’t hold it if never used
    # -------------------------------------------------------------- #
    @property
    def mask_generator(self) -> SamAutomaticMaskGenerator:
        if self._mask_generator is None:
            self._mask_generator = SamAutomaticMaskGenerator(
                self.predictor.model, **self._gen_args
            )
        return self._mask_generator

    # ---------------------------------------------------------------- #
    # PUBLIC API
    # ---------------------------------------------------------------- #
    @torch.no_grad()  # ①  no gradients
    def segment_auto_mask(self, image: Image.Image) -> List[Dict]:
        """
        Generate many masks automatically (unsupervised).
        Returns only the ``"segmentation"`` field for each mask.
        """
        img_np = np.asarray(image.convert("RGB"))
        masks = self.mask_generator.generate(img_np)

        out = [{"segmentation": m["segmentation"]} for m in masks]

        _flush_gpu()  # ④ free VRAM held by generator intermediates
        return out

    # ---------------------------------------------------------------- #
    @torch.no_grad()  # ①
    def segment_by_point_set(
        self,
        image: Image.Image,
        points: List[List[List[float]]],
        point_labels: List[List[int]],
    ) -> List[Dict]:
        """
        Predict one mask for each point‑set + label set.
        """
        img_np = np.asarray(image.convert("RGB"))
        self.predictor.set_image(img_np)
        W, H = image.size

        results = []
        for pt_set, lbl_set in zip(points, point_labels):
            pt_px = np.array([[x * W, y * H] for x, y in pt_set], dtype=np.float32)
            lbl_np = np.array(lbl_set, dtype=np.int32)

            masks, scores, _ = self.predictor.predict(
                point_coords=pt_px,
                point_labels=lbl_np,
                multimask_output=False,
            )
            best = int(np.argmax(scores))
            results.append({"segmentation": masks[best]})

        _release_predictor_features(self.predictor)  # ⑦ + ④
        return results

    # ---------------------------------------------------------------- #
    @torch.no_grad()  # ①
    def segment_by_bboxes(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
    ) -> List[Dict]:
        """
        Predict one mask for each relative bbox [x1,y1,x2,y2] (0‑1 normalized).
        """
        img_np = np.asarray(image.convert("RGB"))
        self.predictor.set_image(img_np)
        W, H = image.size

        results = []
        for bb in bboxes:
            x1, y1, x2, y2 = np.array(bb) * np.array([W, H, W, H])
            box_px = np.array([x1, y1, x2, y2], dtype=np.float32)

            masks, scores, _ = self.predictor.predict(
                box=box_px,
                multimask_output=False,
            )
            best = int(np.argmax(scores))
            results.append({"segmentation": masks[best]})

        _release_predictor_features(self.predictor)  # ⑦ + ④
        return results