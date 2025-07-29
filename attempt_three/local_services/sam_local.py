# local_services/sam_local.py
from __future__ import annotations
import pathlib
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class Segmentor:
    """empty superclass for typing parity"""
    pass


class SAM(Segmentor):
    """
    Local wrapper around Meta-AI’s Segment-Anything-Model (SAM).

    Parameters
    ----------
    checkpoint : str | pathlib.Path
        Path to the *.pth* weights file (default: vit-h weights path you provided).
    model_type : {"vit_h","vit_l","vit_b"}
        Backbone name that matches the checkpoint.
    device : {"cuda","cpu",...} or None
        Where to load the model (auto-detects by default).
    generator_kwargs : dict | None
        Extra kwargs forwarded to SamAutomaticMaskGenerator for `segment_auto_mask`.
    """

    def __init__(
        self,
        checkpoint: str | pathlib.Path = "/home/oguz/Desktop/attempt_three/segment-anything/weights/sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        device: str | None = None,
        generator_kwargs: Dict[str, Any] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- load backbone & helpers --------------------------------------
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(self.device).eval()

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

        self.mask_generator = SamAutomaticMaskGenerator(sam, **default_gen_args)

    # ------------------------------------------------------------------- #
    # PUBLIC API (mirrors the remote-server version)
    # ------------------------------------------------------------------- #
    def segment_auto_mask(self, image: Image.Image) -> List[Dict]:
        """
        Generate *many* masks automatically (unsupervised).

        Returns
        -------
        list of dicts – each contains at least `"segmentation"` (H×W bool array).
        """
        img_np = np.asarray(image.convert("RGB"))
        masks = self.mask_generator.generate(img_np)

        # keep only what the old client needed
        return [{"segmentation": m["segmentation"]} for m in masks]

    # ------------------------------------------------------------------- #
    def segment_by_point_set(
        self,
        image: Image.Image,
        points: List[List[List[float]]],
        point_labels: List[List[int]],
    ) -> List[Dict]:
        """
        Predict one mask for each point-set + label set.

        Parameters
        ----------
        points : shape (N, P, 2) in **relative** coords [0-1].
        point_labels :   (N, P)  1 = positive, 0 = negative.

        Returns
        -------
        list of dicts with `"segmentation"` masks.
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

        return results

    # ------------------------------------------------------------------- #
    def segment_by_bboxes(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
    ) -> List[Dict]:
        """
        Predict one mask for each **relative** bbox [x1,y1,x2,y2] (0-1 normalized).
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

        return results