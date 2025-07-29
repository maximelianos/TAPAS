# keypoint_detector.py
import os, pathlib, time
from typing import Union, Dict

import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensordict import TensorDict
from types import SimpleNamespace

from keypoint_encoder import KeypointEncoder                     # your TAPAS wrapper
from TAPAS.tapas_gmm.utils.observation import SceneObservation, SingleCamObservation


class KeypointDetector:
    """
    Light wrapper around TAPAS to extract 2-D key-points on a single image / crop
    and store the visualisation + csvs to disk.
    """
    # --------------------------------------------------------------------- #
    def __init__(self,
                 device: str | torch.device | None = None,
                 img_h: int = 480,
                 img_w: int = 640) -> None:

        self.device   = torch.device(device) if device else (
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.img_h    = img_h
        self.img_w    = img_w

        self.model    = KeypointEncoder().to(self.device).eval()

        # ↳ exactly what your test script used
        self._transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.img_h, self.img_w)),
        ])

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _forward_tapas(self, pil_img: Image.Image):
        """Internal helper – returns (kp_raw_2d, sm_tensor, info)."""
        rgb   = self._transform(pil_img).unsqueeze(0).to(self.device)        # (1,3,H,W)
        depth = torch.ones(1, self.img_h, self.img_w, device=self.device)    # dummy depth
        intr  = torch.tensor([[500., 0,   self.img_w/2],
                              [0,   500., self.img_h/2],
                              [0,     0,              1]],
                             device=self.device)

        cam_obs = SingleCamObservation(rgb=rgb, depth=depth,
                                       intr=intr, extr=torch.eye(4, device=self.device))
        camdict = TensorDict({"wrist": cam_obs,
                              "_order": SimpleNamespace(order=["wrist"])},
                             batch_size=[])
        scene   = SceneObservation(cameras=camdict)

        kp_raw, info = self.model.encode(scene)          # kp_raw shape: (1, 2·N)
        sm_tensor    = info["sm"][0].squeeze(0)          # (N, h, w)

        return kp_raw, sm_tensor, info

    # --------------------------------------------------------------------- #
    def run_on_image(self,
                     img: Union[str, Image.Image],
                     save_dir: str,
                     prefix: str = "tapas") -> Dict[str, float]:
        """
        Parameters
        ----------
        img       : str | PIL.Image
            Path or PIL image (crop) to analyse.
        save_dir  : str
            Directory to dump visualisation + csv files.  Created if missing.
        prefix    : str
            Prefix for the output filenames (default: 'tapas').

        Returns
        -------
        dict  with keys
          { 'pixel'      : (x_px, y_px),
            'normalised' : (x_norm, y_norm) }
        """
        save_path = pathlib.Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        pil = Image.open(img).convert("RGB") if isinstance(img, str) else img
        crop_w, crop_h = pil.size

        kp_raw, sm, info = self._forward_tapas(pil)

        # ---- most-salient kp from sm ----------------------------------- #
        N, H, W  = sm.shape
        flat_idx = torch.argmax(sm.view(N, -1))
        kp_idx   = flat_idx // (H*W)
        pix_idx  = flat_idx %  (H*W)
        y_loc    = (pix_idx // W).item()
        x_loc    = (pix_idx %  W).item()

        # map to full crop resolution
        x_px = x_loc * (crop_w / W)
        y_px = y_loc * (crop_h / H)

        # normalised to [0,1] wrt *crop*
        x_norm = x_px / crop_w
        y_norm = y_px / crop_h

        # ---- all key-points for viz ----------------------------------- #
        vec       = info["kp_raw_2d"][0] if isinstance(info["kp_raw_2d"], tuple) else info["kp_raw_2d"]
        vec       = vec.squeeze(0).cpu().numpy()                 # (2·N,)
        u_norm    = vec[:N]
        v_norm    = vec[N:]
        kp_norm   = np.stack([u_norm, v_norm], axis=1)           # (N,2)

        kp_px = (kp_norm + 1) / 2
        kp_px[:, 0] *= crop_w
        kp_px[:, 1] *= crop_h

        # ---- save visualisation -------------------------------------- #
        plt.figure(figsize=(8, 6))
        plt.imshow(pil)
        plt.scatter(kp_px[:, 0], kp_px[:, 1], c='red', s=40, label='Keypoints')
        plt.scatter([x_px], [y_px], c='blue', s=80, label='Top-1')
        plt.legend(); plt.axis('off'); plt.tight_layout()
        viz_name = f"{prefix}_viz.png"
        plt.savefig(save_path / viz_name, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

        # ---- save csvs ------------------------------------------------ #
        pd.DataFrame(kp_px, columns=["x", "y"]).to_csv(save_path / f"{prefix}_all.csv",
                                                       index=False)
        pd.DataFrame([[x_px, y_px]], columns=["x", "y"]).to_csv(save_path / f"{prefix}_top1.csv",
                                                                index=False)

        return {
            "pixel"      : (float(x_px),  float(y_px)),
            "normalised" : (float(x_norm), float(y_norm)),
            "kp_index"   : int(kp_idx)
        }