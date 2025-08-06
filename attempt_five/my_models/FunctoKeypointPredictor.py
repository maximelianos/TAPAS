import torch
import torch.nn.functional as F
from loguru import logger
from tapas_gmm.encoder.keypoints import KeypointsPredictor
from tapas_gmm.utils.select_gpu import device

# --- import your Functo pipeline ---
from functo_point_transfer import func_point_transfer   # rename as needed

class FunctoKeypointsPredictor(KeypointsPredictor):
    """
    Thin wrapper that calls Functo’s func_point_transfer(...)
    and converts the single (x, y) keypoint into a TAPAS-compatible
    descriptor map of shape [B, 1, H, W].
    """
    sample_type = None          # TAPAS doesn’t need pre-training hooks here
    descriptor_dim = 1          # one “channel” is enough

    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.last_xy = None     # store latest keypoint for debugging

    # ------------------------------------------------------------------
    # TAPAS will call this in batches [B, 3, H, W]            (RGB, 0-1)
    # ------------------------------------------------------------------
    def compute_descriptor_batch(self, images: torch.Tensor, upscale=True):
        B, C, H, W = images.shape
        assert C == 3, "Functo expects RGB"

        desc_maps = torch.zeros(B, 1, H, W, device=images.device)

        for b in range(B):
            # 1. convert single frame to PIL (HWC, 0-255)
            pil = torch.permute(images[b] * 255.0, (1, 2, 0)).byte().cpu().numpy()
            pil = Image.fromarray(pil)

            # 2. call Functo pipeline (returns x, y in original resolution)
            x, y = self._run_functo_inference(pil)     # <- helper below
            self.last_xy = (x, y)

            # 3. make a 1-pixel gaussian “heat-spot”
            desc_maps[b, 0, y, x] = 1.0

        # optional: blur for smoother gradients
        desc_maps = F.gaussian_blur(desc_maps, (7, 7), sigma=(2.0, 2.0))
        return desc_maps                                             # [B,1,H,W]

    # ------------------------------------------------------------------
    def _run_functo_inference(self, pil_img):
        """
        Minimal stub: call your full func_point_transfer or a simplified
        endpoint that returns the keypoint for *this* image only.
        Replace this with your preferred API.
        """
        # ⚠️ For demo purposes we just pick the image center
        w, h = pil_img.size
        return w // 2, h // 2

    # TAPAS bookkeeping -- nothing to load
    def from_disk(self, *_, **__):
        logger.info("Functo wrapper has no checkpoint to load.")