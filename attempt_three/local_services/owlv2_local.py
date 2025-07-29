"""
Local OWL-ViT-like detector implemented with Grounding-DINO.
"""

from __future__ import annotations
import os, pathlib
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
# Grounding-DINO helpers
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
from groundingdino.util.utils import get_phrases_from_posmap

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# --------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------- #

_TENSOR_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)


def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL->CHW float32 tensor in [0,1] and normalised."""
    return _TENSOR_TRANSFORM(img).unsqueeze(0).to(device)



# --------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------- #

class OWLViT:
    """
    Drop-in OWLv2 replacement using a local Grounding-DINO checkpoint.
    """

    def __init__(
        self,
        config_path: str | pathlib.Path = "/home/oguz/Desktop/attempt_three/GroundingDINO/config/GroundingDINO_SwinT_OGC.py",
        model_path : str | pathlib.Path = "/home/oguz/Desktop/attempt_three/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        device: str | torch.device | None = None,
        box_threshold : float = 0.25,
        text_threshold: float = 0.25,
    ):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = load_model(config_path, model_path).to(self.device).eval()
        self.box_th  = box_threshold
        self.text_th = text_threshold

    # ------------------------------------------------------------------ #
    

    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_detections(
        image: Image.Image | np.ndarray,
        detections: List[Dict],
        color: Tuple[int,int,int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Utility to quickly visualise results.
        Returns an RGB numpy array with rectangles + labels.
        """
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_cv = image.copy()

        h, w = img_cv.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if max(det["bbox"]) <= 1.01:  # assume relative
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            else:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                img_cv,
                f"{det['box_name']} {det['score']:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)


    @torch.inference_mode()
    def detect_objects(
        self,
        image: str | Image.Image | np.ndarray,
        text_prompts: str | List[str],
        *,
        bbox_conf_threshold: float = 0.10,
        bbox_score_top_k: int = 20,
        return_abs: bool = False,
    ) -> List[Dict]:
        # ---------- 1. image → tensor ------------------------------------
        if isinstance(image, str):
            img_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image[..., ::-1] if image.shape[-1] == 3 else image)
        else:
            img_pil = image.convert("RGB")

        W, H = img_pil.size
        img_tensor = _pil_to_tensor(img_pil, self.device)        # 1×3×H×W

        # ---------- 2. normalise prompt list -----------------------------
        if isinstance(text_prompts, str):
            prompt_list = [text_prompts.strip()]
        else:
            prompt_list = [p.strip() for p in text_prompts]

        print(f"[INFO] Running on prompts: {prompt_list}")

        # final caption fed to the model
        caption = ". ".join(prompt_list).strip() + "."

        # ---------- 3. forward pass --------------------------------------
        outputs  = self.model(img_tensor, captions=[caption])
        logits   = outputs["pred_logits"].sigmoid()[0]           # (N, C)
        boxes_c  = outputs["pred_boxes"][0]                      # (N, 4)

        scores, cls_idx = logits.max(dim=1)                      # (N,)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_c)
        boxes_xyxy *= torch.tensor([W, H, W, H], device=boxes_xyxy.device)

        # ---------- 4. build token-span → prompt map ---------------------
        # tokenise whole caption once (to know absolute token indices)
        full_tok = tokenizer(caption, add_special_tokens=False)
        # tokenise each prompt separately, to know how many tokens each one owns
        span_ends = []     # upper-exclusive end of each prompt in full caption
        cursor = 0
        for p in prompt_list:
            n_tok = len(tokenizer(p, add_special_tokens=False)["input_ids"])
            cursor += n_tok + 1              # +1 for the period that follows each prompt
            span_ends.append(cursor)         # e.g. [4, 9, 13, …]

        def prompt_id_from_token(t_idx: int) -> int:
            """return which prompt the token index falls into."""
            for i, end in enumerate(span_ends):
                if t_idx < end:              # still inside prompt i
                    return i
            return len(prompt_list) - 1      # clamp to last prompt just in case

        # ---------- 5. gather results ------------------------------------
        results: list[Dict] = []
        for b, s, idx in zip(boxes_xyxy, scores, cls_idx):
            if s < self.box_th or s < bbox_conf_threshold:
                continue

            token_idx = int(idx)
            # Guard against rare overflow (model sometimes predicts cls_idx == seq_len)
            if token_idx >= len(full_tok["input_ids"]):
                token_idx = len(full_tok["input_ids"]) - 1

            try:
                prompt_id = prompt_id_from_token(token_idx)
                name = prompt_list[prompt_id]
            except Exception as e:           # absolute fallback: heuristic
                logits_per_prompt = 2
                prompt_id = max(
                    0,
                    min((token_idx // logits_per_prompt) - 1, len(prompt_list) - 1),
                )
                name = prompt_list[prompt_id]
                print(f"[WARN] fallback mapping used for cls_idx={token_idx} ({e})")

            box_out = b.cpu().numpy().tolist()
            if not return_abs:
                box_out = [box_out[0] / W, box_out[1] / H,
                        box_out[2] / W, box_out[3] / H]

            results.append({
                "score": float(s),
                "bbox":  box_out,
                "box_name": name,
            })

        # ---------- 6. sort & return top-k -------------------------------
        results.sort(key=lambda d: d["score"], reverse=True)
        return results[:bbox_score_top_k]