from typing import List, Dict, Optional, Union
import os
import numpy as np
from PIL import Image
import cv2

from local_services.owlv2_local import OWLViT
from local_services.sam_local import SAM


class OwlVitSamPipeline:
    def __init__(
        self,
        owl_model: Optional[OWLViT] = None,
        sam_model: Optional[SAM] = None,
        owl_params: Optional[dict] = None,
    ):
        self.owl = owl_model or OWLViT()
        self.sam = sam_model or SAM()
        self.owl_params = owl_params or {
            "bbox_conf_threshold": 0.1,
            "bbox_score_top_k": 10
        }

    def run(
        self,
        image: Union[str, Image.Image],
        prompts: List[str],
        output_dir: Optional[str] = None,
        visualize: bool = True,
    ) -> List[Dict]:
        if isinstance(image, str):
            image_pil = Image.open(image).convert("RGB")
        else:
            image_pil = image.convert("RGB")

        image_np = np.array(image_pil)
        detections = self.owl.detect_objects(
            image_pil,
            prompts,
            bbox_conf_threshold=self.owl_params.get("bbox_conf_threshold", 0.1),
            bbox_score_top_k=self.owl_params.get("bbox_score_top_k", 10),
            return_abs=False
        )

        if not detections:
            print("[WARN] No objects detected.")
            return []

        bboxes = [det["bbox"] for det in detections]
        masks = self.sam.segment_by_bboxes(image_pil, bboxes)

        results = []
        for i, (det, mask_info) in enumerate(zip(detections, masks)):
            mask = mask_info["segmentation"]
            label = det.get("box_name", f"object_{i}")

            result = {
                "box_name": label,
                "bbox": det["bbox"],
                "score": det["score"],
                "mask": mask,
            }
            results.append(result)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                mask_path = os.path.join(output_dir, f"mask_{i}_{label}.png")
                overlay_path = os.path.join(output_dir, f"overlay_{i}_{label}.png")

                # Save mask
                Image.fromarray(mask).save(mask_path)
                print(f"[INFO] Saved mask: {mask_path}")

                if visualize:
                    overlay = self.overlay_mask(image_np, mask)
                    cv2.imwrite(overlay_path, overlay)
                    print(f"[INFO] Saved overlay: {overlay_path}")

        return results

    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 1] = mask  # green channel
        blended = cv2.addWeighted(image, 1.0, mask_colored, alpha, 0)
        return blended