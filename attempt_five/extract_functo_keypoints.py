from __future__ import annotations
import os
import yaml

from cam_to_target_trans import cam_to_target_trans, demo_test_detection
from func_point_transfer import func_point_transfer

from local_services.owlv2_local import OWLViT
from local_services.sam_local import SAM
from PIL import Image
import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
"""
def discard_large_boxes(
    boxes: List[List[float]], max_area_ratio: float = 0.6
) -> List[List[float]]:
    
    return [
        b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) < max_area_ratio
    ]
"""
def discard_large_boxes(
    boxes: List[List[float]], max_area_ratio: float = 0.6
) -> List[List[float]]:
    """
    Return only the smallest box (by area) among those with area < max_area_ratio.
    If none pass the threshold, return an empty list.
    """
    # Sort by area and return the smallest one
    smallest_box = min(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return [smallest_box]

def get_kettle_part_masks(
    image_pil: Image.Image,
    owl: OWLViT,
    sam: SAM,
    pad: int = 20,
    area_thresh: float = 0.7,
    return_debug_png: Optional[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
        handle_masks, spout_masks –  each is a list of SAM mask dicts
    Optional:
        if `return_debug_png` is a str path, saves an image with
        the kept part boxes drawn on the cropped kettle.
    """
    # 1) detect whole kettle (absolute coords) & crop
    kettle_det = owl.detect_objects(image=image_pil, text_prompts="bucket",
                                    return_abs=True)
    if not kettle_det:
        raise RuntimeError("No kettle detected.")

    x1, y1, x2, y2 = map(int, kettle_det[0]["bbox"])
    W, H = image_pil.size
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    crop_pil = image_pil.crop((x1, y1, x2, y2))

    # 2) detect parts inside crop
    detections = owl.detect_objects(
        image=crop_pil,
        text_prompts=["bucket handle", "bucket"],
        return_abs=False
    )

    spout_boxes = discard_large_boxes(
        [d["bbox"] for d in detections if d["box_name"] == "bucket handle"],
        max_area_ratio=area_thresh
    )
    handle_boxes = discard_large_boxes(
        [d["bbox"] for d in detections if d["box_name"] == "bucket"],
        max_area_ratio=area_thresh
    )

    # 3) optional debug visualisation
    if return_debug_png:
        debug_dets = (
            [{"bbox": b, "box_name": "spout",  "score": 1.0} for b in spout_boxes] +
            [{"bbox": b, "box_name": "handle", "score": 1.0} for b in handle_boxes]
        )
        dbg_np = OWLViT.draw_detections(crop_pil, debug_dets)
        Image.fromarray(dbg_np).save(return_debug_png)

    # 4) run SAM on the filtered boxes
    spout_masks  = sam.segment_by_bboxes(crop_pil, spout_boxes)  if spout_boxes  else []
    handle_masks = sam.segment_by_bboxes(crop_pil, handle_boxes) if handle_boxes else []

    return handle_masks, spout_masks


def blend_mask_color(image, mask, color, alpha=0.5):
    """Blend a single-channel mask into an RGB image using the given color."""
    img = image.copy()
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[mask] = color
    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended

def run_functo_pipeline():
    # Hardcoded config
    config = {
        "task_label": "observe",
        "test_tool_label": "mug",
        "test_target_label": "bucket",
        "params": {
            "num_candidate_keypoints": 6,
            "vp_flag": False,
            "sd_dino_flag": False
        },
        "owlv2": {
            "box_threshold": 0.10,
            "text_threshold": 0.10,
            "conf_threshold": 0.05
        },
        "paths": {
            "base_data_path": "/home/oguz/Desktop/attempt_three"
        }
    }

    # Initialize detectors
    owl = OWLViT()
    sam_predictor = SAM()

    # Load config fields
    task_label = config['task_label']
    test_tool_label = config['test_tool_label']
    test_target_label = config['test_target_label']
    num_candidate_keypoints = config['params']['num_candidate_keypoints']
    vp_flag = config['params']['vp_flag']
    sd_dino_flag = config['params']['sd_dino_flag']
    base_data_path = config['paths']['base_data_path']
    data_path = os.path.join(base_data_path, 'demo_data', f'{task_label}_demo')
    test_data_path = os.path.join(base_data_path, 'test_data', f'{task_label}_test')
    owl_params = config.get("owlv2", {})


    # Load your image
    image_path = "/home/oguz/Desktop/attempt_three/test_data/observe_test/rgb/00000.jpg"
    image_pil = Image.open(image_path).convert("RGB")
    """
    # Initialize models
    sam = SAM()
    owl = OWLViT()

    # Step 1: Detect kettle and crop image
    kettle_detections = owl.detect_objects(
        image=image_pil,
        text_prompts="kettle",
        return_abs=True
    )

    if not kettle_detections:
        print("❌ No 'kettle' detected.")
        exit()

    # Get the first kettle box (assume one kettle)
    x1, y1, x2, y2 = map(int, kettle_detections[0]["bbox"])
    pad = 20  # you can increase if needed

    W, H = image_pil.size
    x1_pad = max(0, x1 - pad)
    y1_pad = max(0, y1 - pad)
    x2_pad = min(W, x2 + pad)
    y2_pad = min(H, y2 + pad)

    image_pil = image_pil.crop((x1_pad, y1_pad, x2_pad, y2_pad))

    # Detect spout and handle with OWLViT
    detections = owl.detect_objects(
        image=image_pil,
        text_prompts=["handle", "spout"],
        return_abs=False
    )

    # Extract and filter bounding boxes
    spout_boxes_raw  = [d["bbox"] for d in detections if d["box_name"] == "spout"]
    handle_boxes_raw = [d["bbox"] for d in detections if d["box_name"] == "handle"]

    # Discard overly large boxes
    spout_boxes  = discard_large_boxes(spout_boxes_raw, max_area_ratio=0.7)
    handle_boxes = discard_large_boxes(handle_boxes_raw, max_area_ratio=0.7)

    filtered_detections = []

    for box in spout_boxes:
        filtered_detections.append({
            "bbox": box,          # still in [0-1] relative coords
            "box_name": "spout",
            "score": 1.0          # dummy score just for the label
        })

    for box in handle_boxes:
        filtered_detections.append({
            "bbox": box,
            "box_name": "handle",
            "score": 1.0
        })

    # Draw the boxes on the (already-cropped) image
    boxed_img_np = OWLViT.draw_detections(image_pil, filtered_detections)

    # Save & (optionally) display
    
    boxed_pil = Image.fromarray(boxed_img_np)
    boxed_pil.save("boxes_before_segmentation.png")
    boxed_pil.show()          # comment out if .show() causes issues
    print("✅ Saved bounding-box debug image as: boxes_before_segmentation.png")

    # Run SAM to get masks
    spout_masks = sam.segment_by_bboxes(image_pil, spout_boxes) if spout_boxes else []
    handle_masks = sam.segment_by_bboxes(image_pil, handle_boxes) if handle_boxes else []

    # Overlay masks for visualization
    img_np = np.array(image_pil)
    visual = img_np.copy()

    # Color masks
    for mask in spout_masks:
        seg = mask["segmentation"]
        visual[seg] = (0.5 * visual[seg] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)  # green

    for mask in handle_masks:
        seg = mask["segmentation"]
        visual[seg] = (0.5 * visual[seg] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)  # blue

    # Save and show
    Image.fromarray(visual).save("kettle_output_masked.png")

    """
    """
    get_kettle_part_masks(
    image_pil=image_pil,
    owl=owl,  
    sam = sam_predictor,
    pad = 20,
    area_thresh = 0.65,
    return_debug_png = "show.png"
    )
    """
    # Step 1: Detect and segment tool/target
    print("############ Test target object frame transformation ############")
    print("Using text prompt:", test_tool_label)
    demo_test_detection(
        test_data_path,
        test_tool_label,
        target_label=test_target_label,
        owl_params=owl_params,
        owl=owl,
        sam_predictor=sam_predictor
    )

    #cam_to_target_trans(test_data_path)

    # Step 2: Transfer functional point
    print("############ Function point transfer ############")
    keypoints_tensor = func_point_transfer(
        data_path,
        test_data_path,
        test_tool_label,
        test_target_label,
        task_label,
        num_candidate_keypoints,
        sd_dino_flag
    )

    return keypoints_tensor

if __name__ == '__main__':
    keypoints = run_functo_pipeline()
    print("Extracted keypoints shape:", keypoints.shape)

