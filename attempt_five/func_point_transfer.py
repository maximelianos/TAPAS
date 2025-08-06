
from __future__ import annotations
import os
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
import json
import io
import base64
import requests

import torch
import torchvision.transforms as T
import timm
import random
from PIL import ImageDraw
from utils_IL.perception_utils import map_coordinates_to_original, map_coordinates_to_resized, resize_
from utils_IL.perception_utils import convert_to_serializable
from utils_IL.vp_utils import propose_candidate_keypoints
from utils_IL.vp_utils import annotate_visual_prompts
from utils_IL.vp_utils import annotate_candidate_keypoints 
from utils_IL.vp_utils import load_prompts 
from utils_IL.vp_utils import generate_vertices_and_update
from groundingdino.util.inference import load_model, predict
from local_services.sam_local import SAM
from local_services.owlv2_local import OWLViT
from torchvision import transforms
from PIL import ImageDraw
from typing import List, Dict, Tuple, Optional

#from AffKpNet import AffKpNet
import torch
from PIL import Image

import torchvision.transforms as T
from tensordict import TensorDict
from types import SimpleNamespace
import numpy as np



## If you decide to try to integrate Affkp network again, use something like this.
##def detect_affordance_keypoints(img_pil):
    # assuming the model takes PIL images
##    keypoints = model.predict_keypoints(img_pil)  
##    return keypoints  # e.g., list of (x, y) coordinates

owl = OWLViT()
sam_local = SAM()  # loads model and weights

# Load once globally
config_path = "/home/oguz/Desktop/attempt_three/GroundingDINO/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "/home/oguz/Desktop/attempt_three/GroundingDINO/weights/groundingdino_swint_ogc.pth"

dino_model = load_model(config_path, checkpoint_path)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
"""
def get_features_local(image: Image.Image) -> np.ndarray:
    img_tensor = transform(image).unsqueeze(0).to(next(dino_model.parameters()).device)
    with torch.no_grad():
        features = dino_model.forward_features(img_tensor)
    # shape: (B, C, H, W) if extracted from intermediate layers
    return features.cpu().numpy()
"""



def get_centroid(mask: np.ndarray):
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)
'''
def discard_large_boxes(
    boxes: List[List[float]],
    max_area_ratio: float,
) -> List[List[float]]:
    """Discard boxes that cover more than `max_area_ratio` of the image (normalized coords)."""
    return [
        box for box in boxes
        if (box[2] - box[0]) * (box[3] - box[1]) < max_area_ratio
    ]
'''
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



def get_spout_handle_masks(
    cropped_img: Image.Image,
    owl: OWLViT,
    sam: SAM,
    *,
    area_thresh: float = 0.7,
    debug_png: Optional[str] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect handle/spout on an *already-cropped* kettle image and return SAM masks.
    """
    # 1) detect part boxes (rel coords 0-1)
    dets = owl.detect_objects(
        image=cropped_img,
        text_prompts=["bucket handle", "bucket"],
        return_abs=False,
    )

    spout_boxes  = discard_large_boxes(
        [d["bbox"] for d in dets if d["box_name"] == "bucket handle"],  area_thresh
    )
    handle_boxes = discard_large_boxes(
        [d["bbox"] for d in dets if d["box_name"] == "bucket"], area_thresh
    )

    # optional debug visual of kept boxes
    if debug_png:
        dbg = OWLViT.draw_detections(
            cropped_img,
            ([{"bbox": b, "box_name":"spout",  "score":1.0} for b in spout_boxes] +
             [{"bbox": b, "box_name":"handle", "score":1.0} for b in handle_boxes])
        )
        Image.fromarray(dbg).save(debug_png)

    # 2) SAM segmentation per box
    spout_masks  = sam.segment_by_bboxes(cropped_img, spout_boxes)  if spout_boxes  else []
    handle_masks = sam.segment_by_bboxes(cropped_img, handle_boxes) if handle_boxes else []

    return handle_masks, spout_masks


# ──────────────────────────────────────────────────────────────
def _pick_one_point(masks, cand):
    """Return one candidate point inside ANY mask, or the centroid of the union mask if none."""
    if not masks:
        return None

    h, w = masks[0]["segmentation"].shape
    union = np.zeros((h, w), dtype=bool)
    for m in masks:
        union |= m["segmentation"]

    # First try to pick from candidate points inside mask
    inside = [
        (x, y) for x, y in cand
        if 0 <= int(x) < w and 0 <= int(y) < h and union[int(y), int(x)]
    ]

    if inside:
        return random.choice(inside)

    # Fall back: pick the centroid of the union mask
    ys, xs = np.nonzero(union)
    if len(xs) == 0 or len(ys) == 0:
        return None  # completely empty mask (shouldn't happen)

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    return (cx, cy)


def select_part_points(
    handle_masks: List[Dict],
    spout_masks : List[Dict],
    candidate_xy: List[Tuple[float, float]],
) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]]]:
    """
    Returns (handle_point, spout_point) – at most one (x,y) each,
    chosen randomly among candidates that fall into the respective masks.
    """
    handle_pt = _pick_one_point(handle_masks, candidate_xy)
    spout_pt  = _pick_one_point(spout_masks,  candidate_xy)
    return handle_pt, spout_pt


def flatten_candidate_points(kpt_dict: Dict[str, Any]) -> list[tuple[float, float]]:
    """
    Accepts a dict like {'grasped': ndarray(N,2), 'unattached': ndarray(M,2) or None}
    and returns [(x,y), …] in float pixel coords.
    """
    out: list[tuple[float, float]] = []

    for arr in kpt_dict.values():
        if arr is None:
            continue
        # allow torch tensors too
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        # ensure shape (K,2)
        arr = np.asarray(arr, dtype=float).reshape(-1, 2)
        out += [(float(x), float(y)) for x, y in arr]

    return out

def mark_selected_points_on_image(image_pil, handle_point, spout_point, save_path="annotated_points.png"):
    """
    Draws handle (blue) and spout (green) points on the image and saves it.
    """
    image = image_pil.copy()
    draw = ImageDraw.Draw(image)

    radius = 5

    if handle_point:
        x, y = handle_point
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=(0, 0, 255), outline="black")
    
    if spout_point:
        x, y = spout_point
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=(0, 255, 0), outline="black")

    image.save(save_path)
    print(f"✅ Saved image with marked points: {save_path}")




def dilate_masks(masks: list[dict], kernel_size: int = 15, iterations: int = 1) -> list[dict]:
    """
    Expand binary masks using morphological dilation.

    Args:
        masks: list of {"segmentation": np.ndarray, ...}
        kernel_size: Size of the dilation kernel (must be odd).
        iterations: How many times to apply the dilation.

    Returns:
        New list with dilated masks.
    """
    dilated_masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for mask_dict in masks:
        mask = mask_dict["segmentation"].astype(np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=iterations)
        dilated_masks.append({**mask_dict, "segmentation": dilated.astype(bool)})

    return dilated_masks

def get_features_local(image: Image.Image) -> np.ndarray:
    img_tensor = transform(image).unsqueeze(0).to(next(dino_model.parameters()).device)
    with torch.no_grad():
        # Extract last layer tokens
        intermediate_output = dino_model.get_intermediate_layers(img_tensor, n=1)[0]  # (1, 197, 768)

        # Drop CLS token and reshape
        patch_tokens = intermediate_output[:, 1:, :]  # (1, 196, 768)
        h = w = int(patch_tokens.shape[1]**0.5)       # assuming square layout
        patch_tokens = patch_tokens.reshape(1, h, w, -1).permute(0, 3, 1, 2)  # (1, 768, H, W)
    
    return patch_tokens.cpu().numpy()

## local code is above
def get_features(image):
    return get_features_local(image)

    """
    # Convert the image to a PNG byte stream
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Send the image as base64 to the server
    print("here1")
    response = requests.post("http://crane1.d2.comp.nus.edu.sg:4002/process_image", json={"image": img_encoded})
    print("here2")
    return response.json()
    """
        
def func_point_transfer(data_path, test_data_path, test_tool_label='mug', test_target_label='bowl', task_label='pour', num_candidate_keypoints=8, sd_dino_flag=False):
    """
    Function point transfer
    """

    output_path = os.path.join(test_data_path, 'func_point_transfer_output')
    os.makedirs(output_path, exist_ok=True)

    # load prompts
    prompt_path = os.path.join(test_data_path, '..', '..', 'utils_IL', 'prompts')
    prompts = load_prompts(prompt_path)

    # load demo function point in the initial keyframe
    demo_func_point_init_path = os.path.join(data_path, 'func_point_track_output', 'func_point_2d_init.json')
    with open(demo_func_point_init_path, 'r') as file:
        demo_func_point_init = json.load(file) 
    demo_func_point_init_x, demo_func_point_init_y = demo_func_point_init

    # load demo initial keyframe
    demo_init_frame_index = 0
    demo_init_frame_filenname = f'{demo_init_frame_index:05d}.jpg'
    demo_init_frame_file_path = os.path.join(data_path, 'rgb', demo_init_frame_filenname)
    demo_init_frame_pil = Image.open(demo_init_frame_file_path).convert('RGB')
    demo_init_frame_mask_path = os.path.join(data_path, 'detection_output', 'tool_mask.npy')
    demo_init_frame_mask = np.load(demo_init_frame_mask_path)

    # load test initial keyframe
    test_init_frame_idx = 0
    test_init_frame_filename = f'{test_init_frame_idx:05d}.jpg'
    test_init_frame_file_path = os.path.join(test_data_path, 'rgb', test_init_frame_filename)
    test_init_frame_pil =  Image.open(test_init_frame_file_path).convert('RGB')
    test_init_frame = np.array(test_init_frame_pil)
    test_init_frame_mask_path = os.path.join(test_data_path, 'detection_output', 'tool_mask.npy')
    test_init_frame_mask = np.load(test_init_frame_mask_path)

    ############ Demo tool processing ############ 

    # load demo tool bbox
    demo_tool_box_path = os.path.join(data_path, 'detection_output', 'tool.json')
    with open(demo_tool_box_path, 'r') as file:
        demo_tool_box = json.load(file)
    x_min, y_min, x_max, y_max = demo_tool_box
    x_min *= 0.95
    y_min *= 0.95
    x_max *= 1.03
    y_max *= 1.03
    demo_tool_box = [x_min, y_min, x_max, y_max]

    # crop demo tool and plot function point
    demo_init_frame_pil_crop = demo_init_frame_pil.crop(demo_tool_box)
    demo_init_frame_pil_crop.save(os.path.join(output_path, 'demo_tool_crop.png'))
    demo_func_point_2d_init_x_crop, demo_func_point_2d_init_y_crop = int(demo_func_point_init_x - demo_tool_box[0]), int(demo_func_point_init_y - demo_tool_box[1])
    demo_func_point_2d_init_crop = [demo_func_point_2d_init_x_crop, demo_func_point_2d_init_y_crop]
    demo_annotate_keypoints = {'grasped': [demo_func_point_2d_init_crop]}
    demo_init_frame_pil_crop_annotated = annotate_candidate_keypoints(demo_init_frame_pil_crop, demo_annotate_keypoints, add_caption=False)
    # hack: flip for better global alignment
    demo_init_frame_pil_crop_annotated = demo_init_frame_pil_crop_annotated.transpose(Image.FLIP_LEFT_RIGHT)
    demo_init_frame_pil_crop_annotated.save(os.path.join(output_path, 'demo_tool_func_point_init_vis.jpg'))

    # crop demo tool mask
    demo_init_frame_mask_pil = Image.fromarray(demo_init_frame_mask)
    demo_init_frame_mask_pil_crop = demo_init_frame_mask_pil.crop(demo_tool_box)
    demo_init_frame_mask_crop = np.array(demo_init_frame_mask_pil_crop).astype(np.uint8)
    np.save(os.path.join(output_path, 'demo_tool_mask_crop.npy'), demo_init_frame_mask_crop)
    demo_init_frame_mask_pil_crop.save(os.path.join(output_path, 'demo_tool_mask_crop.png'))

    ############ Test tool processing ############ 

    # load test tool bbox
    test_tool_box_path = os.path.join(test_data_path, 'detection_output', 'tool.json')
    with open(test_tool_box_path, 'r') as file:
        test_tool_box = json.load(file)
    x_min, y_min, x_max, y_max = test_tool_box
    #x_min *= 0.95
    #y_min *= 0.95
    #x_max *= 1.03
    #y_max *= 1.03
    x_min = x_min - 20
    y_min = y_min - 20
    x_max = x_max + 20
    y_max = y_max + 20
    test_tool_box = [x_min, y_min, x_max, y_max]

    # crop test tool and mask
    test_init_frame_pil_crop = test_init_frame_pil.crop(test_tool_box)

    x_min, y_min, _, _ = test_tool_box  # Get crop origin
    crop_offset = np.array([x_min, y_min])

    test_init_frame_pil_crop.save(os.path.join(output_path, 'test_tool_crop.png'))
    test_init_frame_mask = np.load(os.path.join(test_data_path, 'detection_output', 'tool_mask.npy'))
    test_init_frame_mask_pil = Image.fromarray(test_init_frame_mask)
    test_init_frame_mask_pil_crop = test_init_frame_mask_pil.crop(test_tool_box)
    test_init_frame_mask_crop = np.array(test_init_frame_mask_pil_crop).astype(np.uint8)
    np.save(os.path.join(output_path, 'test_tool_mask_crop.npy'), test_init_frame_mask_crop)
    test_init_frame_mask_pil_crop.save(os.path.join(output_path, 'test_tool_mask_crop.png'))

    ############ Coarse-grained region proposal ############ 
    segmasks = {}
    segmasks[test_tool_label] = {'mask': test_init_frame_mask_crop}
    task_instruction = f'use a {test_tool_label} to {task_label} {test_target_label}'

    # ignore target objects
    task = {}
    task['object_grasped'] = test_tool_label
    task['object_unattached'] = ''
    task['task_instruction'] = task_instruction
 
    segmasks = generate_vertices_and_update(segmasks)

    # Annotate visual marks.
    candidate_keypoints = propose_candidate_keypoints(
        task,
        segmasks, 
        num_samples=num_candidate_keypoints)
            
    annotated_image = annotate_visual_prompts(
                test_init_frame_pil_crop,
                candidate_keypoints)
    annotated_image.save(os.path.join(output_path, 'visual_prompting_candidates.png'))



    print(type(candidate_keypoints), len(candidate_keypoints))
    print(candidate_keypoints)
    

    ## TODO
    ## Use groundingSAM to get functional, center and holding keypoints
    print("in-context function point transfer (GroundingDINO + Local SAM)")
    crop_d = (x_max - x_min) * (y_max - y_min)
    handle_masks, spout_masks = get_spout_handle_masks(
    test_init_frame_pil_crop, owl, sam_local,
    area_thresh=0.75,
    debug_png="boxes_before_segmentation.png"
    )   
    
    handle_masks = dilate_masks(handle_masks, kernel_size=25, iterations=1)
    spout_masks  = dilate_masks(spout_masks, kernel_size=25, iterations=1)

    candidate_points = flatten_candidate_points(candidate_keypoints)

    handle_pt, spout_pt = select_part_points(handle_masks, spout_masks, candidate_points)

    print("chosen handle point:", handle_pt)
    print("chosen spout  point:", spout_pt)
   
    mark_selected_points_on_image(test_init_frame_pil_crop, handle_pt, spout_pt)

    x_offset, y_offset, _, _ = test_tool_box
    print("function point transfer done.")
    # Get keypoints in pixel coordinates
    # Convert keypoints to normalized coordinates relative to crop
    keypoints_px = np.array([handle_pt, spout_pt], dtype=np.float32)  # (N, 2)
    crop_w, crop_h = test_init_frame_pil_crop.size



    print("[FUNC] Unnormalized candidate keypoints in crop:")
    print(keypoints_px)  # This is in crop coordinates: (N, 2)

    keypoints_norm = keypoints_px.astype(np.float32)
    keypoints_norm[:, 0] = (keypoints_norm[:, 0] / crop_w) * 2 - 1
    keypoints_norm[:, 1] = (keypoints_norm[:, 1] / crop_h) * 2 - 1

    keypoints_tensor = torch.tensor(keypoints_norm, dtype=torch.float32).unsqueeze(0)  # shape (1, N, 2)


    # Extract top-left offset from test_tool_box
    x_offset, y_offset, _, _ = test_tool_box


    # Return both tensor and crop metadata
    return keypoints_tensor, {
        "crop_offset": (x_offset, y_offset),
        "crop_size": (crop_w, crop_h),
    }


