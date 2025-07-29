
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

from utils_IL.perception_utils import map_coordinates_to_original, map_coordinates_to_resized, resize_
from utils_IL.perception_utils import convert_to_serializable
from utils_IL.vp_utils import propose_candidate_keypoints
from utils_IL.vp_utils import annotate_visual_prompts
from utils_IL.vp_utils import annotate_candidate_keypoints 
from utils_IL.vp_utils import load_prompts 
from utils_IL.vp_utils import generate_vertices_and_update
from groundingdino.util.inference import load_model, predict
from local_services.sam_local import SAM
from torchvision import transforms
from PIL import ImageDraw

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

# Load once globally
dino_model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
dino_model.eval()
dino_model.to("cuda" if torch.cuda.is_available() else "cpu")

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
    x_min *= 0.95
    y_min *= 0.95
    x_max *= 1.03
    y_max *= 1.03
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
    

    ## TODO
    ## Use groundingSAM to get functional, center and holding keypoints

    sam_local = SAM()  # loads model and weights

    print("in-context function point transfer (GroundingDINO + Local SAM)")
    context_tensor, context_cor, vp_img = keypoint_transfer(
        task,
        test_init_frame_pil_crop,
        [demo_init_frame_pil_crop_annotated, annotated_image],
        candidate_keypoints,
        prompts=prompts['select_motion_func_demo'],
        sam_local=sam_local,
        debug=True
    )

    print("Returned keypoints:", context_cor)
    print(task)
    print(context_tensor)

    vp_func_point = context_cor['holding']
    vp_img.save(os.path.join(output_path, 'visual_prompting_result.png'))

    x_offset, y_offset, _, _ = test_tool_box
    print("function point transfer done.")
    # Get keypoints in pixel coordinates
    # Convert keypoints to normalized coordinates relative to crop
    keypoints_px = np.array(candidate_keypoints['grasped'])  # (N, 2)
    crop_w, crop_h = test_init_frame_pil_crop.size



    print("[FUNC] Unnormalized candidate keypoints in crop:")
    print(keypoints_px)  # This is in crop coordinates: (N, 2)

    keypoints_norm = keypoints_px.astype(np.float32)
    keypoints_norm[:, 0] = (keypoints_norm[:, 0] / crop_w) * 2 - 1
    keypoints_norm[:, 1] = (keypoints_norm[:, 1] / crop_h) * 2 - 1

    keypoints_tensor = torch.tensor(keypoints_norm, dtype=torch.float32).unsqueeze(0)  # shape (1, N, 2)

    # Extract top-left offset from test_tool_box
    x_offset, y_offset, _, _ = test_tool_box


    if sd_dino_flag:
        ############## Fine-grained point transfer ##############
        # load demo and test images
        img_size = 480
        img1, sf1, pd1 = resize_(demo_init_frame_pil_crop.convert('RGB'), target_res=img_size, resize=True, to_pil=True)
        img2, sf2, pd2 = resize_(test_init_frame_pil_crop.convert('RGB'), target_res=img_size, resize=True, to_pil=True)

        # load test tool mask
        mask2_raw = test_init_frame_mask_pil_crop
        mask2, sf_m1, pd_m1 = resize_(mask2_raw, target_res=img_size, resize=True, to_pil=False)

        demo_source_x, demo_source_y = map_coordinates_to_resized(demo_func_point_2d_init_crop[0], demo_func_point_2d_init_crop[1], sf1, pd1)
        demo_source_x, demo_source_y = np.int64(demo_source_x), np.int64(demo_source_y)
        test_source_x, test_source_y = map_coordinates_to_resized(vp_func_point[0], vp_func_point[1], sf2, pd2)
        test_source_x, test_source_y = np.int64(test_source_x), np.int64(test_source_y)

        # create vp region mask
        x = np.arange(0, img_size, 1)
        y = np.arange(0, img_size, 1)
        x, y = np.meshgrid(x, y)
        scale = 0.15
        rect_width = (img_size -  pd2[0]*2) * scale # Width of the rectangle
        rect_height = (img_size -  pd2[1]*2) * scale  # Height of the rectangle
        rect_center_x = test_source_x  # X-coordinate of rectangle center
        rect_center_y = test_source_y  # Y-coordinate of rectangle center
        # Calculate region boundaries
        x_min = rect_center_x - rect_width / 2
        x_max = rect_center_x + rect_width / 2
        y_min = rect_center_y - rect_height / 2
        y_max = rect_center_y + rect_height / 2
        # Create the region mask
        vp_mask_normalized = np.where((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max), 1, 0)

        # function point transfer with SD+DINO
        """
        response_1 = get_features(img1)
        features_bytes_1 = base64.b64decode(response_1['features'])
        feat1 = np.load(io.BytesIO(features_bytes_1))
        response_2 = get_features(img2)
        features_bytes_2 = base64.b64decode(response_2['features'])
        feat2 = np.load(io.BytesIO(features_bytes_2))
        feat1_cuda = torch.tensor(feat1).to('cuda')
        feat2_cuda = torch.tensor(feat2).to('cuda')
        ft = torch.cat([feat1_cuda, feat2_cuda], dim=0)
        """

        feat1 = get_features(img1)  # Now returns a NumPy array
        feat2 = get_features(img2)

        feat1_cuda = torch.from_numpy(feat1).to('cuda')
        feat2_cuda = torch.from_numpy(feat2).to('cuda')

        ft = torch.cat([feat1_cuda, feat2_cuda], dim=0)

        # compute cosine similarity map
        num_channel = ft.size(1)
        cos = nn.CosineSimilarity(dim=1)
        src_ft = ft[0].unsqueeze(0)  # []
        src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)
        src_vec = src_ft[0, :, demo_source_y, demo_source_x].view(1, num_channel, 1, 1)  # 1, C, 1, 1
        trg_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(ft[1:]) # 1, C, H, W
        cos_map = cos(src_vec, trg_ft).cpu().numpy()    # 1, H, W
        # search correspondence within the test tool region mask
        cos_map = np.multiply(mask2, cos_map)
        cos_map = np.multiply(vp_mask_normalized, cos_map)  # no vp

        print("fine-grained point transfer done.")

        # compute test function point
        max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
                    
        # test function point visualization
        original_x, original_y = map_coordinates_to_original(int(max_yx[1]), int(max_yx[0]), sf2, pd2)
        test_init_frame_x, test_init_frame_y = int(original_x + x_offset), int(original_y + y_offset) # VP + SD+DINO
    else:
        test_init_frame_x, test_init_frame_y = int(vp_func_point[0] + x_offset), int(vp_func_point[1] + y_offset)  # no SD+DINO, VP only

    # Return both tensor and crop metadata
    return keypoints_tensor, {
        "crop_offset": (x_offset, y_offset),
        "crop_size": (crop_w, crop_h),
    }


## TODO update this
"""
context, context_cor, vp_img = keypoint_transfer(
    task,
        test_init_frame_pil_crop,
        [demo_init_frame_pil_crop_annotated, annotated_image],
        candidate_keypoints,
        prompts=prompts['select_motion_func_demo'], 
        debug=True)
"""


"""
def _tapas_single_salient_point(pil_crop: Image.Image) -> tuple[int, int]:
    ''''''Run KeypointEncoder on a PIL crop and return (x_px, y_px) in *crop* coords.'''
    rgb = _TRANSFORM(pil_crop).unsqueeze(0).to(_DEVICE)          # (1,3,H,W)
    depth = torch.ones(1, _IMG_H, _IMG_W, device=_DEVICE)        # dummy
    intr  = torch.tensor([[500.,0,320.],[0,500.,240.],[0,0,1.]],
                         device=_DEVICE)

    cam_obs = SingleCamObservation(rgb=rgb, depth=depth,
                                   intr=intr, extr=torch.eye(4, device=_DEVICE))
    cam_dict = TensorDict({"wrist": cam_obs,
                           "_order": SimpleNamespace(order=["wrist"])},
                          batch_size=[])

    scene = SceneObservation(cameras=cam_dict)

    with torch.no_grad():
        kp_raw, info = _TAPAS.encode(scene)

    # -- pick the most salient keypoint via the highest entry in sm map --
    sm = info["sm"]
    if isinstance(sm, tuple):
        sm = sm[0]  # TAPAS sometimes returns tuples

    if sm.dim() == 4:
        sm = sm.squeeze(0)  # Remove batch dim if present

    N, H, W = sm.shape
    flat_idx = torch.argmax(sm.view(N, -1))
    kp_idx   = flat_idx // (H*W)
    pix_idx  = flat_idx %  (H*W)
    y = (pix_idx // W).item()
    x = (pix_idx %  W).item()

    # NOTE: sm is on the (H,W)=(_IMG_H,_IMG_W) grid we fed TAPAS
    # If you resized to (_IMG_H,_IMG_W) already, x,y are pixel on the crop.
    return int(x), int(y), int(kp_idx), info, kp_raw.squeeze(0).cpu() 
"""


def get_centroid(mask: np.ndarray):
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def keypoint_transfer(task, image_pil_crop, ref_images, candidate_keypoints, prompts, sam_local, debug=False):
    """
    Extracts 3 functional keypoints (holding, functional, center) using DINO + your local SAM.
    
    Params:
        task: task dict
        image_pil_crop: cropped RGB PIL image of the tool
        prompts: dict of prompts (text descriptions)
        sam_local: instance of your local SAM wrapper (class `SAM`)
    
    Returns:
        - keypoints_tensor: torch.Tensor of shape (1, 3, 2)
        - keypoints_dict: dict with 'holding', 'functional', 'center'
        - debug_img: PIL image with keypoints drawn
    """
    grounding_dino = load_model("GroundingDINO/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to torch.Tensor and scales [0,255] to [0.0,1.0]
    ])

    image_tensor = transform(image_pil_crop).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    def get_mask_from_prompt(prompt_text: str):
        boxes, logits, phrases = predict(
            model=grounding_dino,
            image=image_tensor[0],
            caption=prompt_text,
            box_threshold=0.3,
            text_threshold=0.25
        )
        if len(boxes) == 0:
            return None

        # Convert to relative coords (0–1)
        W, H = image_pil_crop.size
        boxes_rel = []
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            boxes_rel.append([x1 / W, y1 / H, x2 / W, y2 / H])

        masks = sam_local.segment_by_bboxes(image_pil_crop, boxes_rel)
        return masks[0]["segmentation"]

    # 1. Holding (handle)
    holding_mask = get_mask_from_prompt("handle")
    holding_kp = get_centroid(holding_mask) if holding_mask is not None else None

    # 2. Functional (spout, tip)
    functional_mask = get_mask_from_prompt("spout")
    functional_kp = get_centroid(functional_mask) if functional_mask is not None else None

    # 3. Center (of full object mask)
    full_mask = np.array(image_pil_crop.convert("L")) > 0  # Replace with correct mask if available
    center_kp = get_centroid(full_mask)

    # Normalize and convert to torch tensor
    w, h = image_pil_crop.size
    def norm(xy): return [(xy[0] / w) * 2 - 1, (xy[1] / h) * 2 - 1] if xy else [0.0, 0.0]

    keypoints_tensor = torch.tensor([[norm(holding_kp), norm(functional_kp), norm(center_kp)]], dtype=torch.float32)
    keypoints_dict = {
        "holding": holding_kp,
        "functional": functional_kp,
        "center": center_kp
    }

    # Optional debug image
    if debug:
        draw = ImageDraw.Draw(image_pil_crop)
        if holding_kp:
            draw.ellipse([holding_kp[0]-4, holding_kp[1]-4, holding_kp[0]+4, holding_kp[1]+4], fill="blue")
        if functional_kp:
            draw.ellipse([functional_kp[0]-4, functional_kp[1]-4, functional_kp[0]+4, functional_kp[1]+4], fill="red")
        if center_kp:
            draw.ellipse([center_kp[0]-4, center_kp[1]-4, center_kp[0]+4, center_kp[1]+4], fill="green")
        image_pil_crop.save("debug_sam_grounding_kps.png")


    print("Holding raw:", holding_kp)
    print("Functional raw:", functional_kp)
    print("Center raw:", center_kp)
    return keypoints_tensor, keypoints_dict, image_pil_crop