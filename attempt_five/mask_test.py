import numpy as np
from PIL import Image
import cv2
from local_services.sam_local import SAM
from local_services.owlv2_local import OWLViT  # adjust if needed
import torch 

# Load image
image_path = "kettle.png"  # ← Change this to your actual image
image_pil = Image.open(image_path).convert("RGB")

# Initialize models
sam = SAM()
owl = OWLViT()

# Run OWLViT to get bounding boxes for handle and spout
detections = owl.detect_objects(
    image=image_pil,
    text_prompts=["handle", "spout"],
    return_abs=False  # keep boxes in normalized [0,1] format for SAM
)

if not detections:
    print("❌ No detections found.")
    exit()

# Separate boxes for each class
handle_boxes = [d["bbox"] for d in detections if d["box_name"] == "handle"]
spout_boxes  = [d["bbox"] for d in detections if d["box_name"] == "spout"]

if not handle_boxes:
    print("⚠️  No 'handle' boxes found.")
if not spout_boxes:
    print("⚠️  No 'spout' boxes found.")

# Run SAM segmentation
handle_masks = sam.segment_by_bboxes(image_pil, handle_boxes) if handle_boxes else []
spout_masks  = sam.segment_by_bboxes(image_pil, spout_boxes)  if spout_boxes else []

# Convert image for visualization
img_np = np.array(image_pil)
visual = img_np.copy()

# Overlay handle masks (blue)
for mask_dict in handle_masks:
    mask = mask_dict["segmentation"]
    visual[mask > 0] = (
        0.5 * visual[mask > 0] + 0.5 * np.array([0, 0, 255], dtype=np.uint8)
    ).astype(np.uint8)

# Overlay spout masks (green)
for mask_dict in spout_masks:
    mask = mask_dict["segmentation"]
    visual[mask > 0] = (
        0.5 * visual[mask > 0] + 0.5 * np.array([0, 255, 0], dtype=np.uint8)
    ).astype(np.uint8)

# Show or save the result
out_img = Image.fromarray(visual)
out_img.save("kettle_mask_output.png")
out_img.show()