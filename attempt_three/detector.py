# local_models/detector.py
from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry
import cv2
from PIL import Image

# Load GroundingDINO model once
dino_model = load_model("path/to/GroundingDINO/weights/groundingdino_swint_ogc.pth")

# Load SAM model once
sam = sam_model_registry["vit_h"](checkpoint="path/to/sam_vit_h.pth")
sam_predictor = SamPredictor(sam)

def detect_and_segment(image_path, prompts):
    image = Image.open(image_path).convert("RGB")
    detections = predict(dino_model, image, prompts)

    # Convert to OpenCV format for SAM
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    sam_predictor.set_image(image_cv)

    result = []
    for det in detections:
        box = det["bbox"]
        label = det["label"]

        sam_masks = sam_predictor.predict(box=box, multimask_output=True)
        result.append({
            "label": label,
            "bbox": box,
            "mask": sam_masks[0]  # Use first mask
        })
    
    return result