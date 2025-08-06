from keypoint_encoder import KeypointEncoder
import torch
from PIL import Image
import torchvision.transforms as T
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensordict import TensorDict
from types import SimpleNamespace
import pandas as pd
import numpy as np
from PIL import ImageDraw
import pandas as pd


def draw_keypoints_on_image(image_pil, keypoints, color='red', radius=5, save_path=None):
    """
    Draws keypoints on a PIL image and saves it optionally.

    Args:
        image_pil (PIL.Image): The original image.
        keypoints (ndarray): Array of shape (N, 2) with keypoints as (x, y).
        color (str or tuple): Color of the points.
        radius (int): Radius of each point.
        save_path (str): Optional path to save the image.
    """
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)

    for (x, y) in keypoints:
        x, y = float(x), float(y)
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)

    if save_path:
        img_draw.save(save_path)
        print(f"✅ Saved keypoints image at: {save_path}")
    
    return img_draw

torch.backends.cuda.matmul.allow_tf32 = True
# TAPAS utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TAPAS")))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from TAPAS.tapas_gmm.utils.observation import SceneObservation, SingleCamObservation

# === 1. Load and preprocess image ===
image_path = "/home/oguz/Desktop/attempt_three/test_data/observe_test/rgb/00000.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

img = Image.open(image_path).convert("RGB")
img_width, img_height = img.size  # ← Use real image size

transform = T.Compose([
    T.ToTensor(),
    T.Resize((img_height, img_width)),
])
input_tensor = transform(img).unsqueeze(0)

# === 2. Load TAPAS model ===
model = KeypointEncoder()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

# === 3. Construct SceneObservation ===
rgb = input_tensor
depth = torch.ones(1, img_height, img_width, device=device)  # dummy depth
intr = torch.tensor([
    [500.0, 0.0, img_width / 2],
    [0.0, 500.0, img_height / 2],
    [0.0,   0.0,   1.0]
], device=device)

cam_obs = SingleCamObservation(
    rgb=rgb,
    depth=depth,
    intr=intr,
    extr=torch.eye(4, device=device)
)

camera_order = SimpleNamespace(order=["wrist"])
camera_dict = TensorDict(
    {
        "wrist": cam_obs,
        "_order": camera_order,
    },
    batch_size=[]
)
scene_obs = SceneObservation(cameras=camera_dict)

# === 4. Encode image to extract keypoints ===
encoded, info = model.encode(scene_obs)
print("Encoded shape:", encoded.shape)
print("Info keys:", info.keys())

# === 5. Extract and process keypoints ===
kp_raw_2d = info["kp_raw_2d"]
if isinstance(kp_raw_2d, tuple):
    print("Unpacking tuple from kp_raw_2d")
    kp_raw_2d = kp_raw_2d[0]

keypoints_norm = kp_raw_2d.squeeze(0).detach().cpu().numpy()
if keypoints_norm.ndim == 1:
    keypoints_norm = keypoints_norm.reshape(-1, 2)

# === 6. Convert normalized keypoints to pixel coordinates in full image ===
x_offset, y_offset = info["crop_offset"]
crop_w, crop_h = info["crop_size"]

# Denormalize to crop coordinates
keypoints_px = (keypoints_norm + 1) / 2
keypoints_px[:, 0] *= crop_w
keypoints_px[:, 1] *= crop_h

# Save this version before applying offset
keypoints_px_crop = keypoints_px.copy()

# Shift to full image using offset
keypoints_px[:, 0] += x_offset
keypoints_px[:, 1] += y_offset

# === DEBUG PRINT ===
print("[TEST] Keypoints in full image after denormalizing and shifting:")
print(keypoints_px)

#Remember:  x1 y1 is handle coordinates. x2 y2 is spout coordinates
# Convert to list and flatten
flattened = keypoints_px.flatten().tolist()  # [x1, y1, x2, y2]

# Create DataFrame
df = pd.DataFrame([flattened], columns=['x1', 'y1', 'x2', 'y2'])

# Add empty 'correct' column
df['correct'] = ''

# Save to CSV
df.to_csv('keypoints_on_image.csv', index=False)

print("CSV file 'keypoints.csv' created.")



print("[TEST] Reconstructed keypoints in crop (before offset):")
print(keypoints_px_crop)
print(f"Crop box offset: ({x_offset}, {y_offset})")
print(f"Crop size: ({crop_w}, {crop_h})")
print(f"Full image size: ({img_width}, {img_height})")

# === 7. DEBUG VISUALIZATION ===

# Plot keypoints in cropped image
"""
plt.figure(figsize=(6, 5))
cropped_img = img.crop((x_offset, y_offset, x_offset + crop_w, y_offset + crop_h))
plt.imshow(cropped_img)
plt.scatter(keypoints_px_crop[:, 0], keypoints_px_crop[:, 1], c='green', s=40, label='Crop Keypoints')
plt.title("Keypoints in Cropped Image")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.savefig("debug_crop_keypoints.png")
plt.show()
"""
"""
# Plot keypoints in full image + crop box
plt.figure(figsize=(8, 6))
plt.imshow(img.resize((img_width, img_height)))
plt.scatter(keypoints_px[:, 0], keypoints_px[:, 1], c='red', s=40)
##plt.gca().add_patch(
##    patches.Rectangle((x_offset, y_offset), crop_w, crop_h,
##                 linewidth=2, edgecolor='blue', facecolor='none', label='Crop Box')
##)
##plt.title("Keypoints in Full Image with Crop Box")
plt.axis("off")
plt.legend()
plt.tight_layout()
##plt.savefig("debug_full_image_keypoints.png")
plt.show()

print("[DEBUG] Saved debug visualizations to 'debug_crop_keypoints.png' and 'debug_full_image_keypoints.png'")

# === 8. Save keypoints ===
df_all = pd.DataFrame(keypoints_px, columns=["x", "y"])
df_all.to_csv("keypoints_all.csv", index=False)
print("Keypoints saved to 'keypoints_all.csv'")
"""

draw_keypoints_on_image(
    img,       # full original image
    keypoints_px,              # the 2D keypoints as NumPy array
    color='red',
    radius=5,
    save_path="keypoints_on_image.jpg"
)

"""
# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Check outputs
print("Inference successful!")
print("Output keys:", output.keys())
if "kp_2d" in output:
    print("Keypoints shape:", output["kp_2d"].shape)
if "heatmap" in output:
    print("Heatmap shape:", output["heatmap"].shape)
    """

