import torch

import sys
import os

from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tensordict import TensorDict
from types import SimpleNamespace
import pandas as pd
import numpy as np
from io import BytesIO

# Add root project path so TAPAS and other modules are found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TAPAS")))
from TAPAS.tapas_gmm.encoder.vit_extractor import VitKeypointsPredictor
from TAPAS.conf.encoder.vit_keypoints.nofilter import vit_keypoints_predictor_config
from TAPAS.tapas_gmm.utils.observation import SceneObservation, SingleCamObservation

class KeypointEncoder(VitKeypointsPredictor):
    def __init__(self):
        super().__init__(vit_keypoints_predictor_config)
        self.eval()
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def detect_keypoints(image_path, save_outputs=False, output_prefix="keypoints_output"):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = 640, 480
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((img_height, img_width)),
        ])
        input_tensor = transform(img).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)

        # Load model
        model = KeypointEncoder()
        model.eval()

        # Construct SceneObservation
        rgb = input_tensor
        depth = torch.ones(1, img_height, img_width, device=device)
        intr = torch.tensor([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
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

        # Encode keypoints
        encoded, info = model.encode(scene_obs)
        crop_offset = info["crop_offset"]  # Add this if passing offset through info
        kp_raw_2d = info["kp_raw_2d"]
        if isinstance(kp_raw_2d, tuple):
            kp_raw_2d = kp_raw_2d[0]

        keypoints_norm = kp_raw_2d.squeeze(0).detach().cpu().numpy()
        if keypoints_norm.ndim == 1:
            keypoints_norm = keypoints_norm.reshape(-1, 2)

        # Denormalize
        keypoints_px = (keypoints_norm + 1) / 2
        keypoints_px[:, 0] *= img_width
        keypoints_px[:, 1] *= img_height

        keypoints_px += crop_offset


        # Extract most salient keypoint
        sm_tensor = info["sm"][0].squeeze(0)  # shape: (N, H, W)
        N, H, W = sm_tensor.shape
        flat_idx = torch.argmax(sm_tensor.view(N, -1))
        kp_idx = flat_idx // (H * W)
        pixel_idx = flat_idx % (H * W)
        y = (pixel_idx // W).item()
        x = (pixel_idx % W).item()

        x_px = x * (img_width / W)
        y_px = y * (img_height / H)

        # Visualization
        img_resized = img.resize((img_width, img_height))
        plt.figure(figsize=(8, 6))
        plt.imshow(img_resized)
        plt.scatter(keypoints_px[:, 0], keypoints_px[:, 1], c='red', s=40, label='Keypoints')
        plt.scatter([x_px], [y_px], c='blue', s=80, label='Top-1 Keypoint')
        plt.legend()
        plt.title("Detected Keypoints (TAPAS)")
        plt.axis("off")
        plt.tight_layout()

        # Save image if needed
        if save_outputs:
            vis_path = f"{output_prefix}_visualized.png"
            plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
            print(f"Saved visualization to {vis_path}")

        # Save keypoints to CSV if needed
        if save_outputs:
            pd.DataFrame(keypoints_px, columns=["x", "y"]).to_csv(f"{output_prefix}_all.csv", index=False)
            pd.DataFrame([[x_px, y_px]], columns=["x", "y"]).to_csv(f"{output_prefix}_top1.csv", index=False)
            print(f"Saved keypoints to CSV with prefix '{output_prefix}'")

        # Convert visualization to PIL image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        vis_image = Image.open(buf)

        plt.close()

        return keypoints_px, (x_px, y_px), vis_image