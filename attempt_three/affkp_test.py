import sys
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

# Add the AffKpNet path to PYTHONPATH
sys.path.append("C:/freiburg/second_term/deep learning lab/project/attempt_three/AffKpNet/AffKpNet")

# Import the model
from encoding.models.danet_kp import DANetKp

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((480, 480)),
        T.ToTensor(),
        T.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    return transform(img).unsqueeze(0), img

def visualize_keypoints(img_pil, heatmap):
    heatmap_np = heatmap.squeeze().detach().cpu().numpy()
    max_idx = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    y, x = max_idx
    img_cv = np.array(img_pil)
    img_cv = cv2.circle(img_cv, (x, y), 6, (255, 0, 0), -1)
    return Image.fromarray(img_cv)

def main():
    # Load model
    model = DANetKp(nclass=7, backbone='resnet50')  # UMD dataset uses 7
    model.eval().cuda()

    # Load weights if you have any
    # model.load_state_dict(torch.load("your_checkpoint.pth"))

    # Input image
    input_tensor, img_pil = preprocess_image("test_tool_crop.png")
    input_tensor = input_tensor.cuda()

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Visualize first keypoint head (e.g. '1')
    if '1' in output:
        keypoint_map = output['1'][0]  # shape: (num_classes, H, W)
        vis = visualize_keypoints(img_pil, keypoint_map[0])  # visualize first class
        vis.save("affkp_output.png")
        print("Saved output as affkp_output.png")
    else:
        print("No '1' head found in output. Available keys:", output.keys())

if __name__ == "__main__":
    main()
