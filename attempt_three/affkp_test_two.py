import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

from AffKpNet.AffKpNet.encoding.models.danet_kp import DANetKp  # adjust if needed

# Load image
image_path = "your_test_image.jpg"  # <-- change this to your image path
img_pil = Image.open(image_path).convert("RGB")

# Transform
transform = T.Compose([
    T.Resize((480, 480)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
img_tensor = transform(img_pil).unsqueeze(0).cuda()

# Load model
model = DANetKp(nclass=8, backbone='resnet50', aux=False)
model.eval().cuda()

# Forward pass
with torch.no_grad():
    outputs = model(img_tensor)

# Visualize heatmap for first head (e.g., '1')
heatmap = outputs['1'][0][0].cpu().numpy()  # (H, W)
plt.imshow(heatmap, cmap='hot')
plt.title("Affordance Keypoint Heatmap")
plt.axis('off')
plt.show()