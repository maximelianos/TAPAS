# test_keypoint_detector.py

from pathlib import Path
from PIL import Image
from keypoint_detector import KeypointDetector
from keypoint_encoder import KeypointEncoder

"""
detector = KeypointDetector()                          # CUDA if available
img_path = "sample.png"                                # your cropped image
out_dir  = Path("tapas_test_out")                      # where to dump results

result = detector.run_on_image(img_path, out_dir, prefix="sample")

print("⏺  TAPAS most-salient key-point")
print("   pixel      :", result["pixel"])
print("   normalised :", result["normalised"])
print("   N° index   :", result["kp_index"])
print(f"\nVisualisation saved to: {out_dir / 'sample_viz.png'}")
"""

keypoints, top1, vis_img = KeypointEncoder.detect_keypoints("sample.png", save_outputs=True)
print("Top-1 keypoint (pixels):", top1)
vis_img.show()