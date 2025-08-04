import sys
from pathlib import Path
import json

import cv2
import numpy as np

data_dir = Path("images")
n_episode = int(sys.argv[1])

# ======= Camera extrinsics =======
with open(data_dir / "annotations.json", "r") as f:
    annotations = json.load(f)
C = np.array(annotations[n_episode]["extrinsics"])

# ======= Camera intrinsics (replace with real values) =======
K = np.array( [
  [761.18274,   0.0,      320.0     ],
 [  0.0,      761.18274, 240.0     ],
 [  0.0,        0.0 ,       1.0     ]]
)

# ======= Load RGB and depth image =======
image = cv2.imread(data_dir / f"{n_episode:04d}_color.png")  # RGB image
depth_map = cv2.imread(data_dir / f"{n_episode:04d}_depth.png", cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001  # mm to meters

max_depth = float(depth_map.max())
#cv2.imwrite("depth-view.png", (depth_map/max_depth*65536.0*0.8).astype(np.uint16)) # mm to meters

if image is None or depth_map is None:
    raise RuntimeError("Failed to load image or depth map.")

# ======= Convert 2D pixel to 3D point =======
def pixel_to_3d(u, v, depth_map, K):
    Z = depth_map[v, u]
    if Z <= 0:
        print(f"No valid depth at ({u}, {v})")
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

def pt3d_to_pixel(x: float, y: float, z: float, K: np.ndarray):
    """ 
    Project a 3D point (x, y, z) in camera space to 2D point -
    multiply by intrinsic in homogenous coordinates.
    
    Arguments:
        x, y, z: float coordinates
        K: (3, 3), intrinsic matrix
    Returns:
        (u, v)
    """
    M = np.zeros((3, 4))
    M[:3, :3] = K
    
    p3d = np.ones((4))
    p3d[:3] = [x, y, z]
    
    p2d = M @ p3d # (3, 4) x (4, 1)
    p2d = p2d / p2d[2] # homogenous component
    
    u, v = p2d[:2]
    return np.array([u, v])

# ======= Mouse callback =======
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(depth_map[y, x])
        pt3d = pixel_to_3d(x, y, depth_map, K)
        if pt3d is not None:
            # check projection back to 2d
            X, Y, Z = pt3d
            pt2d_check = pt3d_to_pixel(X, Y, Z, K)
            assert np.allclose(np.array([x, y]), pt2d_check)
            
            # to world coors
            pt3d_hom = np.array([1., 1., 1., 1.]) # homogenous component
            pt3d_hom[:3] = pt3d
            world = C @ pt3d_hom
            
            x_1 = np.array([0, 0, 0, 1])
            x_0 = C @ x_1
            print(x_0)
            #print(annotations[n_episode]["camera_t"])
            
            print(f"Clicked: (u={x}, v={y}) → 3D: {pt3d} → world: {world}")
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click to get 3D point", image)

# ======= Show image and register mouse =======
cv2.imshow("Click to get 3D point", image)
cv2.setMouseCallback("Click to get 3D point", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
