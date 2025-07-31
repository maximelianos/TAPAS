import sapien.core as sapien
from pathlib import Path
import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# Look-at Pose Helper (points camera from eye to target)
# ------------------------------------------------------------------
"""
def look_at(eye, target, up=np.array([0, 0, 1])):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f)

    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)

    rot = np.stack([r, u, f], axis=1)
    w = np.sqrt(1.0 + np.trace(rot)) / 2.0
    x = (rot[2, 1] - rot[1, 2]) / (4 * w)
    y = (rot[0, 2] - rot[2, 0]) / (4 * w)
    z = (rot[1, 0] - rot[0, 1]) / (4 * w)
    return sapien.Pose(p=eye, q=[w, x, y, z])
    """

def look_at(eye, target, up=np.array([0, 0, 1], dtype=np.float32)):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)

    rot = np.stack([right, true_up, forward], axis=1)  # 3x3 rotation matrix
    # Convert rotation matrix to quaternion
    from scipy.spatial.transform import Rotation as R
    quat = R.from_matrix(rot).as_quat()  # [x, y, z, w] (scipy format)

    # SAPIEN expects [w, x, y, z]
    quat_sapien = [quat[3], quat[0], quat[1], quat[2]]
    return sapien.Pose(p=eye, q=quat_sapien)


# ------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------
object_dir = Path("objects/3971")
urdf_file = object_dir / "mobility.urdf"
out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 2. Engine, Renderer, Scene
# ------------------------------------------------------------------
engine = sapien.Engine()

# For SAPIEN 2.x
from sapien.core import VulkanRenderer
renderer = VulkanRenderer(offscreen_only=True)
engine.set_renderer(renderer)

scene = engine.create_scene()
scene.set_timestep(1 / 240)

# ------------------------------------------------------------------
# 3. Lighting
# ------------------------------------------------------------------
scene.set_ambient_light([1.0, 1.0, 1.0])                     # bright ambient
scene.add_directional_light([1, -1, -1], [1, 1, 1])         # key light

# ------------------------------------------------------------------
# 4. Camera
# ------------------------------------------------------------------
W, H = 640, 480
cam = scene.add_camera("cam", width=W, height=H, fovy=1.0, near=0.1, far=10)

eye    = np.array([0.6, 0.6, 0.2])      # from front-right-top
target = np.array([0.0, 0.0, 0.0])      # look at object centre
cam.set_pose(look_at(eye, target, up=[0, 0, 1]))

# Point light colocated with the camera for even illumination
scene.add_point_light(
    position=np.array(eye, dtype=np.float32),
    color=np.array([300, 300, 300], dtype=np.float32),
    shadow=False
)


# ------------------------------------------------------------------
# 5. Load Object
# ------------------------------------------------------------------
loader = scene.create_urdf_loader()
loader.fix_root_link = True
art = loader.load(str(urdf_file))

print("Links with visuals:")
for link in art.get_links():
    for vb in link.get_visual_bodies():
        for shape in vb.get_render_shapes():
            mesh = shape.mesh
            print("  ↪ mesh loaded?", mesh is not None, "type:", type(mesh))

# Ensure root and all links sit at the origin
art.set_pose(sapien.Pose([0, 0, 0]))

# ------------------------------------------------------------------
# 6. Render and Save
# ------------------------------------------------------------------
scene.step()
scene.update_render()
cam.take_picture()  # populate textures

# --- RGB ---
rgba = cam.get_float_texture("Color")
rgb  = (rgba[..., :3] * 255).astype(np.uint8)
Image.fromarray(rgb).save(out_dir / "rgb.png")

# --- Depth ---
pos      = cam.get_float_texture("Position")
depth    = (-pos[..., 2]).clip(min=0.0)             # camera looks along –Z
d_max    = depth.max()
if d_max < 1e-6:
    print("⚠️  Depth appears to be all zeros – is the object in view?")
depth_u16 = (depth / (d_max + 1e-6) * 65535).astype(np.uint16)
Image.fromarray(depth_u16).save(out_dir / "depth.png")

# --- Debug Info ---
print("RGB min/max :", rgb.min(),  rgb.max())
print("Depth min/max:", depth.min(), depth.max())
print("✓  Saved", (out_dir / 'rgb.png').resolve(), "and depth.png")

