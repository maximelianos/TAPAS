from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import sapien.core as sapien
import trimesh
import trimesh.sample
from sapien.core import Pose
from scipy.spatial.distance import cdist
from transforms3d.euler import euler2quat

from mani_skill2 import format_path
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.common import np_random, random_choice
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    hex2rgba,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
)
from mani_skill2.utils.trimesh_utils import get_actor_mesh


class LiftBucketBaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = {"panda": Panda}
    agent: Panda

    def __init__(
        self,
        *args,
        robot="panda",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_uid = robot
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

    def _configure_agent(self):
        agent_cls = self.SUPPORTED_ROBOTS[self.robot_uid]
        self._agent_cfg = agent_cls.get_default_config()

    def _load_agent(self):
        agent_cls = self.SUPPORTED_ROBOTS[self.robot_uid]
        self.agent = agent_cls(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent.config.ee_link_name
        )
        set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)

    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            qpos = np.array([0, -0.785, 0, -2.356, 0, 1.57, 0.785, 0, 0])
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.56, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs

    def _register_cameras(self):
        pose = look_at([-0.4, 0, 0.3], [0, 0, 0.1])
        return CameraConfig(
            "base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

    def _register_render_cameras(self):
        pose = look_at([0.5, 0.5, 1.0], [0.0, 0.0, 0.5])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)


@register_env("LiftBucket-v0", max_episode_steps=200)
class LiftBucketEnv(LiftBucketBaseEnv):

    def __init__(
        self,
        asset_root: str = "{ASSET_DIR}/partnet_mobility/dataset",
        model_json: str = "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_bucket_train.json",
        model_ids: List[str] = (),
        lift_height: float = 0.15,
        **kwargs,
    ):
        self.asset_root = Path(format_path(asset_root))
        self.lift_height = lift_height

        model_json = format_path(model_json)
        self.model_db: Dict[str, Dict] = load_json(model_json)

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json

        self.model_ids = model_ids
        self.model_id = None
        self.model_scale = None

        # Find and check urdf paths
        self.model_urdf_paths = {}
        for model_id in self.model_ids:
            self.model_urdf_paths[model_id] = self.find_urdf_path(model_id)

        super().__init__(**kwargs)

    def find_urdf_path(self, model_id):
        model_dir = self.asset_root / str(model_id)

        urdf_names = ["mobility.urdf"]
        for urdf_name in urdf_names:
            urdf_path = model_dir / urdf_name
            if urdf_path.exists():
                return urdf_path

        raise FileNotFoundError(
            f"No valid URDF is found for {model_id}."
            "Please download Partnet-Mobility (ManiSkill2):"
            "`python -m mani_skill2.utils.download_asset partnet_mobility_bucket`."
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        self.set_episode_rng(seed)
        model_id = options.pop("model_id", None)
        model_scale = options.pop("model_scale", None)
        reconfigure = options.pop("reconfigure", False)

        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure
        return super().reset(seed=self._episode_seed, options=options)

    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        # Model ID
        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            reconfigure = True
        self.model_id = model_id
        self.model_info = self.model_db[self.model_id]

        print("model:", model_id)

        # Scale
        if model_scale is None:
            model_scale = self.model_info.get("scale")
        if model_scale is None:
            # Default scale if not provided
            model_scale = 0.3
        if model_scale != self.model_scale:
            reconfigure = True
        self.model_scale = model_scale

        # Handle offset (bucket models typically don't have bbox)
        if "offset" in self.model_info:
            self.model_offset = np.float32(self.model_info["offset"])
        elif "bbox" in self.model_info:
            bbox = self.model_info["bbox"]
            self.model_offset = -np.float32(bbox["min"]) * model_scale
        else:
            # Default offset for bucket models
            self.model_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Add a small clearance
        self.model_offset[2] += 0.01

        return reconfigure

    def _load_actors(self):
        super()._load_actors()
        
        # Create cube
        self.cube = self._create_cube()
        
        # Create sphere  
        self.sphere = self._create_sphere()

    def _load_articulations(self):
        self.bucket = self._load_bucket()
        # Cache qpos to restore
        self._bucket_init_qpos = self.bucket.get_qpos()

        # Set low friction and damping for all joints to allow movement
        for joint in self.bucket.get_active_joints():
            joint.set_friction(1e6)  # Very low friction for easy movement
            joint.set_drive_property(1e6, 1e6)  # No stiffness, low damping
        
        self._set_handle_links()

    def _create_cube(self):
        builder = self._scene.create_actor_builder()
        
        # Add collision shape
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        
        # Add visual shape
        mat = self._renderer.create_material()
        mat.set_base_color([0.8, 0.2, 0.2, 1.0])  # Red color
        mat.metallic = 0.0
        mat.roughness = 0.3
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], material=mat)
        
        cube = builder.build(name="cube")
        cube.set_damping(0.1, 0.1)
        return cube

    def _create_sphere(self):
        builder = self._scene.create_actor_builder()
        
        # Add collision shape
        builder.add_sphere_collision(radius=0.02)
        
        # Add visual shape
        mat = self._renderer.create_material()
        mat.set_base_color([0.2, 0.8, 0.2, 1.0])  # Green color
        mat.metallic = 0.0
        mat.roughness = 0.3
        builder.add_sphere_visual(radius=0.02, material=mat)
        
        sphere = builder.build(name="sphere")
        sphere.set_damping(0.1, 0.1)
        return sphere

    def _load_bucket(self):
        loader = self._scene.create_urdf_loader()
        loader.scale = self.model_scale
        # KEY FIX: Set fix_root_link = False to make bucket movable
        loader.fix_root_link = False

        model_dir = self.asset_root / str(self.model_id)
        urdf_path = model_dir / "mobility.urdf"
        loader.load_multiple_collisions_from_file = True

        density = self.model_info.get("density", 1e2)  # Reduced density for easier lifting
        articulation = loader.load(str(urdf_path), config={"density": density})
        articulation.set_name("bucket")

        set_articulation_render_material(
            articulation, color=hex2rgba("#AAAAAA"), metallic=1, roughness=0.4
        )

        return articulation

    def _set_handle_links(self):
        handle_link_names = []
        
        # Try to get handle links from semantics if available
        if "semantics" in self.model_info:
            for semantic in self.model_info["semantics"]:
                if semantic[2] == "handle":
                    handle_link_names.append(semantic[0])
        
        # If no semantic info or no handles found, use all movable links as potential handles
        if len(handle_link_names) == 0:
            all_links = self.bucket.get_links()
            # Find links that are not the base link (typically the first link)
            for link in all_links[1:]:  # Skip base link
                if not link.get_name().startswith("base"):
                    handle_link_names.append(link.get_name())
        
        # If still no handles, use the bucket itself (base link) for lifting
        if len(handle_link_names) == 0:
            all_links = self.bucket.get_links()
            if len(all_links) > 0:
                handle_link_names.append(all_links[0].get_name())
        
        if len(handle_link_names) == 0:
            raise RuntimeError(f"No suitable links found for bucket {self.model_id}")
            
        self.handle_link_names = handle_link_names

        self.handle_links = []
        self.handle_links_mesh: List[trimesh.Trimesh] = []
        all_links = self.bucket.get_links()
        for name in self.handle_link_names:
            link = get_entity_by_name(all_links, name)
            self.handle_links.append(link)

            # cache mesh
            link_mesh = get_actor_mesh(link, False)
            self.handle_links_mesh.append(link_mesh)

    def _load_agent(self):
        super()._load_agent()

        links = self.agent.robot.get_links()
        self.lfinger = get_entity_by_name(links, "panda_leftfinger")
        self.rfinger = get_entity_by_name(links, "panda_rightfinger")
        self.lfinger_mesh = get_actor_mesh(self.lfinger, False)
        self.rfinger_mesh = get_actor_mesh(self.rfinger, False)

    def _initialize_actors(self):
        # Initialize cube at predefined position
        cube_pos = np.array([0.1, 0.1, 0.05])
        self.cube.set_pose(Pose(cube_pos))
        
        # Initialize sphere at predefined position
        sphere_pos = np.array([-0.1, 0.1, 0.05])
        self.sphere.set_pose(Pose(sphere_pos))

    def _initialize_articulations(self):
        """ Initial bucket pose """
        p = np.zeros(3)
        p[:2] = self._episode_rng.uniform(-0.05, 0.05, [2])
        p[2] = self.model_offset[2] + 0.3
        ori = np.pi/2
        q = euler2quat(0, 0, ori)
        self.bucket.set_pose(Pose(p, q))

    def _initialize_task(self):
        self._set_target_handle()
        
        # Store initial bucket position for lift height calculation
        self.init_bucket_pos = self.bucket.pose.p.copy()
        
        qpos = self._bucket_init_qpos.copy()
        self.bucket.set_qpos(qpos)

        # For dense reward - cache finger point clouds
        self.lfinger_pcd = trimesh.sample.sample_surface(
            self.lfinger_mesh, 256, seed=self._episode_seed
        )[0]
        self.rfinger_pcd = trimesh.sample.sample_surface(
            self.rfinger_mesh, 256, seed=self._episode_seed
        )[0]

        self.last_bucket_height = self.bucket.pose.p[2]

    def _set_target_handle(self):
        n_handle_links = len(self.handle_link_names)
        idx = random_choice(np.arange(n_handle_links), self._episode_rng)

        self.handle_link_name = self.handle_link_names[idx]
        self.handle_link: sapien.Link = self.handle_links[idx]
        self.handle_link_mesh: trimesh.Trimesh = self.handle_links_mesh[idx]

        # NOTE(jigu): trimesh uses np.random to sample
        self.handle_link_pcd = trimesh.sample.sample_surface(
            self.handle_link_mesh, 256, seed=self._episode_seed
        )[0]

        # Use center of mass for handle position
        cmass_pose = self.handle_link.pose * self.handle_link.cmass_local_pose
        self.handle_link_pos = cmass_pose.p

    @property
    def bucket_pose(self):
        """Get the center of mass (COM) pose of the bucket."""
        return self.bucket.pose.transform(self.bucket.cmass_local_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            lift_height=np.array(self.lift_height),
            handle_link_pos=self.handle_link_pos,
            bucket_pos=self.bucket.pose.p,
            cube_pos=self.cube.pose.p,
            sphere_pos=self.sphere.pose.p,
        )
        if self._obs_mode in ["state", "state_dict", "state_dict+image"]:
            current_height = self.bucket.pose.p[2] - self.init_bucket_pos[2]
            height_diff = self.lift_height - current_height
            obs["height_diff"] = np.array(height_diff)
            obs["handle_pose"] = vectorize_pose(self.handle_link.pose)
            obs["cube_pose"] = vectorize_pose(self.cube.pose)
            obs["sphere_pose"] = vectorize_pose(self.sphere.pose)
        return obs

    @property
    def current_bucket_height(self):
        return self.bucket.pose.p[2] - self.init_bucket_pos[2]

    def evaluate(self, **kwargs):
        current_height = self.current_bucket_height
        height_diff = self.lift_height - current_height
        success = height_diff < 0.02  # 2cm tolerance
        return dict(success=success, height_diff=height_diff, current_height=current_height)

    def _compute_distance(self):
        """Compute the distance between the handle and robot fingers."""
        T = self.handle_link.pose.to_transformation_matrix()
        pcd = transform_points(T, self.handle_link_pcd)
        T1 = self.lfinger.pose.to_transformation_matrix()
        T2 = self.rfinger.pose.to_transformation_matrix()
        pcd1 = transform_points(T1, self.lfinger_pcd)
        pcd2 = transform_points(T2, self.rfinger_pcd)
        distance1 = cdist(pcd, pcd1)
        distance2 = cdist(pcd, pcd2)

        return min(distance1.min(), distance2.min())

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            return 10.0

        # Reward for getting close to the handle
        distance = self._compute_distance()
        reward += 1 - np.tanh(distance * 5.0)

        # Reward for lifting the bucket
        current_height = self.current_bucket_height
        lift_reward = 3 * min(current_height / self.lift_height, 1.0)
        reward += lift_reward

        # Reward for upward motion
        delta_height = current_height - (self.last_bucket_height - self.init_bucket_pos[2])
        if delta_height > 0:
            motion_reward = np.tanh(delta_height * 10) * 2
        else:
            motion_reward = -np.tanh(-delta_height * 10) * 0.5
        reward += motion_reward
        self.last_bucket_height = self.bucket.pose.p[2]

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 10.0

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.lift_height, self.init_bucket_pos])

    def set_state(self, state):
        self.lift_height = state[-4]
        self.init_bucket_pos = state[-3:]
        super().set_state(state[:-4])
        self.last_bucket_height = self.bucket.pose.p[2]