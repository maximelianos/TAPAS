from collections import OrderedDict
from copy import copy
from loguru import logger
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from math import pi
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2.envs.sapien_env import Action, ActionType
from mani_skill2 import format_path
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import check_actor_static, vectorize_pose, set_actor_visibility

from .base_env import StationaryManipulationEnv
from .pick_single import PickSingleYCBEnv, build_actor_ycb, random_choice
from .stack_cube import UniformSampler


class MultiObjectYCB(StationaryManipulationEnv):
    objs: tuple[sapien.Actor, ...]

    DEFAULT_ASSET_ROOT = PickSingleYCBEnv.DEFAULT_ASSET_ROOT
    DEFAULT_MODEL_JSON = PickSingleYCBEnv.DEFAULT_MODEL_JSON

    def __init__(
        self,
        asset_root: str = None,
        model_json: str = None,
        model_ids: Tuple[Tuple[str]] = (("014_lemon",), ("024_bowl",)),
        obj_init_rot_z=True,
        obj_init_rot=(0, 0),
        **kwargs,
    ):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))

        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        model_json = self.asset_root / format_path(model_json)

        if not model_json.exists():
            raise FileNotFoundError(
                f"{model_json} is not found."
                "Please download the corresponding assets:"
                "`python -m mani_skill2.utils.download_asset ${ENV_ID}`."
            )

        self.model_db: Dict[str, Dict] = load_json(model_json)

        for m in model_ids:
            assert len(m) > 0, model_json
        self.model_ids = model_ids

        self.model_id_per_obj = [m[0] for m in model_ids]

        self.model_scales = [None for _ in range(len(model_ids))]
        self.model_bbox_size = [None for _ in range(len(model_ids))]

        self.obj_init_rot_z = obj_init_rot_z
        self.obj_init_rot = obj_init_rot

        self._check_assets()
        super().__init__(**kwargs)

    @property
    def _get_n_objs(self):
        return len(self.model_ids)

    def _check_assets(self):
        models_dir = self.asset_root / "models"
        for model_id in tuple([id for tup in self.model_ids for id in tup]):
            model_dir = models_dir / model_id
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"{model_dir} is not found."
                    "Please download (ManiSkill2) YCB models:"
                    "`python -m mani_skill2.utils.download_asset ycb`."
                )

            collision_file = model_dir / "collision.obj"
            if not collision_file.exists():
                raise FileNotFoundError(
                    "convex.obj has been renamed to collision.obj. "
                    "Please re-download YCB models."
                )

    # NOTE: not sure if enable_pcm is necessary. They do it in stack_cube and
    # in the assmebly envs. Maybe it's for collision detection?
    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.enable_pcm = True
        return scene_config

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        self._load_objects()

        for o in self.objs:
            o.set_damping(0.1, 0.1)

    def _load_objects(self):
        objs =[]
        for i, s in zip(self.model_id_per_obj, self.model_scales):
            density = self.model_db[i].get("density", 1000)
            obj = build_actor_ycb(
                i,
                self._scene,
                scale=s,
                density=density,
                root_dir=self.asset_root,
            )
            obj.name = i

            objs.append(obj)

        self.objs = tuple(objs)

    def reset(self, seed=None, options=None, model_ids=None,
              model_scales=None):
        if options is None:
            options = dict()
        reconfigure = options.get("reconfigure", False)
        self.set_episode_rng(seed)
        _reconfigure = self._set_models(model_ids, model_scales)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure
        return super().reset(seed=self._episode_seed, options=options)

    def _set_models(self, model_ids, model_scales):
        """Set the model ids and scale. If not provided, choose randomly."""
        reconfigure = False

        if model_ids is None:
            model_ids = (None for _ in range(len(self.model_ids)))
        if model_scales is None:
            model_scales = (None for _ in range(len(self.model_ids)))

        for j, (model_id, model_scale) in enumerate(zip(model_ids, model_scales)):
            if model_id is None:
                model_id = random_choice(self.model_ids[j], self._episode_rng)
            if model_id != self.model_id_per_obj[j]:
                self.model_id_per_obj[j] = model_id
                reconfigure = True

            if model_scale is None:
                obj_model_scales = self.model_db[self.model_id_per_obj[j]].get(
                    "scales")
                if obj_model_scales is None:
                    model_scale = 1.0
                else:
                    model_scale = random_choice(obj_model_scales,
                                                self._episode_rng)
            if model_scale != self.model_scales[j]:
                self.model_scales[j] = model_scale
                reconfigure = True

            model_info = self.model_db[self.model_id_per_obj[j]]
            if "bbox" in model_info:
                bbox = model_info["bbox"]
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                self.model_bbox_size[j] = bbox_size * model_scale
            else:
                self.model_bbox_size[j] = None

        return reconfigure

    def _get_init_z(self, obj_idx):
        bbox_min = self.model_db[self.model_id_per_obj[obj_idx]]["bbox"]["min"]
        return -bbox_min[2] * self.model_scales[obj_idx] + 0.05

    def _settle(self, t):
        return PickSingleYCBEnv._settle(self, t)

    def _initialize_actors(self):
        self._random_object_placement(self.objs)

    def _random_object_placement(self, objs):
        for j, obj in enumerate(objs):
            # The object will fall from a certain height
            xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            z = self._get_init_z(j)
            p = np.hstack([xy, z])
            q = [1, 0, 0, 0]

            # Rotate along z-axis
            if self.obj_init_rot_z:
                ori = self._episode_rng.uniform(0, 2 * np.pi)
                q = euler2quat(0, 0, ori)

            # Rotate along a random axis by a small angle
            if self.obj_init_rot[j] > 0:
                axis = self._episode_rng.uniform(-1, 1, 3)
                axis = axis / max(np.linalg.norm(axis), 1e-6)
                ori = self._episode_rng.uniform(0, self.obj_init_rot)
                q = qmult(q, axangle2quat(axis, ori, True))
            obj.set_pose(Pose(p, q))

            # Move the robot far away to avoid collision
            # The robot should be initialized later
            self.agent.robot.set_pose(Pose([-10, 0, 0]))

            # Lock rotation around x and y
            obj.lock_motion(0, 0, 0, 1, 1, 0)
            self._settle(0.5)

            # Unlock motion
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
            self._settle(0.5)

            # Some objects need longer time to settle
            lin_vel = np.linalg.norm(obj.velocity)
            ang_vel = np.linalg.norm(obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                self._settle(0.5)

    @property
    def obj_poses(self):
        return [obj.pose.transform(obj.cmass_local_pose) for obj in self.objs]

    def _get_obs_extra(self):
       obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
       )
       pose_dict = {
           f"obj{i+1}_pose": vectorize_pose(self.obj_poses[i])
           for i in range(len(self.objs))
       }
       if self._obs_mode in ["state", "state_dict", "state_dict+image"]:
            obs.update(**pose_dict)
       return obs

@register_env("AToB-v0", max_episode_steps=1000)
class AToBEnv(MultiObjectYCB):
    def __init__(
        self,
        *args,
        goal_tresh=0.05,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.goal_tresh = goal_tresh

    def _initialize_actors(self):
        self._radial_obj_placement(self.objs)

    def _radial_obj_placement(self, objs, iterative=False):
        x_prev, y_prev = 0.0, 0.0
        angle_min, angle_max = 1/4 * np.pi, 3/4 * np.pi

        for j, obj in enumerate(objs):
            # The object will fall from a certain height
            angle = self._episode_rng.uniform(angle_min, angle_max)
            radius = self._episode_rng.uniform(0.15, 0.20)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            x += x_prev
            y += y_prev
            z = self._get_init_z(j)
            p = np.hstack([x, y, z])
            q = [1, 0, 0, 0]

            if iterative:
                x_prev = x
                y_prev = y
            else:
                angle_min += np.pi / 2
                angle_max += np.pi / 2

            # Rotate along z-axis
            if self.obj_init_rot_z:
                ori = self._episode_rng.uniform(0, 1 / 2 * np.pi)
                q = euler2quat(0, 0, ori)

            # Rotate along a random axis by a small angle
            if self.obj_init_rot[j] > 0:
                axis = self._episode_rng.uniform(-1, 1, 3)
                axis = axis / max(np.linalg.norm(axis), 1e-6)
                ori = self._episode_rng.uniform(0, self.obj_init_rot)
                q = qmult(q, axangle2quat(axis, ori, True))
            obj.set_pose(Pose(p, q))

            # Move the robot far away to avoid collision
            # The robot should be initialized later
            self.agent.robot.set_pose(Pose([-10, 0, 0]))

            # Lock rotation around x and y
            obj.lock_motion(0, 0, 0, 1, 1, 0)
            self._settle(0.5)

            # Unlock motion
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
            self._settle(0.5)

            # Some objects need longer time to settle
            lin_vel = np.linalg.norm(obj.velocity)
            ang_vel = np.linalg.norm(obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                self._settle(0.5)

    def get_solution_sequence(self):
        goal_a2w = copy(self.objs[0].pose)
        goal_b2w = copy(self.objs[1].pose)

        root2w = self.agent.robot.get_root_pose()
        w2root = root2w.inv()

        root2move_goal_a = w2root.transform(goal_a2w)
        root2move_goal_b = w2root.transform(goal_b2w)

        a_quat = root2move_goal_a.q
        a_euler = quat2euler(a_quat)
        a_angle_z = a_euler[2] + 1/4 * pi  # (a_euler[2] + 3/4 * pi) % pi - 1/2 * pi
        a_euler = (-pi, 0, a_angle_z)  # rotate 180 degrees around x axis
        a_rot = euler2quat(*a_euler)

        b_quat = root2move_goal_b.q
        b_euler = quat2euler(b_quat)
        b_rot = b_euler[2]
        b_euler = (-pi, 0, b_rot)
        b_rot = euler2quat(*b_euler)

        z_offset = np.array([0, 0, self.model_bbox_size[1][2]])

        # Transform to np.ndarray
        move_goal_above_a = np.concatenate(
            [root2move_goal_a.p + z_offset * 2, a_rot])
        move_goal_a = np.concatenate([root2move_goal_a.p, a_rot])
        move_goal_above_b = np.concatenate(
            [root2move_goal_b.p + z_offset * 2, b_rot])
        move_goal_on_b = np.concatenate(
            [root2move_goal_b.p + z_offset, b_rot])

        seq = [
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.MOVE_TO, goal=move_goal_a),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.CLOSE_GRIPPER),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.MOVE_TO, goal=move_goal_above_b),
            Action(ActionType.MOVE_TO, goal=move_goal_on_b),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.OPEN_GRIPPER),
        ]

        return seq

    def _check_objA_on_objB(self):
        pos_A = self.objs[0].pose.p
        pos_B = self.objs[1].pose.p
        offset = pos_A - pos_B
        is_obj_placed = np.linalg.norm(offset) <= self.goal_tresh

        return is_obj_placed

    def evaluate(self, **kwargs):
        is_obj_placed = self._check_objA_on_objB()
        # sometimes the static check gives false negatives, so ignore for now
        is_objA_static = check_actor_static(self.objs[0])
        is_objA_grasped = self.agent.check_grasp(self.objs[0])
        success = is_obj_placed and is_objA_static and (not is_objA_grasped)

        return {
            "is_obj_placed": is_obj_placed,
            "is_objA_static": is_objA_static,
            "is_objA_grasped": is_objA_grasped,
            # "cubeA_vel": np.linalg.norm(self.cubeA.velocity),
            # "cubeA_ang_vel": np.linalg.norm(self.cubeA.angular_velocity),
            "success": success,
        }

    def compute_dense_reward(self, info, **kwargs):
        logger.warning("Did not implement dense reward.")
        return 0


@register_env("AToBCluttered-v0", max_episode_steps=1000)
class AToBClutteredEnv(AToBEnv):
    def __init__(
        self, *args,
        model_ids: Tuple[Tuple[str]] = (("014_lemon",), ("024_bowl",)),
        clutter_model_ids: Tuple[Tuple[str]] = (
            ("012_strawberry",), ("077_rubiks_cube",), ("007_tuna_fish_can",)
        ),
        n_placement_attempts=10,
        **kwargs,
    ):
        self.clutter_start_idx = len(model_ids)
        self.n_placement_attempts = n_placement_attempts

        model_ids = tuple(list(model_ids) + list(clutter_model_ids))

        super().__init__(*args, model_ids=model_ids, **kwargs)

    def _random_clutter_placement(self, objs, obj_idx_offset,
                                  avoid_collision=True,
                                  dist_scale=1.1, dist_const=0.05):
        for j, obj in enumerate(objs):
            success = False
            for _ in range(self.n_placement_attempts):
                idx = j + obj_idx_offset
                x = self._episode_rng.uniform(-0.4, 0.4, [1])
                y = self._episode_rng.uniform(-0.6, 0.6, [1])

                obj_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
                z =  self._get_init_z(idx) - 0.05
                obj_pose = sapien.Pose([x, y, z], obj_quat)

                collision = self._check_placement_collision(obj_pose, idx,
                                                            dist_scale,
                                                            dist_const)

                if not avoid_collision or not collision:
                    success = True
            if not success:
                logger.warning(f"Failed to place object {obj} in "
                               f"{self.n_placement_attempts} attempts.")

            obj.set_pose(obj_pose)

    def _check_placement_collision(self, pose, obj_idx, dist_scale, dist_const):
        for i in range(obj_idx):
            # print(i, obj_idx, self.objs[i].name, self.objs[obj_idx].name)
            other_pose = self.objs[i].pose
            offset = pose.p - other_pose.p
            bbox_size = self.model_bbox_size[i] + self.model_bbox_size[obj_idx]
            # print(np.abs(offset), bbox_size * dist_scale + dist_const)
            collision = np.all(np.abs(offset) < (
                                bbox_size * dist_scale + dist_const))

            if collision:
                return True

        return False

    def _initialize_actors(self):
        self._radial_obj_placement(self.objs[:self.clutter_start_idx])

        self._random_clutter_placement(self.objs[self.clutter_start_idx:],
                                       obj_idx_offset=self.clutter_start_idx)