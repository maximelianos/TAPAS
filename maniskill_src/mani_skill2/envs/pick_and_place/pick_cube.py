from collections import OrderedDict
from copy import copy

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from math import pi
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2.envs.sapien_env import Action, ActionType

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("PickCube-v0", max_episode_steps=200)
class PickCubeEnv(StationaryManipulationEnv):
    goal_thresh = 0.025
    min_goal_dist = 0.05

    def __init__(self, *args, obj_init_rot_z=True, **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.cube_half_size = np.array([0.02] * 3, np.float32)
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.obj = self._build_cube(self.cube_half_size)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        xyz = np.hstack([xy, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)
        self.obj.set_pose(Pose(xyz, q))

    def _initialize_task(self, max_trials=100, verbose=False):
        obj_pos = self.obj.pose.p

        # Sample a goal position far enough from the object
        for i in range(max_trials):
            goal_xy = self._episode_rng.uniform(-0.1, 0.1, [2])
            goal_z = self._episode_rng.uniform(0, 0.5) + obj_pos[2]
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - obj_pos) > self.min_goal_dist:
                if verbose:
                    print(f"Found a valid goal at {i}-th trial")
                break

        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict", "state_dict+image"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
                obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
            )
        return obs

    def check_obj_placed(self):
        return np.linalg.norm(self.goal_pos - self.obj.pose.p) <= self.goal_thresh

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        is_robot_static = self.check_robot_static()
        return dict(
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed and is_robot_static,
        )

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 5
            return reward

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        reward += 1 if is_grasped else 0.0

        if is_grasped:
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
            reward += place_reward

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 5.0

    def render_human(self):
        self.goal_site.unhide_visual()
        ret = super().render_human()
        self.goal_site.hide_visual()
        return ret

    def render_rgb_array(self):
        self.goal_site.unhide_visual()
        ret = super().render_rgb_array()
        self.goal_site.hide_visual()
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


@register_env("LiftCube-v0", max_episode_steps=200)
class LiftCubeEnv(PickCubeEnv):
    """Lift the cube to a certain height."""

    goal_height = 0.2

    def _initialize_task(self):
        self.goal_pos = self.obj.pose.p + [0, 0, self.goal_height]
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict", "state_dict+image"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
            )
        return obs

    def check_obj_placed(self):
        return self.obj.pose.p[2] >= self.goal_height + self.cube_half_size[2]

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            reward += 2.25
            return reward

        # reaching reward
        gripper_pos = self.tcp.get_pose().p
        obj_pos = self.obj.get_pose().p
        dist = np.linalg.norm(gripper_pos - obj_pos)
        reaching_reward = 1 - np.tanh(5 * dist)
        reward += reaching_reward

        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)

        # grasp reward
        if is_grasped:
            reward += 0.25

        # lifting reward
        if is_grasped:
            lifting_reward = self.obj.pose.p[2] - self.cube_half_size[2]
            lifting_reward = min(lifting_reward / self.goal_height, 1.0)
            reward += lifting_reward

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 2.25
    
@register_env("LiftBlock-v0", max_episode_steps=200)
class LiftBlockEnv(LiftCubeEnv):
    """Lift the cube to a certain height. Simplified block rotation."""

    def __init__(self, *args, obj_init_rot_z=True, **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.cube_half_size = np.array([0.03] * 3, np.float32)
        StationaryManipulationEnv.__init__(self, *args, **kwargs)


    def _initialize_actors(self):
        # angle_min, angle_max = -1/4 * np.pi, 1/4 * np.pi
        # angle = self._episode_rng.uniform(angle_min, angle_max)
        # radius = self._episode_rng.uniform(0.14, 0.16)
        # x = radius * np.cos(angle)
        # y = radius * np.sin(angle)
        # xyz = np.hstack([x, y, self.cube_half_size[2]])
        xy = self._episode_rng.uniform(0.05, 0.1, [2])
        xyz = np.hstack([xy, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(np.pi/4, np.pi/2)
            q = euler2quat(0, 0, ori)
        self.obj.set_pose(Pose(xyz, q))

    def get_solution_sequence(self):
        goal_a2w = copy(self.obj.pose)

        root2w = self.agent.robot.get_root_pose()
        w2root = root2w.inv()

        root2move_goal_a = w2root.transform(goal_a2w)

        a_quat = root2move_goal_a.q
        a_euler = quat2euler(a_quat)
        a_angle_z = a_euler[2]
        a_euler = (-pi, 0, a_angle_z)
        a_rot = euler2quat(*a_euler)

        z_offset = np.array([0, 0, 0.1])
        lift_offset = np.array([0, 0, 0.3])

        # Transform to np.ndarray
        move_goal_above_a = np.concatenate(
            [root2move_goal_a.p + z_offset, a_rot]
        )
        move_goal_at_a = np.concatenate(
            [root2move_goal_a.p, a_rot]
        )
        move_goal_b = np.concatenate(
            [root2move_goal_a.p + lift_offset, a_rot])

        seq = [
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.MOVE_TO, goal=move_goal_at_a),
            # Action(ActionType.NOOP, goal=10),
            Action(ActionType.CLOSE_GRIPPER),
            # Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=move_goal_b),
            Action(ActionType.NOOP, goal=30),
        ]

        return seq

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        is_robot_static = self.check_robot_static()
        return dict(
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed,
        )

@register_env("PushCube-v0", max_episode_steps=500)
class PushCubeEnv(PickCubeEnv):
    "Push the cube to the goal position."
    goal_thresh = 0.05
    clutter_size = np.array([0.1, 0.1, 0.02])

    def _load_actors(self):
        super()._load_actors()
        self.clutter = self._build_cube(
            self.clutter_size,
            name="clutter",
            color=(0, 0, 1),
            )

    def _initialize_actors(self):
        # Y: left-right, X: front-back
        x = self._episode_rng.uniform(-0.3, -0.1)
        y = self._episode_rng.uniform(-0.2, -0.1)
        xyz = np.hstack([x, y, self.cube_half_size[2]])
        q = [1, 0, 0, 0]
        self.obj.set_pose(Pose(xyz, q))
        xyz_clutter = xyz + np.array([0, 0.2, 0])
        self.clutter.set_pose(Pose(xyz_clutter, q))

    def _initialize_task(self):
        # TODO: randomize goal_dist_x, too? Or even only x?
        self.goal_dist_y = self._episode_rng.uniform(0.18, 0.22)
        self.goal_pos = self.obj.pose.p + [0.2, self.goal_dist_y, 0]
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict", "state_dict+image"]:
            obs.update(
                goal_pose=vectorize_pose(self.goal_site.pose),
                obj_pose=vectorize_pose(self.obj.pose),
            )
        return obs

    def get_solution_sequence(self):
        goal_a2w = copy(self.obj.pose)
        goal_b2w = copy(self.goal_site.pose)

        root2w = self.agent.robot.get_root_pose()
        w2root = root2w.inv()

        root2move_goal_a = w2root.transform(goal_a2w)
        root2move_goal_b = w2root.transform(goal_b2w)

        a_quat = root2move_goal_a.q
        a_euler = quat2euler(a_quat)
        a_angle_z = a_euler[2]
        a_euler = (-pi, 0, a_angle_z)  # rotate 180 degrees around x axis
        mid_euler = (-pi, 0, a_angle_z + pi/2)
        a_rot = euler2quat(*a_euler)
        mid_rot = euler2quat(*mid_euler)

        b_quat = root2move_goal_b.q
        b_euler = quat2euler(b_quat)
        b_rot = b_euler[2]
        b_euler = (-pi, 0, b_rot)
        b_rot = euler2quat(*b_euler)

        ab_diff = root2move_goal_b.p - root2move_goal_a.p
        ab_dist = np.linalg.norm(ab_diff)
        z_offset = np.array([0, 0, 4 * self.cube_half_size[2]])
        
        x_offset = np.array([-0.04, 0, 0])
        small_x_offset = np.array([-0.02, 0, 0])
        y_offset = np.array([0, -0.06, 0])
        mid_dist = np.array([0.2, 0, 0])

        # Transform to np.ndarray
        move_goal_above_a = np.concatenate(
            [root2move_goal_a.p + x_offset + z_offset, a_rot]
        )
        move_goal_next_to_a = np.concatenate(
            [root2move_goal_a.p + x_offset, a_rot])
        move_goal_mid = np.concatenate(
            [root2move_goal_a.p + mid_dist + small_x_offset, a_rot])
        movel_goal_backoff = np.concatenate(
            [root2move_goal_a.p + mid_dist + 1.5*x_offset, a_rot]
        )
        move_goal_second_push = np.concatenate(
            [root2move_goal_a.p + mid_dist + y_offset - 0.5 * small_x_offset, mid_rot])
        move_goal_b = np.concatenate(
            [root2move_goal_b.p, mid_rot])

        seq = [
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.CLOSE_GRIPPER),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=move_goal_next_to_a),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=move_goal_mid),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=movel_goal_backoff),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=move_goal_second_push),
            Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=move_goal_b, with_screw=True),
            Action(ActionType.NOOP, goal=30),
        ]

        return seq
    

@register_env("SlideBlock-v0", max_episode_steps=500)
class SlideBlockeEnv(PushCubeEnv):
    "Slide the block to the goal position."
    goal_thresh = 0.05
    block_half_size = np.array([0.04, 0.04, 0.04])
    clutter_size = np.array([0.02, 0.02, 0.02])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.obj = self._build_cube(self.block_half_size, color=(1, 0, 0))
        # self.clutter = self._build_cube(self.clutter_size, color=(0, 0, 1))
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        x = self._episode_rng.uniform(0.05, 0.15)
        y = self._episode_rng.uniform(-0.2, -0.1)
        xyz = np.hstack([x, y, self.block_half_size[2]])
        q = [1, 0, 0, 0]
        self.obj.set_pose(Pose(xyz, q))
        # xyz_clutter = np.array([0, 0, self.clutter_size[2]])
        # self.clutter.set_pose(Pose(xyz_clutter, q))

    def _initialize_task(self):
        self.goal_pos = self.obj.pose.p + [0, 0.2, 0]
        self.goal_site.set_pose(Pose(self.goal_pos))

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        return dict(
            is_obj_placed=is_obj_placed,
            success=is_obj_placed,
        )

    def get_solution_sequence(self):
        goal_a2w = copy(self.obj.pose)
        goal_b2w = copy(self.goal_site.pose)

        root2w = self.agent.robot.get_root_pose()
        w2root = root2w.inv()

        root2move_goal_a = w2root.transform(goal_a2w)
        root2move_goal_b = w2root.transform(goal_b2w)

        a_quat = root2move_goal_a.q
        a_euler = quat2euler(a_quat)
        a_angle_z = a_euler[2]
        a_euler = (-pi, 0, a_angle_z + pi/2)
        a_rot = euler2quat(*a_euler)

        y_offset = np.array([0, -0.1, -0.02])

        # Transform to np.ndarray
        move_goal_behind_a = np.concatenate(
            [root2move_goal_a.p + y_offset, a_rot]
        )
        move_goal_b = np.concatenate(
            [root2move_goal_b.p, a_rot])

        seq = [
            Action(ActionType.CLOSE_GRIPPER),
            Action(ActionType.MOVE_TO, goal=move_goal_behind_a),
            Action(ActionType.NOOP, goal=10),
            # Action(ActionType.CLOSE_GRIPPER),
            # Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=move_goal_b),
            Action(ActionType.NOOP, goal=30),
        ]

        return seq
    

@register_env("LiftBottle-v0", max_episode_steps=500)
class LiftBottleEnv(LiftCubeEnv):
    """Lift the bottle to a certain height."""

    goal_height = 0.2

    def __init__(self, *args, obj_init_rot_z=True, **kwargs):
        self.obj_init_rot_z = obj_init_rot_z
        self.block_half_size = np.array([0.02, 0.02, 0.10] * 3, np.float32)
        StationaryManipulationEnv.__init__(self, *args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self.obj = self._build_cube(self.block_half_size, color=(1, 0, 0))
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _initialize_actors(self):
        rand_x = self._episode_rng.uniform(0., 0.05)
        x = 0.2 + rand_x
        self.rand_x = rand_x
        y = -0.2
        xyz = np.hstack([x, y, self.block_half_size[2]])
        q = [1, 0, 0, 0]
        self.obj.set_pose(Pose(xyz, q))

    def check_obj_placed(self):
        return self.obj.pose.p[2] >= self.goal_height + self.block_half_size[2]

    def get_solution_sequence(self):
        goal_a2w = copy(self.obj.pose)

        root2w = self.agent.robot.get_root_pose()
        w2root = root2w.inv()

        root2move_goal_a = w2root.transform(goal_a2w)

        a_quat = root2move_goal_a.q
        a_euler = quat2euler(a_quat)
        a_angle_z = a_euler[2]
        a_euler = (-pi, -pi/2, a_angle_z)
        a_rot = euler2quat(*a_euler)

        z_offset = np.array([0, 0, 0.1])
        lift_offset = np.array([0, 0, 0.25])

        x_offset = np.array([-0.05, 0, 0])

        # Transform to np.ndarray
        move_goal_above_a = np.concatenate(
            [root2move_goal_a.p + x_offset, a_rot]
        )

        move_goal_at_a = np.concatenate(
            [root2move_goal_a.p, a_rot]
        )
        move_goal_b = np.concatenate(
            [root2move_goal_a.p + lift_offset, a_rot])

        seq = [
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.NOOP, goal=20),
            Action(ActionType.MOVE_TO, goal=move_goal_at_a),
            Action(ActionType.CLOSE_GRIPPER),
            # Action(ActionType.NOOP, goal=10),
            Action(ActionType.MOVE_TO, goal=move_goal_b),
            Action(ActionType.NOOP, goal=30),
        ]

        return seq