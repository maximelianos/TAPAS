from collections import OrderedDict
from copy import copy
from typing import List, Tuple

import numpy as np
import sapien.core as sapien

from math import pi
from transforms3d.euler import euler2quat, quat2euler

from mani_skill2.envs.sapien_env import Action, ActionType

from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import check_actor_static, vectorize_pose

from .base_env import StationaryManipulationEnv


class UniformSampler:
    """Uniform placement sampler.

    Args:
        ranges: ((low1, low2, ...), (high1, high2, ...))
        rng (np.random.RandomState): random generator
    """

    def __init__(
        self, ranges: Tuple[List[float], List[float]], rng: np.random.RandomState
    ) -> None:
        assert len(ranges) == 2 and len(ranges[0]) == len(ranges[1])
        self._ranges = ranges
        self._rng = rng
        self.fixtures = []

    def sample(self, radius, max_trials, append=True, verbose=False):
        """Sample a position.

        Args:
            radius (float): collision radius.
            max_trials (int): maximal trials to sample.
            append (bool, optional): whether to append the new sample to fixtures. Defaults to True.
            verbose (bool, optional): whether to print verbosely. Defaults to False.

        Returns:
            np.ndarray: a sampled position.
        """
        if len(self.fixtures) == 0:
            pos = self._rng.uniform(*self._ranges)
        else:
            fixture_pos = np.array([x[0] for x in self.fixtures])
            fixture_radius = np.array([x[1] for x in self.fixtures])
            for i in range(max_trials):
                pos = self._rng.uniform(*self._ranges)
                dist = np.linalg.norm(pos - fixture_pos, axis=1)
                if np.all(dist > fixture_radius + radius):
                    if verbose:
                        print(f"Found a valid sample at {i}-th trial")
                    break
            else:
                if verbose:
                    print("Fail to find a valid sample!")
        if append:
            self.fixtures.append((pos, radius))
        return pos


@register_env("StackCube-v0", max_episode_steps=1000)
class StackCubeEnv(StationaryManipulationEnv):
    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.enable_pcm = True
        return scene_config

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        self.box_half_size = np.float32([0.02] * 3)
        self.cubeA = self._build_cube(self.box_half_size, color=(1, 0, 0), name="cubeA")
        self.cubeB = self._build_cube(
            self.box_half_size, color=(0, 1, 0), name="cubeB", static=False
        )

    def _initialize_actors(self):
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        region = [[-0.1, -0.2], [0.1, 0.2]]
        sampler = UniformSampler(region, self._episode_rng)
        radius = np.linalg.norm(self.box_half_size[:2]) + 0.001
        cubeA_xy = xy + sampler.sample(radius, 100)
        cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

        cubeA_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        cubeB_quat = euler2quat(0, 0, self._episode_rng.uniform(0, 2 * np.pi))
        z = self.box_half_size[2]
        cubeA_pose = sapien.Pose([cubeA_xy[0], cubeA_xy[1], z], cubeA_quat)
        cubeB_pose = sapien.Pose([cubeB_xy[0], cubeB_xy[1], z], cubeB_quat)

        self.cubeA.set_pose(cubeA_pose)
        self.cubeB.set_pose(cubeB_pose)

    def _get_obs_extra(self):
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict", "state_dict+image"]:
            obs.update(
                cubeA_pose=vectorize_pose(self.cubeA.pose),
                cubeB_pose=vectorize_pose(self.cubeB.pose),
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def get_solution_sequence(self):
        goal_a2w = copy(self.cubeA.pose)
        goal_b2w = copy(self.cubeB.pose)

        # Translate move_goal from world frame to robot root link frame, see
        # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/docs/source/tutorial/motion_planning/plan_a_path.rst#L37
        root2w = self.agent.robot.get_root_pose()
        w2root = root2w.inv()

        root2move_goal_a = w2root.transform(goal_a2w)
        root2move_goal_b = w2root.transform(goal_b2w)

        # get z rotation of cubes to match gripper rotation
        a_quat = root2move_goal_a.q
        a_euler = quat2euler(a_quat)
        # NOTE: get the closest rotation to pick the cube. With the + pi/4, we
        # get the closest rotation to the gripper [-45, 45) degrees. Without it
        # the angle would be [0, 90) degrees and unimodal.
        a_angle_z = (a_euler[2] + pi/4) % (pi/2) - (pi/4)  # mod by 90 degrees
        a_euler = (-pi, 0, a_angle_z)  # rotate 180 degrees around x axis
        a_rot = euler2quat(*a_euler)

        b_quat = root2move_goal_b.q
        b_euler = quat2euler(b_quat)
        b_angle_z = (b_euler[2] + pi/4) % (pi/2) - (pi/4)  # mod by 90 degrees
        b_euler = (-pi, 0, b_angle_z)  # rotate 180 degrees around x axis
        b_rot = euler2quat(*b_euler)

        z_offset = np.array([0, 0, self.box_half_size[2]])

        # Transform to np.ndarray
        move_goal_above_a = np.concatenate(
            [root2move_goal_a.p + z_offset * 2, a_rot])
        move_goal_a = np.concatenate([root2move_goal_a.p, a_rot])
        move_goal_above_b = np.concatenate(
            [root2move_goal_b.p + z_offset * 4, b_rot])
        move_goal_on_b = np.concatenate(
            [root2move_goal_b.p + 2 * z_offset, b_rot])

        seq = [
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.MOVE_TO, goal=move_goal_a),
            Action(ActionType.CLOSE_GRIPPER),
            Action(ActionType.MOVE_TO, goal=move_goal_above_a),
            Action(ActionType.MOVE_TO, goal=move_goal_above_b),
            Action(ActionType.MOVE_TO, goal=move_goal_on_b),
            Action(ActionType.OPEN_GRIPPER),
        ]

        return seq

    def _check_cubeA_on_cubeB(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            np.linalg.norm(offset[:2]) <= np.linalg.norm(self.box_half_size[:2]) + 0.005
        )
        z_flag = np.abs(offset[2] - self.box_half_size[2] * 2) <= 0.005
        return bool(xy_flag and z_flag)

    def evaluate(self, **kwargs):
        is_cubeA_on_cubeB = self._check_cubeA_on_cubeB()
        is_cubeA_static = check_actor_static(self.cubeA)
        is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
        success = is_cubeA_on_cubeB and is_cubeA_static and (not is_cubeA_grasped)

        return {
            "is_cubaA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            # "cubeA_vel": np.linalg.norm(self.cubeA.velocity),
            # "cubeA_ang_vel": np.linalg.norm(self.cubeA.angular_velocity),
            "success": success,
        }

    def compute_dense_reward(self, info, **kwargs):
        gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1] * 2
        )  # NOTE: hard-coded with panda
        reward = 0.0

        if info["success"]:
            reward = 15.0
        else:
            # grasp pose rotation reward
            grasp_rot_loss_fxn = lambda A: np.tanh(
                1 / 8 * np.trace(A.T @ A)
            )  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
            tcp_pose_wrt_cubeA = self.cubeA.pose.inv() * self.tcp.pose
            tcp_rot_wrt_cubeA = tcp_pose_wrt_cubeA.to_transformation_matrix()[:3, :3]
            gt_rots = [
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            ]
            grasp_rot_loss = min(
                [grasp_rot_loss_fxn(x - tcp_rot_wrt_cubeA) for x in gt_rots]
            )
            reward += 1 - grasp_rot_loss

            cubeB_vel_penalty = np.linalg.norm(self.cubeB.velocity) + np.linalg.norm(
                self.cubeB.angular_velocity
            )
            reward -= cubeB_vel_penalty

            # reaching object reward
            tcp_pose = self.tcp.pose.p
            cubeA_pos = self.cubeA.pose.p
            cubeA_to_tcp_dist = np.linalg.norm(tcp_pose - cubeA_pos)
            reaching_reward = 1 - np.tanh(3.0 * cubeA_to_tcp_dist)
            reward += reaching_reward

            # check if cubeA is on cubeB
            cubeA_pos = self.cubeA.pose.p
            cubeB_pos = self.cubeB.pose.p
            goal_xyz = np.hstack(
                [cubeB_pos[0:2], cubeB_pos[2] + self.box_half_size[2] * 2]
            )
            cubeA_on_cubeB = (
                np.linalg.norm(goal_xyz[:2] - cubeA_pos[:2])
                < self.box_half_size[0] * 0.8
            )
            cubeA_on_cubeB = cubeA_on_cubeB and (
                np.abs(goal_xyz[2] - cubeA_pos[2]) <= 0.005
            )
            if cubeA_on_cubeB:
                reward = 10.0
                # ungrasp reward
                is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
                if not is_cubeA_grasped:
                    reward += 2.0
                else:
                    reward = (
                        reward
                        + 2.0 * np.sum(self.agent.robot.get_qpos()[-2:]) / gripper_width
                    )
            else:
                # grasping reward
                is_cubeA_grasped = self.agent.check_grasp(self.cubeA)
                if is_cubeA_grasped:
                    reward += 1.0

                # reaching goal reward, ensuring that cubeA has appropriate height during this process
                if is_cubeA_grasped:
                    cubeA_to_goal = goal_xyz - cubeA_pos
                    # cubeA_to_goal_xy_dist = np.linalg.norm(cubeA_to_goal[:2])
                    cubeA_to_goal_dist = np.linalg.norm(cubeA_to_goal)
                    appropriate_height_penalty = np.maximum(
                        np.maximum(2 * cubeA_to_goal[2], 0.0),
                        np.maximum(2 * (-0.02 - cubeA_to_goal[2]), 0.0),
                    )
                    reaching_reward2 = 2 * (
                        1 - np.tanh(5.0 * appropriate_height_penalty)
                    )
                    # qvel_penalty = np.sum(np.abs(self.agent.robot.get_qvel())) # prevent the robot arm from moving too fast
                    # reaching_reward2 -= 0.0003 * qvel_penalty
                    # if appropriate_height_penalty < 0.01:
                    reaching_reward2 += 4 * (1 - np.tanh(5.0 * cubeA_to_goal_dist))
                    reward += np.maximum(reaching_reward2, 0.0)

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 15.0
