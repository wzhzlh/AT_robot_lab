# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.utils.warp import raycast_mesh

from robot_lab.gait import contact_violation, foot_geometry, stair_edge_error, standing_motion_error, swing_clearance_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def hip_outward_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    outward_signs: tuple[float, ...] | list[float],
) -> torch.Tensor:
    """Penalize hip deviations from default only in each joint's outward direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if len(joint_ids) != len(outward_signs):
        raise ValueError(
            f"Expected one outward sign per selected hip joint, got {len(outward_signs)} signs "
            f"for {len(joint_ids)} joints."
        )
    signs = torch.as_tensor(outward_signs, device=asset.device, dtype=asset.data.joint_pos.dtype)
    joint_deviation = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    outward_deviation = torch.clamp(joint_deviation * signs.unsqueeze(0), min=0.0)
    return torch.sum(outward_deviation, dim=1)


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Compare magnitudes so mirrored joints are encouraged to move with similar amplitude
        diff = torch.sum(
            torch.square(
                torch.abs(asset.data.joint_pos[:, joint_pair[0][0]])
                - torch.abs(asset.data.joint_pos[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_variance_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize variance in wheel velocities (absolute values) to encourage uniform wheel speeds.
    
    This reward encourages all four wheels to rotate at similar speeds (in absolute value),
    which helps maintain stable and coordinated wheeled locomotion.
    
    Args:
        env: The RL environment.
        asset_cfg: Scene entity configuration for the robot asset. Defaults to "robot".
            The joint_ids should correspond to the four wheel joints.
    
    Returns:
        Penalty value based on the variance of absolute wheel velocities.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get wheel joint velocities
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    # compute variance of absolute wheel velocities
    wheel_vel_abs = torch.abs(wheel_vel)
    reward = torch.var(wheel_vel_abs, dim=1)
    # scale by gravity to reduce penalty when robot is not upright
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def diagonal_trot_contact_pattern(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    foot_names: tuple[str, str, str, str],
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize non-trot contact patterns while the robot is commanded to move.

    The foot order is FL, FR, RL, RR. A low penalty means FL/RR and FR/RL are synchronized, while the two
    diagonal pairs are in opposite contact states. This biases the policy away from pacing, bounding, and
    four-foot hopping without affecting stand-still behavior.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cache_name = "_diagonal_trot_contact_body_ids"
    cached_foot_names, body_ids = getattr(env, cache_name, (None, None))
    if cached_foot_names != foot_names:
        body_ids = contact_sensor.find_bodies(list(foot_names))[0]
        if len(body_ids) != 4:
            raise ValueError(f"Expected four foot bodies for diagonal trot reward, got {len(body_ids)}: {foot_names}")
        setattr(env, cache_name, (foot_names, body_ids))

    contact = (contact_sensor.data.current_contact_time[:, body_ids] > 0.0).float()
    fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]

    diagonal_mismatch = torch.abs(fl - rr) + torch.abs(fr - rl)
    diagonal_pair_same_phase = 1.0 - torch.abs(fl - fr)
    reward = diagonal_mismatch + diagonal_pair_same_phase
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


class GaitContactPenalty(ManagerTermBase):
    """Phase contact violations; use a negative weight, including through stopping."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.body_ids = []
        for name in cfg.params["foot_names"]:
            ids, _ = self.sensor.find_bodies(name)
            if len(ids) != 1:
                raise ValueError(f"Expected exactly one contact body for {name!r}, got {len(ids)}.")
            self.body_ids.append(ids[0])
        if len(self.body_ids) != 4:
            raise ValueError("Gait contact feet must be ordered FL, FR, RL, RR.")

    def __call__(self, env, sensor_cfg, foot_names, command_name):
        clock = env.command_manager.get_term(command_name).gait_clock
        stance, swing, _ = clock.foot_targets()
        contact = self.sensor.data.current_contact_time[:, self.body_ids] > 0.0
        return contact_violation(contact, stance, swing, clock.active)


class SwingClearancePenalty(ManagerTermBase):
    """Track actual sole clearance above terrain, only during planned swing.

    Rays query the terrain mesh already loaded by the privileged height scanner;
    no extra actor observations or sensors are introduced.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.body_ids = cfg.params["asset_cfg"].body_ids
        self.scanner = env.scene.sensors[cfg.params["terrain_sensor_cfg"].name]
        self.offsets = torch.tensor(cfg.params["foot_offsets"], device=env.device)
        if self.offsets.shape != (4, 3) or len(self.body_ids) != 4:
            raise ValueError("Swing clearance requires four ordered foot bodies and four xyz offsets.")
        self._terrain_cache_key = (
            cfg.params["asset_cfg"].name, tuple(self.body_ids),
            tuple(tuple(offset) for offset in cfg.params["foot_offsets"]),
            tuple(self.scanner.cfg.mesh_prim_paths),
            cfg.params["foot_radius"], cfg.params["foot_half_width"],
        )
        self._sample_offsets = self.offsets.new_tensor(
            ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))
        )

    def reset(self, env_ids=None):
        # A reset can change poses without advancing common_step_counter.
        self._env._gait_terrain_cache = None

    def __call__(
        self, env, asset_cfg, terrain_sensor_cfg, command_name, foot_offsets,
        foot_radius: float, foot_half_width: float, height: float, std: float,
        terrain_sample_radius: float = 0.0, lookahead_distance: float = 0.0,
        deficit_only: bool = False,
        profile: str = "sin_squared",
    ):
        clock = env.command_manager.get_term(command_name).gait_clock
        _, swing, progress = clock.foot_targets()
        data = self.asset.data
        center, extent = foot_geometry(
            data.body_link_pos_w[:, self.body_ids], data.body_link_quat_w[:, self.body_ids],
            self.offsets, foot_radius, foot_half_width,
        )
        # Accessing data ensures the lazy ray caster has initialized its mesh.
        _ = self.scanner.data
        mesh = RayCaster.meshes[self.scanner.cfg.mesh_prim_paths[0]]
        # Probe the sole footprint and the next edge in the commanded direction.
        # At swing boundaries lookahead fades out, allowing normal touchdown.
        sample_count = 6 if terrain_sample_radius > 0 or lookahead_distance > 0 else 1
        starts = center.unsqueeze(-2).expand(-1, -1, sample_count, -1).clone()
        starts[..., :2] += self._sample_offsets[:sample_count] * terrain_sample_radius
        if lookahead_distance > 0:
            command = env.command_manager.get_command(command_name)[:, :2]
            yaw = data.heading_w
            cosine, sine = yaw.cos(), yaw.sin()
            direction = torch.stack((
                cosine * command[:, 0] - sine * command[:, 1],
                sine * command[:, 0] + cosine * command[:, 1],
            ), dim=-1)
            direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            starts[:, :, -1, :2] += (
                direction[:, None, :] * lookahead_distance * torch.sin(torch.pi * progress).unsqueeze(-1)
            )
        starts[..., 2] += 0.5
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1.0
        hits, _, normals, _ = raycast_mesh(starts, directions, mesh=mesh, max_dist=2.0, return_normal=True)
        if sample_count >= 5:
            env._gait_terrain_cache = (
                env.common_step_counter, self._terrain_cache_key,
                terrain_sample_radius, hits[..., :5, 2],
            )
        valid_samples = torch.isfinite(hits).all(dim=-1) & torch.isfinite(normals).all(dim=-1)
        valid_samples &= normals[..., 2].abs() >= 0.5
        valid = valid_samples.all(dim=-1)
        ground = torch.where(valid_samples, hits[..., 2], 0.0).max(dim=-1).values
        clearance = center[..., 2] - extent - ground
        return swing_clearance_error(clearance, progress, swing, valid, clock.active, height, std, deficit_only, profile)


class StairLandingPenalty(SwingClearancePenalty):
    """Penalize supporting a foot across a step edge, using a sole footprint proxy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.contact_ids = cfg.params["sensor_cfg"].body_ids
        self._landing_offsets = torch.cat((
            self._sample_offsets[:5], self.offsets.new_zeros((5, 1)),
        ), dim=-1) * cfg.params["foot_radius"]

    def __call__(self, env, asset_cfg, terrain_sensor_cfg, command_name, foot_offsets,
                 sensor_cfg, foot_radius: float, foot_half_width: float, edge_tolerance: float):
        cached = getattr(env, "_gait_terrain_cache", None)
        if (cached is not None and cached[0] == env.common_step_counter
                and cached[1] == self._terrain_cache_key and cached[2] == foot_radius):
            heights = cached[3]
        else:
            data = self.asset.data
            center, _ = foot_geometry(
                data.body_link_pos_w[:, self.body_ids], data.body_link_quat_w[:, self.body_ids],
                self.offsets, foot_radius, foot_half_width,
            )
            starts = center.unsqueeze(-2) + self._landing_offsets
            starts[..., 2] += 0.5
            directions = torch.zeros_like(starts)
            directions[..., 2] = -1.0
            _ = self.scanner.data
            mesh = RayCaster.meshes[self.scanner.cfg.mesh_prim_paths[0]]
            hits, _, _, _ = raycast_mesh(starts, directions, mesh=mesh, max_dist=2.0)
            heights = hits[..., 2]
        contact = self.sensor.data.current_contact_time[:, self.contact_ids] > 0.0
        active = env.command_manager.get_term(command_name).gait_clock.active
        return stair_edge_error(heights, contact, active, edge_tolerance)


def standing_support(
    env, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, foot_offsets,
) -> torch.Tensor:
    """Reward sustained quiet support, never the number of touchdown events."""
    clock = env.command_manager.get_term(command_name).gait_clock
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float().mean(dim=-1)
    data = asset.data
    offsets = data.root_pos_w.new_tensor(foot_offsets)
    center, _ = foot_geometry(
        data.body_link_pos_w[:, asset_cfg.body_ids], data.body_link_quat_w[:, asset_cfg.body_ids], offsets, 0.0, 0.0
    )
    lever = center - data.body_link_pos_w[:, asset_cfg.body_ids]
    foot_velocity = data.body_link_lin_vel_w[:, asset_cfg.body_ids] + torch.cross(
        data.body_link_ang_vel_w[:, asset_cfg.body_ids], lever, dim=-1
    )
    body_motion = data.root_lin_vel_b.square().sum(dim=-1) / 0.1**2
    body_motion += data.root_ang_vel_b.square().sum(dim=-1) / 0.2**2
    foot_motion = foot_velocity.square().sum(dim=-1).mean(dim=-1) / 0.1**2
    action_change = (env.action_manager.action - env.action_manager.prev_action).square().mean(dim=-1) / 0.1**2
    upright = (-data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
    return contact * torch.exp(-body_motion - foot_motion - action_change) * upright * ~clock.active


def standing_joint_motion(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    clock = env.command_manager.get_term(command_name).gait_clock
    joint_velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    action_delta = env.action_manager.action - env.action_manager.prev_action
    return standing_motion_error(joint_velocity, action_delta, clock.active)


def standing_yaw_drift(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw drift after stopping, without restricting commanded turns."""
    clock = env.command_manager.get_term(command_name).gait_clock
    yaw_rate = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2]
    return (yaw_rate / std).square().clamp(max=9.0) * ~clock.active


def gait_foot_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, foot_offsets) -> torch.Tensor:
    """Horizontal slip at fixed foot centers, rather than calf link origins."""
    data = env.scene[asset_cfg.name].data
    center, _ = foot_geometry(
        data.body_link_pos_w[:, asset_cfg.body_ids], data.body_link_quat_w[:, asset_cfg.body_ids],
        data.root_pos_w.new_tensor(foot_offsets), 0.0, 0.0,
    )
    velocity = data.body_link_lin_vel_w[:, asset_cfg.body_ids] + torch.cross(
        data.body_link_ang_vel_w[:, asset_cfg.body_ids], center - data.body_link_pos_w[:, asset_cfg.body_ids], dim=-1
    )
    contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return ((velocity[..., :2].square().sum(dim=-1) / 0.2**2).clamp(max=4.0) * contact).mean(dim=-1)


class PeriodicFootLiftReward(ManagerTermBase):
    """Match alternating diagonal foot contact to a sinusoidal gait clock."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        foot_names = cfg.params["foot_names"]
        if len(foot_names) != 4:
            raise ValueError("Periodic foot lift requires feet ordered FL, FR, RL, RR.")
        self.body_ids = []
        for name in foot_names:
            ids, _ = sensor.find_bodies(name)
            if len(ids) != 1:
                raise ValueError(f"Expected exactly one contact body for {name!r}, got {len(ids)}.")
            self.body_ids.append(ids[0])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        foot_names: tuple[str, str, str, str],
        cycle_time: float,
        command_name: str,
        command_threshold: float,
        progress_weighted: bool = False,
        phase_from_command: bool = False,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact = (sensor.data.current_contact_time[:, self.body_ids] > 0.0).float()
        if phase_from_command:
            phase = 2.0 * math.pi * env.command_manager.get_term(command_name).gait_phase
        else:
            phase = 2.0 * math.pi * env.episode_length_buf * env.step_dt / cycle_time
        swing = 0.5 * (torch.sin(phase) + 1.0)
        desired_air = torch.stack((swing, 1.0 - swing, 1.0 - swing, swing), dim=1)
        reward = (desired_air * (1.0 - contact) + (1.0 - desired_air) * contact).mean(dim=1)
        moving = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
        if progress_weighted:
            command = env.command_manager.get_command(command_name)[:, :2]
            target_speed = torch.linalg.norm(command, dim=1)
            direction = command / target_speed.clamp_min(1.0e-6).unsqueeze(1)
            actual_speed = torch.sum(env.scene["robot"].data.root_lin_vel_b[:, :2] * direction, dim=1)
            reward *= torch.clamp(actual_speed / target_speed.clamp_min(0.2), min=0.0, max=1.0)
        return reward * moving


def feet_height_body_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
) -> torch.Tensor:
    """Reward feet as their height in the body frame approaches the target."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    relative_pos = foot_pos - asset.data.root_pos_w[:, None, :]
    root_quat = asset.data.root_quat_w
    foot_height = torch.stack(
        [quat_apply_inverse(root_quat, relative_pos[:, index])[:, 2] for index in range(relative_pos.shape[1])],
        dim=1,
    )
    return torch.exp(-torch.square((foot_height - target_height) / std)).mean(dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_smoothness_2_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action differences to suppress high-frequency oscillation."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev_action = getattr(env, "_action_smoothness_2_prev_action", None)
    if prev_prev_action is None or prev_prev_action.shape != prev_action.shape:
        prev_prev_action = torch.zeros_like(prev_action)

    diff = torch.square(action - 2.0 * prev_action + prev_prev_action)
    valid = torch.logical_and(prev_action != 0.0, prev_prev_action != 0.0)
    reward = torch.sum(diff * valid, dim=1)

    env._action_smoothness_2_prev_action = prev_action.detach().clone()
    return reward


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
