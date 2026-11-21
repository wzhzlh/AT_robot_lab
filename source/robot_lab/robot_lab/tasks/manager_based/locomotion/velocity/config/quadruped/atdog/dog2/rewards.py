"""Dog2 adaptations of Cyberstar's phase-guided swing reward."""

import math

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply
from robot_lab.terrain_raycast import raycast_mesh

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from .phase_observations import CSDog1Phase, moving_command


def standing_contact(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    stationary = ~moving_command(command)
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    upright = (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0.0, 0.7) / 0.7
    return contact.float().sum(dim=1) * stationary * upright


def standing_joint_velocity(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    stationary = ~moving_command(command)
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return velocity.square().mean(dim=1).clamp(max=25.0) * stationary


def standing_posture(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return error.square().mean(dim=1) * ~moving_command(env.command_manager.get_command(command_name))


def standing_base_velocity(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    velocity = data.root_lin_vel_b.square().sum(dim=1) + data.root_ang_vel_b.square().sum(dim=1)
    return velocity * ~moving_command(env.command_manager.get_command(command_name))


def turning_two_foot_support(env, command_name: str, sensor_cfg: SceneEntityCfg,
                             yaw_threshold: float, translation_threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    turning = ((command[:, 2].abs() >= yaw_threshold)
               & (torch.linalg.vector_norm(command[:, :2], dim=1) < translation_threshold))
    sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float().sum(dim=1)
    upright = (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0.0, 0.7) / 0.7
    return (contacts == 2).float() * turning * upright


class CommandedGaitReward(mdp.GaitReward):
    def __call__(self, env, std, command_name, max_err, velocity_threshold, command_threshold,
                 synced_feet_pair_names, asset_cfg, sensor_cfg):
        reward = super().__call__(env, std, command_name, max_err, velocity_threshold, command_threshold,
                                  synced_feet_pair_names, asset_cfg, sensor_cfg)
        command = env.command_manager.get_command(self.command_name)
        return reward * moving_command(command)


class PhaseSwingClearance(ManagerTermBase):
    """Penalize swing-foot height error against the Cyberstar trot profile."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.phase_clock = CSDog1Phase(cfg, env)
        body_names = cfg.params["asset_cfg"].body_names
        if len(body_names) != 4 or any(name not in {"FL_calf", "FR_calf", "RL_calf", "RR_calf"} for name in body_names):
            raise ValueError("PhaseSwingClearance requires four ordered Dog2 calf bodies.")
        self.foot_offsets = torch.tensor(
            [(0.22, 0.0, 0.001 if name == "FL_calf" else 0.0) for name in body_names],
            device=env.device,
        )
        terrain_cfg = cfg.params.get("terrain_sensor_cfg")
        self.terrain_scanner = env.scene.sensors[terrain_cfg.name] if terrain_cfg is not None else None
        self.terrain_mesh = None
        self.debug_metrics = {}

    def reset(self, env_ids=None):
        self.phase_clock.reset(env_ids)

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        cycle_time: float,
        stop_cycle_time: float,
        stop_duration: float,
        translation_threshold: float,
        yaw_threshold: float,
        phase_offset: float,
        stance_ratio: float,
        contact_smoothing: float,
        foot_radius: float,
        swing_height: float,
        turn_height_bonus: float,
        terrain_sensor_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        phase_features = self.phase_clock(
            env, command_name, cycle_time, stop_cycle_time, stop_duration,
            translation_threshold, yaw_threshold,
        )
        phase = torch.remainder(torch.atan2(phase_features[:, 0], phase_features[:, 1]) / (2 * math.pi), 1.0)
        opposite_phase = torch.remainder(phase + phase_offset, 1.0)
        leg_phase = torch.stack((phase, opposite_phase, opposite_phase, phase), dim=1)

        swing_phase = ((leg_phase - stance_ratio) / (1.0 - stance_ratio)).clamp(0.0, 1.0)
        swing_profile = torch.sin(math.pi * swing_phase)
        command = env.command_manager.get_command(command_name)
        turning = (command[:, 2].abs() > 0.2) & (torch.linalg.vector_norm(command[:, :2], dim=1) < 0.05)
        target_clearance = (swing_height + turn_height_bonus * turning[:, None]) * swing_profile

        asset = env.scene[asset_cfg.name]
        body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
        body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
        offsets = self.foot_offsets.unsqueeze(0).expand(env.num_envs, -1, -1)
        foot_offset_w = quat_apply(body_quat.reshape(-1, 4), offsets.reshape(-1, 3))
        foot_pos = body_pos + foot_offset_w.reshape(env.num_envs, 4, 3)
        foot_height = foot_pos[..., 2]
        valid = torch.ones_like(foot_height, dtype=torch.bool)
        terrain_height = torch.zeros_like(foot_height)
        if terrain_sensor_cfg is not None:
            if self.terrain_mesh is None:
                self.terrain_mesh = RayCaster.meshes[self.terrain_scanner.cfg.mesh_prim_paths[0]]
            starts = foot_pos.clone()
            starts[..., 2] = torch.maximum(foot_height, asset.data.root_pos_w[:, None, 2]) + 0.5
            directions = torch.zeros_like(starts)
            directions[..., 2] = -1.0
            hits, _, normals, _ = raycast_mesh(starts, directions, mesh=self.terrain_mesh,
                                               max_dist=3.0, return_normal=True)
            valid = torch.isfinite(hits).all(dim=-1) & torch.isfinite(normals).all(dim=-1) & (normals[..., 2].abs() > 0.5)
            terrain_height = torch.where(valid, hits[..., 2], 0.0)
        clearance = foot_height - terrain_height - foot_radius

        score = torch.cos(2 * math.pi * (leg_phase - 0.5 * stance_ratio))
        desired_contact = torch.sigmoid((score - math.cos(math.pi * stance_ratio)) / contact_smoothing)
        desired_swing = (1.0 - desired_contact) * valid
        error = ((clearance - target_clearance).square() * desired_swing).sum(dim=1)
        self.debug_metrics = {
            "foot_world_height": foot_height.detach(), "terrain_height": terrain_height.detach(),
            "foot_clearance": clearance.detach(), "target_clearance": target_clearance.detach(),
            "clearance_error": (clearance - target_clearance).detach(), "swing_phase": swing_phase.detach(),
            "terrain_valid": valid.detach(),
        }
        return error / desired_swing.sum(dim=1).clamp_min(1.0) * moving_command(command).float()


def configure_dog2_phase_swing_reward(cfg):
    cfg.rewards.phase_swing_clearance = RewardTermCfg(
        func=PhaseSwingClearance,
        weight=-160.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"], preserve_order=True,
            ),
            "command_name": "base_velocity",
            "cycle_time": 0.5,
            "stop_cycle_time": 0.5,
            "stop_duration": 0.5,
            "translation_threshold": 0.05,
            "yaw_threshold": 0.05,
            "phase_offset": 0.5,
            "stance_ratio": 0.55,
            "contact_smoothing": 0.07,
            "foot_radius": 0.02,
            "swing_height": 0.135,
            "turn_height_bonus": 0.02,
        },
    )
