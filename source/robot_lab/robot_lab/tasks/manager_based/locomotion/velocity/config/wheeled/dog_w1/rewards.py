"""dog_w1 contact and clearance targets driven by the observation phase."""

import math

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster

from robot_lab.terrain_raycast import raycast_mesh
from .phase_observations import DogW1Phase, moving_command


FOOT_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def phase_gait_targets(phase, stance_ratio, swing_height, phase_offset=0.5, contact_smoothing=0.0):
    """FL/RR are synchronous; FR/RL lag half a cycle. Heights are above support."""
    if not 0.5 <= stance_ratio < 1.0:
        raise ValueError("dog_w1 stance_ratio must be in [0.5, 1.0).")
    if not 0.0 <= phase_offset < 1.0:
        raise ValueError("dog_w1 phase_offset must be in [0, 1).")
    if contact_smoothing < 0:
        raise ValueError("dog_w1 contact_smoothing must be nonnegative.")
    opposite = (phase + phase_offset) % 1.0
    leg_phase = torch.stack((phase, opposite, opposite, phase), dim=1)
    stance = leg_phase < stance_ratio
    if contact_smoothing > 0:
        # Same smooth contact window as dog2's PhaseSwingClearance.
        score = torch.cos(2 * math.pi * (leg_phase - 0.5 * stance_ratio))
        stance = torch.sigmoid((score - math.cos(math.pi * stance_ratio)) / contact_smoothing)
    swing = ((leg_phase - stance_ratio) / (1.0 - stance_ratio)).clamp(0.0, 1.0)
    clearance = swing_height * torch.sin(math.pi * swing)
    return stance, clearance


def standing_contact(env, command_name, sensor_cfg, translation_threshold=0.05, yaw_threshold=0.05):
    """Reward ongoing wheel support, not only the first frame of touchdown."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contacts = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    command = env.command_manager.get_command(command_name)
    stationary = ~moving_command(command, translation_threshold, yaw_threshold)
    upright = (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0.0, 0.7) / 0.7
    return contacts.float().sum(dim=1) * stationary * upright


class PhaseContactReward(ManagerTermBase):
    """Reward each wheel matching its phase-dependent stance/swing state."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if list(cfg.params["sensor_cfg"].body_names) != FOOT_ORDER:
            raise ValueError("PhaseContactReward requires ordered FL, FR, RL, RR wheel contacts.")
        self.clock = DogW1Phase(cfg, env)
        self.debug_metrics = {}

    def reset(self, env_ids=None):
        self.clock.reset(env_ids)

    def __call__(self, env, sensor_cfg, command_name, cycle_time, stop_cycle_time, stop_duration,
                 translation_threshold, yaw_threshold, stance_ratio, phase_offset=0.5, contact_smoothing=0.07, centered=False):
        phase_features = self.clock(env, command_name, cycle_time, stop_cycle_time, stop_duration,
                                    translation_threshold, yaw_threshold)
        phase = self.clock.phase
        stance, _ = phase_gait_targets(phase, stance_ratio, 0.0, phase_offset, contact_smoothing)
        contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        match = 1.0 - (contact.float() - stance.float()).square().mean(dim=1)
        if centered:
            # Per wheel: +1 for correct support/swing, -1 for wrong state.
            # Four permanently grounded wheels no longer receive ~half the maximum.
            match = ((2.0 * contact.float() - 1.0) * (2.0 * stance.float() - 1.0)).mean(dim=1)
        moving = moving_command(env.command_manager.get_command(command_name), translation_threshold, yaw_threshold)
        upright = (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0.0, 0.7) / 0.7
        self.debug_metrics = {"desired_contact": stance.detach(), "actual_contact": contact.detach(),
                              "phase": phase_features.detach()}
        return match * moving * upright


class PhaseSwingClearance(ManagerTermBase):
    """Track a sine lift in swing, using terrain height under each wheel center."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if list(cfg.params["asset_cfg"].body_names) != FOOT_ORDER:
            raise ValueError("PhaseSwingClearance requires ordered FL, FR, RL, RR wheel bodies.")
        self.clock = DogW1Phase(cfg, env)
        terrain_cfg = cfg.params.get("terrain_sensor_cfg")
        self.scanner = env.scene.sensors[terrain_cfg.name] if terrain_cfg is not None else None
        self.mesh = None
        self.debug_metrics = {}

    def reset(self, env_ids=None):
        self.clock.reset(env_ids)

    def __call__(self, env, asset_cfg, command_name, cycle_time, stop_cycle_time, stop_duration,
                 translation_threshold, yaw_threshold, stance_ratio, swing_height, foot_radius,
                 phase_offset=0.5, contact_smoothing=0.07, turn_height_bonus=0.0, terrain_sensor_cfg=None):
        self.clock(env, command_name, cycle_time, stop_cycle_time, stop_duration,
                   translation_threshold, yaw_threshold)
        command = env.command_manager.get_command(command_name)
        turning = ((command[:, 2].abs() > 0.2)
                   & (torch.linalg.vector_norm(command[:, :2], dim=1) < translation_threshold))
        lift_height = swing_height + turn_height_bonus * turning[:, None]
        stance, target = phase_gait_targets(
            self.clock.phase, stance_ratio, lift_height, phase_offset, contact_smoothing,
        )
        asset = env.scene[asset_cfg.name]
        positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
        terrain_height = env.scene.env_origins[:, 2:3].expand_as(target)
        valid = torch.ones_like(target, dtype=torch.bool)
        if terrain_sensor_cfg is not None:
            if self.mesh is None:
                self.mesh = RayCaster.meshes[self.scanner.cfg.mesh_prim_paths[0]]
            starts = positions.clone()
            starts[..., 2] = torch.maximum(positions[..., 2], asset.data.root_pos_w[:, None, 2]) + 0.5
            directions = torch.zeros_like(starts)
            directions[..., 2] = -1.0
            hits, _, normals, _ = raycast_mesh(starts, directions, mesh=self.mesh,
                                               max_dist=3.0, return_normal=True)
            valid = torch.isfinite(hits).all(dim=-1) & torch.isfinite(normals).all(dim=-1) & (normals[..., 2].abs() > 0.5)
            terrain_height = torch.where(valid, hits[..., 2], terrain_height)
        clearance = positions[..., 2] - terrain_height - foot_radius
        swing = (1.0 - stance.float()) * valid
        error = ((clearance - target).square() * swing).sum(dim=1) / swing.sum(dim=1).clamp_min(1)
        moving = moving_command(env.command_manager.get_command(command_name), translation_threshold, yaw_threshold)
        self.debug_metrics = {"phase": self.clock.phase.detach(), "desired_contact": stance.detach(),
                              "target_clearance": target.detach(), "clearance": clearance.detach(),
                              "terrain_valid": valid.detach()}
        return error * moving


def lateral_velocity_error(env, command_name, asset_cfg):
    """Dense lateral error without the XY exponential's vanishing gradient at large errors."""
    command = env.command_manager.get_command(command_name)
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b
    return (command[:, 1] - velocity[:, 1]).square()
