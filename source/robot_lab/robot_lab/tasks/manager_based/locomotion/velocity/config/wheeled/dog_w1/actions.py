"""Wheel velocity control with a finite-torque parking brake at zero SE(2) command."""

import torch
from isaaclab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg
from isaaclab.utils import configclass

from .phase_observations import moving_command


def parking_brake_step(position, velocity, moving, was_locked, hold_position,
                       velocity_target, stiffness, damping, effort_limit):
    locked = ~moving
    # Capture the angle once on entry, rather than chasing the wheel every step.
    capture = locked & ~was_locked
    hold = torch.where(capture[:, None], position, hold_position)
    effort = (stiffness * (hold - position) - damping * velocity).clamp(-effort_limit, effort_limit)
    effort = torch.where(locked[:, None], effort, torch.zeros_like(effort))
    target = torch.where(locked[:, None], torch.zeros_like(velocity_target), velocity_target)
    return locked, hold, target, effort


class DogW1WheelVelocityAction(JointVelocityAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._brake_env = env
        self._locked = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._hold_position = self._asset.data.joint_pos[:, self._joint_ids].clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._locked[ids] = False
        # The first apply after reset captures the newly reset joint positions.

    def apply_actions(self):
        cfg = self.cfg
        command = self._brake_env.command_manager.get_command(cfg.command_name)
        moving = moving_command(command, cfg.translation_threshold, cfg.yaw_threshold)
        self._locked, self._hold_position, target, effort = parking_brake_step(
            self._asset.data.joint_pos[:, self._joint_ids],
            self._asset.data.joint_vel[:, self._joint_ids],
            moving, self._locked, self._hold_position, self.processed_actions,
            cfg.brake_stiffness, cfg.brake_damping, cfg.brake_effort_limit,
        )
        self._asset.set_joint_velocity_target(target, joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(effort, joint_ids=self._joint_ids)


@configclass
class DogW1WheelVelocityActionCfg(JointVelocityActionCfg):
    class_type = DogW1WheelVelocityAction
    command_name: str = "base_velocity"
    translation_threshold: float = 0.05
    yaw_threshold: float = 0.05
    brake_stiffness: float = 20.0
    brake_damping: float = 2.0
    brake_effort_limit: float = 15.0
