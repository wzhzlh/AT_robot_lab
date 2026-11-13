"""Fixed-frequency trot clock shared by training and deployment (PyTorch only).

Phase zero starts FL/RR stance. Feet are always ordered FL, FR, RL, RR.
Updating the clock belongs to the command loop, never an observation/reward.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass
class TrotClockCfg:
    cycle_time: float = 0.7
    duty_factor: float = 0.6
    transition_width: float = 0.05
    linear_enter: float = 0.08
    linear_exit: float = 0.04
    angular_enter: float = 0.10
    angular_exit: float = 0.05

    def __post_init__(self):
        if not math.isfinite(self.cycle_time) or self.cycle_time <= 0:
            raise ValueError("cycle_time must be finite and positive.")
        if not 0.5 < self.duty_factor < 1.0:
            raise ValueError("duty_factor must leave both double support and swing: 0.5 < beta < 1.")
        if not 0.0 < self.transition_width < (1.0 - self.duty_factor) / 2:
            raise ValueError("transition_width must leave a nonempty swing core.")
        for enter, leave in ((self.linear_enter, self.linear_exit), (self.angular_enter, self.angular_exit)):
            if not 0 <= leave < enter or not math.isfinite(enter):
                raise ValueError("Motion thresholds must satisfy 0 <= exit < enter < infinity.")


class TrotClock:
    """Batched command clock; stop at the next planned landing, without resetting phase.

    Stopping takes at most half a cycle. It is phase driven, with no hidden
    stop timer, contact-dependent phase correction, or variable frequency.
    """

    def __init__(self, num_envs: int, device, cfg: TrotClockCfg):
        # Hydra mutates nested configuration fields after dataclass construction.
        cfg.__post_init__()
        self.cfg = cfg
        self.phase = torch.zeros(num_envs, device=device)
        self.translating = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.turning = torch.zeros_like(self.translating)
        self.moving = torch.zeros_like(self.translating)
        self.stopping = torch.zeros_like(self.translating)

    @classmethod
    def from_metadata(cls, num_envs: int, device, metadata: dict):
        if metadata.get("version") != 1:
            raise ValueError("Unsupported gait clock metadata version.")
        cfg = TrotClockCfg(**{name: metadata[name] for name in TrotClockCfg.__dataclass_fields__})
        return cls(num_envs, device, cfg)

    @property
    def active(self):
        return self.moving | self.stopping

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.phase, self.translating, self.turning, self.moving, self.stopping):
            value[ids] = 0

    def update(self, command: torch.Tensor, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Clock dt must be finite and positive.")
        linear = torch.linalg.vector_norm(command[:, :2], dim=-1)
        angular = command[:, 2].abs()
        self.translating[:] = torch.where(
            self.translating, linear > self.cfg.linear_exit, linear >= self.cfg.linear_enter
        )
        self.turning[:] = torch.where(self.turning, angular > self.cfg.angular_exit, angular >= self.cfg.angular_enter)
        self.moving[:] = self.translating | self.turning

        # Double support: phi mod 0.5 in [0, beta - 0.5]. A stopped
        # clock in this interval stays still, rather than starting a new swing.
        half_phase = torch.remainder(self.phase, 0.5)
        in_support = half_phase <= self.cfg.duty_factor - 0.5 + 1.0e-6
        self.stopping[:] = ~self.moving & ~in_support
        advance = torch.full_like(self.phase, dt / self.cfg.cycle_time)
        until_landing = 0.5 - half_phase
        advance = torch.where(self.stopping, torch.minimum(advance, until_landing), advance)
        self.phase[:] = torch.remainder(self.phase + advance * self.active, 1.0)
        landed = self.stopping & (advance >= until_landing - 1.0e-6)
        self.stopping[landed] = False

    def observation(self):
        angle = 2.0 * math.pi * self.phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    def foot_targets(self):
        return trot_targets(self.phase, self.cfg)

    def metadata(self):
        return {
            "version": 1,
            **asdict(self.cfg),
            "initial_phase": 0.0,
            "encoding": ["sin", "cos"],
            "foot_order": ["FL", "FR", "RL", "RR"],
            "phase_offsets": [0.0, 0.5, 0.5, 0.0],
            "stance_interval": "[0, duty_factor)",
            "stop_rule": "finish_current_swing_then_freeze_in_double_support",
            "update_order": "reward_with_action_phase_then_update_clock_then_next_observation",
            "reference_implementation": "robot_lab.gait:TrotClock",
        }


def trot_targets(phase: torch.Tensor, cfg: TrotClockCfg):
    """Return stance/swing confidence and swing progress, with tolerant boundaries."""
    offsets = phase.new_tensor((0.0, 0.5, 0.5, 0.0))
    local_phase = torch.remainder(phase[:, None] + offsets, 1.0)
    stance_distance = torch.minimum(local_phase, cfg.duty_factor - local_phase)
    swing_distance = torch.minimum(local_phase - cfg.duty_factor, 1.0 - local_phase)

    def smooth_weight(distance):
        x = (distance / cfg.transition_width).clamp(0.0, 1.0)
        return x.square() * (3.0 - 2.0 * x)

    progress = ((local_phase - cfg.duty_factor) / (1.0 - cfg.duty_factor)).clamp(0.0, 1.0)
    return smooth_weight(stance_distance), smooth_weight(swing_distance), progress


def contact_violation(contact, stance_weight, swing_weight, active):
    """Nonnegative violation: use a negative reward weight, no constant bonus."""
    contact = contact.to(stance_weight.dtype)
    return (swing_weight * contact + stance_weight * (1.0 - contact)).mean(dim=-1) * active


def standing_motion_error(joint_velocity, action_delta, active):
    """Quiet-joint penalty after planned landing; leave moving/recovery steps free."""
    velocity_error = (joint_velocity / 0.5).square().clamp(max=4.0).mean(dim=-1)
    action_error = (action_delta / 0.05).square().clamp(max=4.0).mean(dim=-1)
    return (velocity_error + action_error) * ~active


def swing_clearance_error(clearance, progress, swing_weight, valid, active, height: float, std: float,
                          deficit_only: bool = False):
    target = height * torch.sin(math.pi * progress).square()
    delta = clearance - target
    if deficit_only:
        delta = delta.clamp(max=0.0)
    error = (delta / std).square().clamp(max=4.0)
    # Invalid geometry must not contaminate rewards with NaN/inf.
    error = torch.where(valid, error, 0.0)
    return (error * swing_weight).mean(dim=-1) * active


def stair_traversal_success(progress, height_change, eligible, failed, distance: float, min_height: float):
    """Evaluate completed stair episodes, excluding falls and non-stair commands."""
    return eligible & ~failed & (progress >= distance) & (height_change >= min_height)


def foot_geometry(body_pos, body_quat, offsets, radius: float, half_width: float):
    """Foot centers and world vertical sole extent for the URDF foot cylinders.

    Quaternions are wxyz. Offsets locate fixed foot joints in calf link frames.
    Radius/half_width bound dog2's link4 collision mesh; the cylinder axis is
    calf-local z (the fixed foot rotation only reverses this axis).
    """
    xyz = body_quat[..., 1:]
    offset = offsets.expand_as(body_pos)
    cross = 2.0 * torch.cross(xyz, offset, dim=-1)
    center = body_pos + offset + body_quat[..., :1] * cross + torch.cross(xyz, cross, dim=-1)
    axis_z = (1.0 - 2.0 * (xyz[..., 0].square() + xyz[..., 1].square())).clamp(-1.0, 1.0)
    extent = radius * (1.0 - axis_z.square()).clamp_min(0.0).sqrt() + half_width * axis_z.abs()
    return center, extent
