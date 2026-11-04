# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_dir"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()

        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        lin_vel_error_b = self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2]
        self.metrics["error_vel_x"] += torch.abs(lin_vel_error_b[:, 0]) / max_command_step
        self.metrics["error_vel_y"] += torch.abs(lin_vel_error_b[:, 1]) / max_command_step
        self.metrics["error_vel_z"] += torch.abs(self.robot.data.root_lin_vel_b[:, 2]) / max_command_step

        command_heading = torch.atan2(self.vel_command_b[:, 1], self.vel_command_b[:, 0])
        actual_heading = torch.atan2(self.robot.data.root_lin_vel_b[:, 1], self.robot.data.root_lin_vel_b[:, 0])
        command_speed = torch.linalg.norm(self.vel_command_b[:, :2], dim=-1)
        actual_speed = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        valid_mask = (command_speed > 1e-6) & (actual_speed > 1e-6)
        direction_error = torch.zeros(self.num_envs, device=self.device)
        if valid_mask.any():
            heading_delta = command_heading[valid_mask] - actual_heading[valid_mask]
            direction_error[valid_mask] = torch.abs(torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta)))
        self.metrics["error_vel_dir"] += direction_error / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class PhaseVelocityCommand(mdp.UniformVelocityCommand):
    """Keep one command-driven gait clock shared by observations and rewards."""

    def __init__(self, cfg: mdp.UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._stop_elapsed = torch.zeros_like(self.gait_phase)
        self._was_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stopping = torch.zeros_like(self._was_moving)

    def _update_command(self):
        super()._update_command()
        command = self.vel_command_b
        moving = (torch.linalg.norm(command[:, :2], dim=1) >= 0.05) | (command[:, 2].abs() >= 0.05)
        starting_stop = ~moving & self._was_moving
        self._stopping[starting_stop] = True
        self._stop_elapsed[starting_stop] = 0.0
        active = moving | self._stopping
        self.gait_phase[active] = torch.remainder(
            self.gait_phase[active] + self._env.step_dt / getattr(self.cfg, "cycle_time", 0.5), 1.0
        )
        self._stop_elapsed[self._stopping & ~moving] += self._env.step_dt
        self._stopping[moving | (self._stop_elapsed >= getattr(self.cfg, "stop_duration", 0.5))] = False
        self.gait_phase[~moving & ~self._stopping] = 0.0
        self._was_moving[:] = moving

    def reset(self, env_ids: Sequence[int] | None = None):
        metrics = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.gait_phase[ids] = 0.0
        self._stop_elapsed[ids] = 0.0
        self._was_moving[ids] = False
        self._stopping[ids] = False
        return metrics


class StairsVelocityCommand(PhaseVelocityCommand):
    """Command outward from the stair platform and measure actual traversal."""

    cfg: StairsVelocityCommandCfg

    def __init__(self, cfg: StairsVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.metrics["stair_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["stair_height_change"] = torch.zeros(self.num_envs, device=self.device)
        self._had_translation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._start_valid = torch.zeros_like(self._had_translation)
        self._start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._start_height = torch.zeros(self.num_envs, device=self.device)
        self._travel_direction = torch.zeros_like(self._start_xy)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        flat_mask = getattr(self._env, "_stair_flat_mask", None)
        stair_ids = env_ids if flat_mask is None else env_ids[~flat_mask[env_ids]]
        count = len(stair_ids)
        if count == 0:
            return
        direction = getattr(self._env, "_stair_spawn_direction", None)
        if direction is None:
            lateral = torch.rand(count, device=self.device) < self.cfg.lateral_ratio
            sign = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
            selected = torch.stack((torch.where(lateral, 0.0, sign), torch.where(lateral, sign, 0.0)), dim=1)
        else:
            selected = direction[stair_ids]
        speed = torch.empty(count, device=self.device).uniform_(*self.cfg.stair_speed_range)
        self.vel_command_b[stair_ids, :3] = 0.0
        self.vel_command_b[stair_ids, :2] = selected * speed.unsqueeze(1)
        self.vel_command_b[stair_ids[self.is_standing_env[stair_ids]], :2] = 0.0
        self._travel_direction[stair_ids] = selected

    def _update_metrics(self):
        super()._update_metrics()
        command = self.vel_command_b[:, :2]
        speed = torch.linalg.norm(command, dim=1)
        moving = speed > self.cfg.min_command_speed
        flat_mask = getattr(self._env, "_stair_flat_mask", None)
        if flat_mask is not None:
            moving &= ~flat_mask
        self._had_translation |= moving
        ground = self._env.scene.sensors["height_scanner_base"].data.ray_hits_w[..., 2].median(dim=1).values
        position = self.robot.data.root_pos_w[:, :2]
        initialize = moving & ~self._start_valid & torch.isfinite(ground)
        self._start_xy[initialize] = position[initialize]
        self._start_height[initialize] = ground[initialize]
        self._start_valid[initialize] = True
        displacement = ((position - self._start_xy) * self._travel_direction).sum(dim=1)
        self.metrics["stair_progress"] = torch.where(
            self._start_valid, displacement.clamp_min(0.0), 0.0
        )
        height_change = (ground - self._start_height).abs()
        self.metrics["stair_height_change"] = torch.where(
            self._start_valid & torch.isfinite(ground),
            height_change,
            0.0,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        flat_mask = getattr(self._env, "_stair_flat_mask", None)
        eligible = self._had_translation[ids]
        if flat_mask is not None:
            eligible = eligible & ~flat_mask[ids]
        success = (
            (self.metrics["stair_progress"][ids] >= self.cfg.success_distance)
            & (self.metrics["stair_height_change"][ids] >= self.cfg.min_height_change)
        )
        stair_success = (success & eligible).float().sum() / eligible.float().sum().clamp_min(1)
        metrics = super().reset(env_ids)
        metrics["stair_success"] = stair_success.item()
        self._had_translation[ids] = False
        self._start_valid[ids] = False
        return metrics


@configclass
class StairsVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    class_type: type = StairsVelocityCommand
    stair_speed_range: tuple[float, float] = (0.25, 0.8)
    lateral_ratio: float = 0.35
    min_command_speed: float = 0.2
    success_distance: float = 1.5
    min_height_change: float = 0.06
    cycle_time: float = 0.5
    stop_duration: float = 0.5


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
