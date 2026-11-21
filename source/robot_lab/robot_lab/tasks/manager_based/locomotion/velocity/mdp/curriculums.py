# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_stairs(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Change stair rows only after measured traversal succeeds."""
    terrain = env.scene.terrain
    command_term = env.command_manager.get_term("base_velocity")
    progress = command_term.metrics["stair_progress"][env_ids]
    height_change = command_term.metrics["stair_height_change"][env_ids]
    eligible = command_term._had_translation[env_ids] & command_term._start_valid[env_ids]
    failed = env.termination_manager.terminated[env_ids]
    success = eligible & ~failed & (progress >= command_term.cfg.success_distance) & (
        height_change >= command_term.cfg.min_height_change
    )
    if not hasattr(env, "_stairs_success_streak"):
        env._stairs_success_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    streak = env._stairs_success_streak
    streak[env_ids] = torch.where(success, streak[env_ids] + 1, 0)
    move_up = success & (streak[env_ids] >= 2) & (terrain.terrain_levels[env_ids] < terrain.max_terrain_level - 1)
    move_down = eligible & (failed | (progress < 0.6))
    terrain.update_env_origins(env_ids, move_up, move_down)
    streak[env_ids] = torch.where(move_up, 0, streak[env_ids])
    return terrain.terrain_levels.float().mean()


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)
