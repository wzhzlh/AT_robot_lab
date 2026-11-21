"""Observation helpers matching Cyberstar's deployed policy input."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.managers import ManagerTermBase, ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def moving_command(command, translation_threshold=0.05, yaw_threshold=0.05):
    return ((torch.linalg.vector_norm(command[:, :2], dim=-1) >= translation_threshold)
            | (command[:, 2].abs() >= yaw_threshold))


class CSDog1Phase(ManagerTermBase):
    """Advance once per control step while moving; hold zero while standing."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._phase = torch.zeros(env.num_envs, device=env.device)
        self._last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self._phase[ids] = 0.0
        self._last_step[ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        cycle_time: float,
        stop_cycle_time: float,
        stop_duration: float,
        translation_threshold: float,
        yaw_threshold: float,
    ) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        moving = moving_command(command, translation_threshold, yaw_threshold)
        update = self._last_step != env.common_step_counter
        running = update & moving
        self._phase[running] = (self._phase[running] + env.step_dt / cycle_time) % 1.0
        self._phase[~moving] = 0.0
        self._last_step[update] = env.common_step_counter

        angle = 2 * torch.pi * self._phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous network output after deployment's action clipping."""
    return env.action_manager.action.clamp(-100.0, 100.0)


def velocity_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Pre-scale limits equivalent to clipping scaled commands to +/-100."""
    command = env.command_manager.get_command(command_name)
    limits = command.new_tensor((100.0, 100.0, 400.0))
    return torch.clamp(command, min=-limits, max=limits)


class CSDog1JointPositionAction(JointPositionAction):
    """Clip network actions before scaling/offset, as deployment does."""

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions.clamp(-100.0, 100.0))
