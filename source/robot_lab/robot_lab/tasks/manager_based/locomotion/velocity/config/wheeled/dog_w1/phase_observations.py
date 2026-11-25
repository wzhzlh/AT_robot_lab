"""dog_w1 command-gated sine/cosine phase, matching the dog2 phase convention."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def moving_command(command, translation_threshold=0.05, yaw_threshold=0.05):
    return ((torch.linalg.vector_norm(command[:, :2], dim=-1) >= translation_threshold)
            | (command[:, 2].abs() >= yaw_threshold))


class DogW1Phase(ManagerTermBase):
    """Advance once per control step while moving; hold zero while standing."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # All observation/reward instances share state; reward is evaluated before
        # observations in Isaac Lab, so independent clocks can drift after resets.
        if not hasattr(env, "_dog_w1_phase_state"):
            env._dog_w1_phase_state = {
                "phase": torch.zeros(env.num_envs, device=env.device),
                "last_step": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
                "parameters": None,
            }
        self._state = env._dog_w1_phase_state
        self._phase = self._state["phase"]
        self._last_step = self._state["last_step"]

    @property
    def phase(self):
        return self._phase

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
        if cycle_time <= 0:
            raise ValueError("dog_w1 cycle_time must be positive.")
        parameters = (command_name, cycle_time, stop_cycle_time, stop_duration,
                      translation_threshold, yaw_threshold)
        if self._state["parameters"] is None:
            self._state["parameters"] = parameters
        elif self._state["parameters"] != parameters:
            raise ValueError("dog_w1 observation and reward phase parameters must match.")
        command = env.command_manager.get_command(command_name)
        moving = moving_command(command, translation_threshold, yaw_threshold)
        update = self._last_step != env.common_step_counter
        running = update & moving
        self._phase[running] = (self._phase[running] + env.step_dt / cycle_time) % 1.0
        self._phase[~moving] = 0.0
        self._last_step[update] = env.common_step_counter

        angle = 2 * torch.pi * self._phase
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
