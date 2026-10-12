# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
import re
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from .him_actor_critic import HIMActorCritic
from .him_ppo import HIMPPO


class HIMOnPolicyRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.cfg = train_cfg
        self.policy_cfg = dict(train_cfg.get("policy", {}))
        self.alg_cfg = dict(train_cfg.get("algorithm", {}))
        self.device = device
        self.env = env
        self.num_steps_per_env = int(train_cfg["num_steps_per_env"])
        self.save_interval = int(train_cfg["save_interval"])
        self.max_iterations = int(train_cfg.get("max_iterations", 0))

        self.policy_cfg.pop("num_one_step_obs", None)
        if self.env.num_privileged_obs is None:
            raise ValueError(
                "HIM requires a dedicated critic observation group; "
                "actor observations cannot be used as a silent fallback."
            )
        num_critic_obs = self.env.num_privileged_obs
        self._resolve_estimator_slices(num_critic_obs)
        actor_critic = HIMActorCritic(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_one_step_obs,
            self.env.num_actions,
            **self.policy_cfg,
        )
        print(
            "[INFO] HIM observation layout: "
            f"actor_obs={self.env.num_obs}, one_step_obs={self.env.num_one_step_obs}, "
            f"history={actor_critic.history_size}, critic_obs={num_critic_obs}, "
            f"vel_slice={actor_critic.vel_slice}, target_slices={actor_critic.target_slices}"
        )
        self.alg = HIMPPO(actor_critic, device=self.device, **self.alg_cfg)
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.latest_episode_metrics: dict[str, float] = {}
        self.latest_mean_reward: float | None = None
        self.latest_mean_episode_length: float | None = None
        self.episode_metric_history: deque[dict[str, torch.Tensor | float | int]] = deque(maxlen=100)

        _, reset_infos = self.env.reset()
        self._update_episode_metric_cache(self._extract_log_dict(reset_infos))

    def _resolve_estimator_slices(self, num_critic_obs: int):
        vel_slice = self._select_slice(
            self.policy_cfg.get("estimator_vel_slice"),
            getattr(self.env, "default_estimator_vel_slice", None),
            num_critic_obs,
            "estimator_vel_slice",
        )
        target_slices = getattr(self.env, "default_estimator_target_slices", None)
        if vel_slice is None:
            raise ValueError("HIM requires a base_lin_vel term in the critic observation group.")
        if not target_slices:
            raise ValueError("HIM could not infer estimator target fields from critic observations.")
        self.policy_cfg["estimator_vel_slice"] = vel_slice
        self.policy_cfg.pop("estimator_target_slice", None)
        self.policy_cfg["estimator_target_slices"] = target_slices

    @staticmethod
    def _is_valid_slice(obs_slice: tuple[int, int] | list[int] | None, upper_bound: int) -> bool:
        if obs_slice is None:
            return False
        if len(obs_slice) != 2:
            return False
        start, stop = int(obs_slice[0]), int(obs_slice[1])
        return 0 <= start < stop <= upper_bound

    def _select_slice(
        self,
        configured_slice: tuple[int, int] | list[int] | None,
        inferred_slice: tuple[int, int] | None,
        upper_bound: int,
        slice_name: str,
    ) -> tuple[int, int] | None:
        if self._is_valid_slice(inferred_slice, upper_bound):
            resolved_slice = (int(inferred_slice[0]), int(inferred_slice[1]))
            configured_valid = self._is_valid_slice(configured_slice, upper_bound)
            configured_value = None
            if configured_valid:
                configured_value = (int(configured_slice[0]), int(configured_slice[1]))
            if configured_slice is not None and configured_value != resolved_slice:
                print(
                    f"[WARN] HIM {slice_name}={tuple(configured_slice)} does not match the environment layout. "
                    f"Using inferred slice {resolved_slice}."
                )
            return resolved_slice
        if self._is_valid_slice(configured_slice, upper_bound):
            return (int(configured_slice[0]), int(configured_slice[1]))
        if configured_slice is not None:
            raise ValueError(f"Invalid HIM {slice_name}={tuple(configured_slice)} for critic dim {upper_bound}.")
        return None

    def add_git_repo_to_log(self, source_file: str):
        del source_file

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.actor_critic.estimator.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": {
                "num_one_step_obs": self.alg.actor_critic.num_one_step_obs,
                "vel_slice": self.alg.actor_critic.vel_slice,
                "target_slices": self.alg.actor_critic.target_slices,
            },
            "observation_contract": self.env.get_observation_contract(),
        }
        torch.save(payload, path)

    def load(self, path: str, allow_legacy: bool = False, inference: bool = False):
        loaded_dict = torch.load(path, map_location=self.device)
        contract = loaded_dict.get("observation_contract")
        if contract is None and not allow_legacy:
            raise ValueError("HIM checkpoint has no observation contract; explicit allow_legacy=True is required.")
        if contract is None:
            print("[WARN] Loading legacy HIM weights without verified observation or teacher semantics.")
        if contract is not None:
            legacy = "format_version" not in contract
            current_contract = self.env.get_observation_contract(legacy=legacy)
            saved_contract = copy.deepcopy(contract)
            if inference:
                for candidate in (saved_contract, current_contract):
                    self._relax_inference_contract(candidate)
            differences = self._contract_differences(saved_contract, current_contract)
            if differences:
                details = "\n".join(f"  - {difference}" for difference in differences[:20])
                if len(differences) > 20:
                    details += f"\n  - ... {len(differences) - 20} more differences"
                raise ValueError(
                    f"HIM checkpoint observation contract differs from the current environment:\n{details}\n"
                    "Restore the checkpoint's observation configuration or start a new training run. "
                    "--him_allow_legacy_checkpoint only applies to checkpoints without a contract."
                )
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        if "optimizer_state_dict" in loaded_dict:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except (RuntimeError, ValueError):
                print("[WARN] HIM optimizer state is incompatible with the current model. Skipping optimizer load.")
        if "estimator_optimizer_state_dict" in loaded_dict:
            try:
                self.alg.actor_critic.estimator.optimizer.load_state_dict(
                    loaded_dict["estimator_optimizer_state_dict"]
                )
            except (RuntimeError, ValueError):
                print("[WARN] HIM estimator optimizer state is incompatible. Skipping estimator optimizer load.")
        else:
            print("[WARN] HIM checkpoint has no estimator optimizer state. Estimator optimizer starts fresh.")
        self.current_learning_iteration = int(loaded_dict.get("iter", 0))

    @staticmethod
    def _relax_inference_contract(contract: dict):
        contract.pop("policy_corruption", None)
        contract.pop("critic_corruption", None)
        for group in contract["groups"].values():
            for term in group:
                term.pop("noise", None)
                if term["name"] == "velocity_commands":
                    term.pop("params", None)
                    term.pop("scale", None)
                    term.pop("clip", None)
                    term.pop("func", None)

    @classmethod
    def _contract_differences(cls, saved, current, path: str = "") -> list[str]:
        if isinstance(saved, dict) and isinstance(current, dict):
            differences = []
            for key in sorted(saved.keys() | current.keys(), key=str):
                child_path = f"{path}.{key}" if path else str(key)
                if key not in saved or key not in current:
                    differences.append(f"{child_path}: checkpoint={cls._contract_display(saved.get(key, '<missing>'))}, current={cls._contract_display(current.get(key, '<missing>'))}")
                else:
                    differences.extend(cls._contract_differences(saved[key], current[key], child_path))
            return differences
        if isinstance(saved, list) and isinstance(current, list):
            differences = []
            if len(saved) != len(current):
                differences.append(f"{path}.length: checkpoint={len(saved)}, current={len(current)}")
            for index, (saved_item, current_item) in enumerate(zip(saved, current)):
                term_name = saved_item.get("name") if isinstance(saved_item, dict) else None
                child_path = f"{path}[{index}{':' + term_name if term_name else ''}]"
                differences.extend(cls._contract_differences(saved_item, current_item, child_path))
            return differences
        if isinstance(saved, str) and isinstance(current, str):
            saved = re.sub(r" at 0x[0-9a-fA-F]+", "", saved)
            current = re.sub(r" at 0x[0-9a-fA-F]+", "", current)
        if saved != current:
            return [f"{path}: checkpoint={cls._contract_display(saved)}, current={cls._contract_display(current)}"]
        return []

    @staticmethod
    def _contract_display(value) -> str:
        rendered = repr(value)
        return rendered if len(rendered) <= 180 else rendered[:177] + "..."

    def get_inference_policy(self, device: str | None = None):
        if device is None:
            device = self.device
        self.alg.actor_critic.eval()

        def policy(obs: torch.Tensor):
            with torch.inference_mode():
                return self.alg.actor_critic.act_inference(obs.to(device))

        return policy

    def _get_critic_observations(self) -> torch.Tensor:
        privileged_obs = self.env.get_critic_observations()
        if privileged_obs is None:
            return self.env.get_observations()
        return privileged_obs

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf") and hasattr(self.env, "max_episode_length"):
            env_unwrapped = getattr(self.env, "unwrapped", self.env)
            env_unwrapped.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        critic_obs = self._get_critic_observations().to(self.device)
        self.alg.actor_critic.train()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        ep_infos = []
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for iteration in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            episodes_completed = 0
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    critic_obs = self._get_critic_observations().to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_critic_obs, estimator_valid = self.env.get_estimator_next_critic_observations()
                    self.alg.process_env_step(
                        rewards, dones, infos, next_critic_obs.to(self.device), estimator_valid.to(self.device)
                    )

                    cur_reward_sum += rewards.view(-1)
                    cur_episode_length += 1
                    done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if done_ids.numel() > 0:
                        episodes_completed += done_ids.numel()
                        rewbuffer.extend(cur_reward_sum[done_ids].detach().cpu().tolist())
                        lenbuffer.extend(cur_episode_length[done_ids].detach().cpu().tolist())
                        self._append_episode_log(ep_infos, infos)
                        cur_reward_sum[done_ids] = 0
                        cur_episode_length[done_ids] = 0

                collection_time = time.time() - start
                learn_start = time.time()
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_estimation_loss,
                mean_swap_loss,
                mean_entropy_loss,
            ) = self.alg.update()
            learn_time = time.time() - learn_start

            self._log(
                iteration,
                tot_iter,
                collection_time,
                learn_time,
                mean_value_loss,
                mean_surrogate_loss,
                mean_estimation_loss,
                mean_swap_loss,
                mean_entropy_loss,
                rewbuffer,
                lenbuffer,
                ep_infos,
                episodes_completed,
                float(cur_reward_sum.mean().item()),
                float(cur_episode_length.mean().item()),
            )
            ep_infos.clear()

            self.current_learning_iteration = iteration + 1
            if self.log_dir is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log(
        self,
        iteration: int,
        total_iterations: int,
        collection_time: float,
        learn_time: float,
        mean_value_loss: float,
        mean_surrogate_loss: float,
        mean_estimation_loss: float,
        mean_swap_loss: float,
        mean_entropy_loss: float,
        rewbuffer: deque,
        lenbuffer: deque,
        ep_infos: list[dict[str, torch.Tensor | float | int]],
        episodes_completed: int,
        partial_mean_reward: float,
        partial_mean_episode_length: float,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time
        fps = int(self.num_steps_per_env * self.env.num_envs / max(collection_time + learn_time, 1e-6))
        mean_reward = statistics.mean(rewbuffer) if rewbuffer else None
        mean_episode_length = statistics.mean(lenbuffer) if lenbuffer else None
        mean_noise_std = self.alg.actor_critic.std.mean().item()
        self.episode_metric_history.extend(info.copy() for info in ep_infos)
        episode_metrics = self._aggregate_episode_infos(list(self.episode_metric_history))
        if mean_reward is not None:
            self.latest_mean_reward = mean_reward
        if mean_episode_length is not None:
            self.latest_mean_episode_length = mean_episode_length
        if episode_metrics:
            self.latest_episode_metrics = dict(episode_metrics)

        display_reward = mean_reward
        display_episode_length = mean_episode_length
        if display_reward is None:
            display_reward = self.latest_mean_reward if self.latest_mean_reward is not None else partial_mean_reward
        if display_episode_length is None:
            if self.latest_mean_episode_length is not None:
                display_episode_length = self.latest_mean_episode_length
            else:
                display_episode_length = partial_mean_episode_length
        display_episode_metrics = episode_metrics if episode_metrics else list(self.latest_episode_metrics.items())
        running_metrics = self._collect_running_episode_metrics(partial_mean_reward, partial_mean_episode_length)

        if self.writer is not None:
            self.writer.add_scalar("Loss/value_function", mean_value_loss, iteration)
            self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, iteration)
            self.writer.add_scalar("Loss/estimation", mean_estimation_loss, iteration)
            self.writer.add_scalar("Loss/swap", mean_swap_loss, iteration)
            self.writer.add_scalar("Loss/entropy", mean_entropy_loss, iteration)
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
            self.writer.add_scalar(
                "Loss/estimator_learning_rate",
                self.alg.actor_critic.estimator.learning_rate,
                iteration,
            )
            self.writer.add_scalar("Policy/mean_noise_std", mean_noise_std, iteration)
            self.writer.add_scalar("Perf/fps", fps, iteration)
            self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
            self.writer.add_scalar("Perf/learning_time", learn_time, iteration)
            self.writer.add_scalar("Train/mean_reward", display_reward, iteration)
            self.writer.add_scalar("Train/mean_episode_length", display_episode_length, iteration)
            self.writer.add_scalar("Train/episodes_completed", episodes_completed, iteration)
            self.writer.add_scalar("Train/episode_metric_batches", len(self.episode_metric_history), iteration)
            for key, value in display_episode_metrics:
                self.writer.add_scalar(key, value, iteration)
            for key, value in running_metrics.items():
                self.writer.add_scalar(key, value, iteration)

        width = 80
        pad = 35
        title = f" Learning iteration {iteration + 1}/{total_iterations} "
        log_string = f"{'#' * width}\n{title.center(width, ' ')}\n\n"
        log_string += (
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
            f"{'Mean action noise std:':>{pad}} {mean_noise_std:.2f}\n"
            f"{'Mean value_function loss:':>{pad}} {mean_value_loss:.4f}\n"
            f"{'Mean surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"
            f"{'Mean estimation loss:':>{pad}} {mean_estimation_loss:.4f}\n"
            f"{'Mean swap loss:':>{pad}} {mean_swap_loss:.4f}\n"
            f"{'Mean entropy loss:':>{pad}} {mean_entropy_loss:.4f}\n"
        )
        log_string += (
            f"{'Mean reward:':>{pad}} {display_reward:.2f}\n"
            f"{'Mean episode length:':>{pad}} {display_episode_length:.2f}\n"
            f"{'Episodes completed this iteration:':>{pad}} {episodes_completed}\n"
            f"{'Episode metric history batches:':>{pad}} {len(self.episode_metric_history)}\n"
        )
        for key, value in display_episode_metrics:
            log_string += f"{key:>{pad}}: {value:.4f}\n"
        for key, value in running_metrics.items():
            log_string += f"{key:>{pad}}: {value:.4f}\n"
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {self._format_duration(self.tot_time)}\n"
            f"{'ETA:':>{pad}} {self._format_duration(self.tot_time / max(iteration + 1, 1) * (total_iterations - iteration - 1))}\n"
        )
        print(log_string, flush=True)

    def _collect_running_episode_metrics(self, running_reward: float, running_length: float) -> dict[str, float]:
        """Snapshot metrics for in-progress episodes once per rollout iteration."""
        env = getattr(self.env, "unwrapped", self.env)
        metrics = {
            "Running/episode_reward": running_reward,
            "Running/episode_length": running_length,
        }

        reward_sums = getattr(getattr(env, "reward_manager", None), "_episode_sums", {})
        for term_name, values in reward_sums.items():
            metrics[f"Running/Episode_Reward/{term_name}"] = self._mean_metric_value(values)

        command_manager = getattr(env, "command_manager", None)
        if command_manager is not None:
            command_term = command_manager.get_term("base_velocity")
            for name, values in command_term.metrics.items():
                metrics[f"Running/Metrics/base_velocity/{name}"] = self._mean_metric_value(values)

        terrain = getattr(getattr(env, "scene", None), "terrain", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        if terrain_levels is not None:
            metrics["Running/Curriculum/terrain_levels"] = self._mean_metric_value(terrain_levels)
        return metrics

    @staticmethod
    def _mean_metric_value(value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().item())
        return float(value)

    @staticmethod
    def _append_episode_log(ep_infos: list[dict], infos: dict):
        ep_info = HIMOnPolicyRunner._extract_log_dict(infos)
        if isinstance(ep_info, dict) and ep_info:
            ep_infos.append(ep_info.copy())

    @staticmethod
    def _extract_log_dict(infos: dict | None) -> dict | None:
        if not isinstance(infos, dict):
            return None
        ep_info = infos.get("log")
        return ep_info if isinstance(ep_info, dict) else None

    def _update_episode_metric_cache(self, ep_info: dict | None):
        if not ep_info:
            return
        for key, value in self._aggregate_episode_infos([ep_info]):
            self.latest_episode_metrics[key] = value

    def _aggregate_episode_infos(
        self, ep_infos: list[dict[str, torch.Tensor | float | int]]
    ) -> list[tuple[str, float]]:
        if not ep_infos:
            return []
        metrics: list[tuple[str, float]] = []
        ordered_keys = self._ordered_metric_keys(ep_infos)
        for key in ordered_keys:
            values = []
            for ep_info in ep_infos:
                if key not in ep_info:
                    continue
                value = ep_info[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([value], device=self.device, dtype=torch.float32)
                else:
                    value = value.to(self.device, dtype=torch.float32)
                if value.ndim == 0:
                    value = value.unsqueeze(0)
                values.append(value.flatten())
            if values:
                metrics.append((key, torch.cat(values).mean().item()))
        return metrics

    @staticmethod
    def _ordered_metric_keys(ep_infos: list[dict[str, torch.Tensor | float | int]]) -> list[str]:
        priority_keys = [
            "Metrics/base_velocity/error_vel_x",
            "Metrics/base_velocity/error_vel_y",
            "Metrics/base_velocity/error_vel_z",
            "Metrics/base_velocity/error_vel_dir",
            "Metrics/base_velocity/error_vel_xy",
            "Metrics/base_velocity/error_vel_yaw",
        ]
        keys = []
        seen = set()
        for key in priority_keys:
            for ep_info in ep_infos:
                if key in ep_info and key not in seen:
                    keys.append(key)
                    seen.add(key)
                    break
        for ep_info in ep_infos:
            for key in ep_info:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        return keys

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(int(seconds), 0)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
