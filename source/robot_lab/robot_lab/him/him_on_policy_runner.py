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

        configured_history = self.policy_cfg.pop("history_length", None)
        if configured_history is not None and int(configured_history) != self.env.history_length:
            raise ValueError(
                "HIM policy history length differs from the live observation history: "
                f"configured={configured_history}, actual={self.env.history_length}."
            )
        self.policy_cfg.pop("num_one_step_obs", None)
        configured_order = self.policy_cfg.pop("history_order", None)
        live_order = getattr(self.env, "history_order", None)
        if configured_order is not None and live_order is not None and configured_order != live_order:
            raise ValueError("HIM policy history_order differs from the live wrapper history_order.")
        expected_frame = {"oldest_to_latest": "last", "latest_to_oldest": "first"}.get(live_order)
        if expected_frame and self.policy_cfg.get("actor_current_frame", "first") != expected_frame:
            raise ValueError(f"HIM actor_current_frame must be {expected_frame!r} for history_order={live_order!r}.")
        # This runner uses fixed observation-manager scaling, not running statistics.
        # Reject unsupported flags instead of silently ignoring normalization requests.
        if (train_cfg.get("empirical_normalization") or
                self.policy_cfg.get("actor_obs_normalization") or self.policy_cfg.get("critic_obs_normalization")):
            raise ValueError("HIM running observation normalization is unsupported; use fixed observation term scales.")
        self.policy_cfg.pop("estimator_target_term_names", None)
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
        self.observation_logger = None
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        self.latest_episode_metrics: dict[str, float] = {}
        self.latest_mean_reward: float | None = None
        self.latest_mean_episode_length: float | None = None
        # Each window entry stores (sum, count) per metric key, so the running
        # average costs O(keys) per iteration instead of O(batches * keys).
        self.episode_metric_window: deque[dict[str, tuple[float, int]]] = deque(maxlen=100)
        self._episode_metric_totals: dict[str, list[float]] = {}

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
        if vel_slice is None or vel_slice[1] - vel_slice[0] != 3:
            raise ValueError("HIM requires an inferred, exactly 3-D base_lin_vel term in the critic observation group.")
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
        raise ValueError(f"HIM requires a live, valid {slice_name} inferred from the critic observation layout.")

    def add_git_repo_to_log(self, source_file: str):
        del source_file

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "checkpoint_format_version": 2,
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.actor_critic.estimator.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": {
                "num_one_step_obs": self.alg.actor_critic.num_one_step_obs,
                "vel_slice": self.alg.actor_critic.vel_slice,
                "target_slices": self.alg.actor_critic.target_slices,
                "actor_current_frame": getattr(self.alg.actor_critic, "actor_current_frame", None),
            },
            "observation_contract": self.env.get_observation_contract(),
        }
        torch.save(payload, path)

    def load(
        self,
        path: str,
        allow_legacy: bool = False,
        inference: bool = False,
        allow_observation_migration: bool = False,
        weights_only: bool = False,
    ):
        loaded_dict = torch.load(path, map_location=self.device)
        contract = loaded_dict.get("observation_contract")
        saved_frame = (loaded_dict.get("infos") or {}).get("actor_current_frame")
        if saved_frame is None and contract is not None:
            saved_frame = {"time_major_oldest_to_current": "last", "time_major_current_to_oldest": "first"}.get(
                contract.get("history_layout")
            )
        live_frame = getattr(self.alg.actor_critic, "actor_current_frame", None)
        if saved_frame is not None and live_frame is not None and saved_frame != live_frame:
            raise ValueError(f"HIM checkpoint actor_current_frame={saved_frame!r} differs from model={live_frame!r}.")
        critic_input_migrated = False
        observation_migrated = False
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
            if differences and allow_observation_migration and not inference:
                observation_migrated = self._is_atdog2_observation_migration(saved_contract, current_contract)
                if observation_migrated:
                    differences = []
                else:
                    # Recognize a phase/noise migration even when critic inputs
                    # also changed. Preserve the explicit opt-in for policy semantics.
                    candidate = copy.deepcopy(current_contract)
                    new_by_name = {term["name"]: term for term in current_contract["groups"]["critic"]}
                    candidate["groups"]["critic"] = [
                        copy.deepcopy(new_by_name.get(term["name"], term))
                        for term in saved_contract["groups"]["critic"]
                    ]
                    candidate["teacher_target_slices"] = saved_contract.get("teacher_target_slices")
                    candidate["velocity_slice"] = saved_contract.get("velocity_slice")
                    if self._is_atdog2_observation_migration(saved_contract, candidate):
                        saved_contract = candidate
                        observation_migrated = True
            if differences:
                critic_input_migrated = self._can_migrate_critic_inputs(saved_contract, current_contract)
                if critic_input_migrated:
                    differences = []
            if differences:
                details = "\n".join(f"  - {difference}" for difference in differences[:20])
                if len(differences) > 20:
                    details += f"\n  - ... {len(differences) - 20} more differences"
                raise ValueError(
                    f"HIM checkpoint observation contract differs from the current environment:\n{details}\n"
                    "Restore the checkpoint's observation configuration or start a new training run. "
                    "Use --him_allow_observation_migration only for the known ATDog2 phase/critic-noise change. "
                    "--him_allow_legacy_checkpoint only applies to checkpoints without a contract."
                )
        model_state = loaded_dict["model_state_dict"]
        if critic_input_migrated:
            model_state = self._migrate_critic_input_weights(model_state, saved_contract, current_contract)
            print(
                "[WARN] HIM critic input layout changed: matched input columns by observation name/definition; "
                "removed columns discarded, new/changed inputs zero-initialized. "
                "PPO optimizer starts fresh; saved iteration retained. Value estimates must readapt."
            )
        if critic_input_migrated or observation_migrated:
            self.alg.optimizer.state.clear()
        if observation_migrated:
            self.alg.actor_critic.estimator.optimizer.state.clear()
            print(
                "[WARN] Warm-starting HIM from the previous ATDog2 observation semantics: "
                "the actor/critic weights are retained, while PPO and estimator optimizer states start fresh. "
                "The saved iteration count is retained."
            )
        self.alg.actor_critic.load_state_dict(model_state, strict=True)
        if "optimizer_state_dict" in loaded_dict and not critic_input_migrated and not observation_migrated and not weights_only:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            except (RuntimeError, ValueError):
                print("[WARN] HIM optimizer state is incompatible with the current model. Skipping optimizer load.")
        if "estimator_optimizer_state_dict" in loaded_dict and not observation_migrated and not weights_only:
            try:
                self.alg.actor_critic.estimator.optimizer.load_state_dict(
                    loaded_dict["estimator_optimizer_state_dict"]
                )
            except (RuntimeError, ValueError):
                print("[WARN] HIM estimator optimizer state is incompatible. Skipping estimator optimizer load.")
        elif not observation_migrated and not weights_only:
            print("[WARN] HIM checkpoint has no estimator optimizer state. Estimator optimizer starts fresh.")
        if weights_only:
            self.current_learning_iteration = 0
            print("[INFO] HIM weights initialized; iteration and both optimizer states start fresh.")
        else:
            self.current_learning_iteration = int(loaded_dict.get("iter", 0))

    @classmethod
    def _is_atdog2_observation_migration(cls, saved: dict, current: dict) -> bool:
        """Accept only the prior ATDog2 phase clock and critic noise semantics."""
        old_phase = "robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations.phase"
        new_phase = "robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations.stairs_phase"
        old_noise_fields = {"base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"}
        previous = copy.deepcopy(saved)
        updated = copy.deepcopy(current)
        if previous.get("critic_corruption") is not True or updated.get("critic_corruption") is not False:
            return False
        previous["critic_corruption"] = False
        try:
            for group_name in ("policy", "critic"):
                old = next(term for term in previous["groups"][group_name] if term["name"] == "phase")
                new = next(term for term in updated["groups"][group_name] if term["name"] == "phase")
                if old.get("func") != old_phase or old.get("params") != {"cycle_time": 0.5}:
                    return False
                if new.get("func") != new_phase or new.get("params") != {"command_name": "base_velocity"}:
                    return False
                old["func"] = new["func"]
                old["params"] = new["params"]
            for term in previous["groups"]["critic"]:
                name = term["name"]
                if name in old_noise_fields:
                    if term.get("noise") is None:
                        return False
                    term["noise"] = None
                elif term.get("noise") is not None:
                    return False
        except (KeyError, StopIteration, TypeError):
            return False
        return not cls._contract_differences(previous, updated)

    @classmethod
    def _can_migrate_critic_inputs(cls, saved: dict, current: dict) -> bool:
        """Allow only critic layout changes; actor and teacher semantics stay fixed."""
        try:
            previous, updated = copy.deepcopy(saved), copy.deepcopy(current)
            old_terms = previous["groups"].pop("critic")
            new_terms = updated["groups"].pop("critic")
            old = cls._critic_term_layout(old_terms)
            new = cls._critic_term_layout(new_terms)
            # Teacher fields drive estimator training and cannot silently change.
            required = ["base_lin_vel", *saved.get("teacher_target_terms", [])]
            for name in required:
                if name not in old or name not in new or old[name][2] != new[name][2]:
                    return False
            # Offsets may change, but must actually refer to the declared fields.
            for contract, layout in ((previous, old), (updated, new)):
                if contract.get("velocity_slice") != list(layout["base_lin_vel"][:2]):
                    return False
                expected = [list(layout[name][:2]) for name in contract["teacher_target_terms"]]
                if contract.get("teacher_target_slices") != expected:
                    return False
                contract.pop("velocity_slice")
                contract.pop("teacher_target_slices")
            return not cls._contract_differences(previous, updated)
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _critic_term_layout(terms):
        layout = {}
        offset = 0
        for term in terms:
            name, dim = term["name"], term["dim"]
            if name in layout or not isinstance(dim, int) or dim <= 0:
                raise ValueError("Critic terms must have unique names and positive dimensions")
            layout[name] = (offset, offset + dim, term)
            offset += dim
        return layout

    def _migrate_critic_input_weights(self, checkpoint_state, saved, current):
        old = self._critic_term_layout(saved["groups"]["critic"])
        new = self._critic_term_layout(current["groups"]["critic"])
        current_state = self.alg.actor_critic.state_dict()
        if set(checkpoint_state) != set(current_state):
            raise ValueError("Cannot migrate checkpoint with missing/unexpected model parameters")
        key = "critic.0.weight"
        source, target = checkpoint_state[key], current_state[key]
        old_width = sum(stop - start for start, stop, _ in old.values())
        new_width = sum(stop - start for start, stop, _ in new.values())
        if source.shape != (target.shape[0], old_width) or target.shape[1] != new_width:
            raise ValueError("Critic first layer does not match checkpoint/current observation contracts")
        adapted = dict(checkpoint_state)
        for name, value in checkpoint_state.items():
            if name != key and value.shape != current_state[name].shape:
                raise ValueError(f"Cannot migrate non-input parameter {name}: {value.shape} -> {current_state[name].shape}")
        weight = torch.zeros_like(target)
        for name, (start, stop, descriptor) in new.items():
            if name in old and old[name][2] == descriptor:
                old_start, old_stop, _ = old[name]
                weight[:, start:stop] = source[:, old_start:old_stop]
        adapted[key] = weight
        return adapted

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
        # Finished-episode bookkeeping is accumulated on the device and moved to the
        # host once per iteration; per-step masked selects would sync the stream.
        episode_done = torch.zeros(
            (self.num_steps_per_env, self.env.num_envs), dtype=torch.bool, device=self.device
        )
        episode_reward = torch.zeros(
            (self.num_steps_per_env, self.env.num_envs), dtype=torch.float, device=self.device
        )
        episode_length = torch.zeros_like(episode_reward)
        timer = _SectionTimer(self.device)
        # Kept across iterations so a stale extras["log"] is not logged twice.
        last_log = None

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for iteration in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            episodes_completed = 0
            timer.reset()
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    if self.observation_logger is not None:
                        self.observation_logger.record(obs)
                    timer.start("act")
                    actions = self.alg.act(obs, critic_obs)
                    timer.stop("act")
                    timer.start("env")
                    obs, rewards, dones, infos = self.env.step(actions)
                    timer.stop("env")
                    timer.start("post")
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
                    done_mask = dones.view(-1) > 0
                    episode_done[step].copy_(done_mask)
                    torch.where(done_mask, cur_reward_sum, torch.zeros_like(cur_reward_sum), out=episode_reward[step])
                    torch.where(done_mask, cur_episode_length, torch.zeros_like(cur_episode_length), out=episode_length[step])
                    cur_reward_sum.masked_fill_(done_mask, 0.0)
                    cur_episode_length.masked_fill_(done_mask, 0.0)
                    # A reset replaces extras["log"] with a fresh dictionary, so object
                    # identity reports "episodes finished this step" without a sync.
                    log_dict = infos.get("log") if isinstance(infos, dict) else None
                    if log_dict is not None and log_dict is not last_log:
                        last_log = log_dict
                        self._append_episode_log(ep_infos, infos)
                    timer.stop("post")

                completed_mask = episode_done.cpu()
                episodes_completed = int(completed_mask.sum())
                if episodes_completed:
                    rewbuffer.extend(episode_reward.cpu()[completed_mask].tolist())
                    lenbuffer.extend(episode_length.cpu()[completed_mask].tolist())
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
                float(cur_reward_sum.mean().item())
                if not rewbuffer and self.latest_mean_reward is None else 0.0,
                float(cur_episode_length.mean().item())
                if not lenbuffer and self.latest_mean_episode_length is None else 0.0,
                timer.finish(),
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
        profile_times: dict[str, float] | None = None,
    ):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time
        fps = int(self.num_steps_per_env * self.env.num_envs / max(collection_time + learn_time, 1e-6))
        mean_reward = statistics.mean(rewbuffer) if rewbuffer else None
        mean_episode_length = statistics.mean(lenbuffer) if lenbuffer else None
        mean_noise_std = self.alg.actor_critic.std.mean().item()
        self._push_episode_metric_batch(self._episode_metric_batch(ep_infos))
        episode_metrics = self._episode_metric_averages()
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
            self.writer.add_scalar("Train/episode_metric_batches", len(self.episode_metric_window), iteration)
            for name, value in (profile_times or {}).items():
                self.writer.add_scalar(f"Perf/{name}_time", value, iteration)
            for key, value in display_episode_metrics:
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
            f"{'Episode metric history batches:':>{pad}} {len(self.episode_metric_window)}\n"
        )
        for key, value in display_episode_metrics:
            log_string += f"{key:>{pad}}: {value:.4f}\n"
        for name, value in (profile_times or {}).items():
            log_string += f"{('Profile ' + name + ' time:'):>{pad}} {value * 1e3:.3f}ms\n"
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {self._format_duration(self.tot_time)}\n"
            f"{'ETA:':>{pad}} {self._format_duration(self.tot_time / max(iteration + 1, 1) * (total_iterations - iteration - 1))}\n"
        )
        print(log_string, flush=True)

    @staticmethod
    def _append_episode_log(ep_infos: list[dict], infos: dict):
        ep_info = HIMOnPolicyRunner._extract_log_dict(infos)
        if isinstance(ep_info, dict) and ep_info:
            # Snapshot on the producer device. Moving each reset's individual
            # scalar to CPU would synchronize CUDA repeatedly inside rollout.
            ep_infos.append({
                key: value.detach().to(dtype=torch.float32).clone()
                if isinstance(value, torch.Tensor) else value
                for key, value in ep_info.items()
            })

    @staticmethod
    def _extract_log_dict(infos: dict | None) -> dict | None:
        if not isinstance(infos, dict):
            return None
        ep_info = infos.get("log")
        if not isinstance(ep_info, dict):
            return None
        return {key: value for key, value in ep_info.items() if key.startswith(("Episode", "Metrics/"))}

    def _update_episode_metric_cache(self, ep_info: dict | None):
        if not ep_info:
            return
        for key, value in self._aggregate_episode_infos([ep_info]):
            self.latest_episode_metrics[key] = value

    def _aggregate_episode_infos(
        self, ep_infos: list[dict[str, torch.Tensor | float | int]]
    ) -> list[tuple[str, float]]:
        batch = self._episode_metric_batch(ep_infos)
        return self._order_metric_items(
            [(key, total / count) for key, (total, count) in batch.items() if count]
        )

    @staticmethod
    def _episode_metric_batch(
        ep_infos: list[dict[str, torch.Tensor | float | int]]
    ) -> dict[str, tuple[float, int]]:
        """Per-key (sum, count) of episode metrics, read back with one transfer."""
        if not ep_infos:
            return {}
        grouped: dict[str, list] = {}
        for ep_info in ep_infos:
            for key, value in ep_info.items():
                grouped.setdefault(key, []).append(value)

        pieces: list[torch.Tensor] = []
        spans: dict[str, tuple[int, int]] = {}
        cursor = 0
        device = next(
            (value.device for values in grouped.values() for value in values
             if isinstance(value, torch.Tensor) and value.is_cuda),
            torch.device("cpu"),
        )
        for key, values in grouped.items():
            flat = []
            for value in values:
                if isinstance(value, torch.Tensor):
                    value = value.detach()
                    if value.dtype != torch.float32:
                        value = value.to(torch.float32)
                    flat.append(value.reshape(-1).to(device))
                else:
                    flat.append(torch.tensor([float(value)], dtype=torch.float32, device=device))
            merged = torch.cat(flat) if len(flat) > 1 else flat[0]
            pieces.append(merged)
            spans[key] = (cursor, cursor + merged.numel())
            cursor += merged.numel()
        combined = (torch.cat(pieces) if len(pieces) > 1 else pieces[0]).cpu()
        return {
            key: (float(combined[start:stop].sum()), stop - start)
            for key, (start, stop) in spans.items()
        }

    def _push_episode_metric_batch(self, batch: dict[str, tuple[float, int]]):
        """Slide the metric window forward while keeping running per-key totals."""
        if not batch:
            return
        window = self.episode_metric_window
        if len(window) == window.maxlen:
            for key, (total, count) in window[0].items():
                running = self._episode_metric_totals.get(key)
                if running is None:
                    continue
                running[0] -= total
                running[1] -= count
                if running[1] <= 0:
                    self._episode_metric_totals.pop(key, None)
        window.append(batch)
        for key, (total, count) in batch.items():
            running = self._episode_metric_totals.setdefault(key, [0.0, 0])
            running[0] += total
            running[1] += count

    def _episode_metric_averages(self) -> list[tuple[str, float]]:
        return self._order_metric_items(
            [(key, total / count) for key, (total, count) in self._episode_metric_totals.items() if count]
        )

    @staticmethod
    def _order_metric_items(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
        priority_keys = (
            "Metrics/base_velocity/error_vel_x",
            "Metrics/base_velocity/error_vel_y",
            "Metrics/base_velocity/error_vel_z",
            "Metrics/base_velocity/error_vel_dir",
            "Metrics/base_velocity/error_vel_xy",
            "Metrics/base_velocity/error_vel_yaw",
        )
        priority = {key: index for index, key in enumerate(priority_keys)}
        return sorted(items, key=lambda item: priority.get(item[0], len(priority)))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(int(seconds), 0)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class _SectionTimer:
    """Split collection time per rollout section with CUDA events.

    Disabled unless ``ROBOT_LAB_PROFILE=1``; the rollout then reports how much of
    each iteration the policy, the environment step and post-processing take.
    """

    SECTIONS = ("act", "env", "post")

    def __init__(self, device):
        self.enabled = (
            bool(os.environ.get("ROBOT_LAB_PROFILE"))
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
        )
        self._pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            name: [] for name in self.SECTIONS
        }
        self._cursor: dict[str, int] = {name: 0 for name in self.SECTIONS}

    def reset(self):
        if not self.enabled:
            return
        self._cursor = {name: 0 for name in self.SECTIONS}

    def start(self, name: str):
        if not self.enabled:
            return
        index = self._cursor[name]
        pairs = self._pairs[name]
        if index == len(pairs):
            pairs.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
        pairs[index][0].record()

    def stop(self, name: str):
        if not self.enabled:
            return
        pairs = self._pairs[name]
        pairs[self._cursor[name]][1].record()
        self._cursor[name] += 1

    def finish(self) -> dict[str, float] | None:
        if not self.enabled:
            return None
        torch.cuda.synchronize()
        totals = {}
        for name in self.SECTIONS:
            total_ms = 0.0
            for start, stop in self._pairs[name][: self._cursor[name]]:
                total_ms += start.elapsed_time(stop)
            totals[name] = total_ms * 1e-3
        return totals
