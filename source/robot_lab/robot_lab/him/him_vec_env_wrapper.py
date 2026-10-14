# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import partial
import re

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class HIMVecEnvWrapper:
    def __init__(
        self,
        env,
        clip_actions: float,
        num_one_step_obs: int | None = None,
        reorder_history: bool = True,
    ):
        self._raw_env = env
        self._wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

        self.num_envs = int(self._wrapped_env.num_envs)
        self.num_actions = int(self._wrapped_env.num_actions)
        self.device = self._wrapped_env.device
        self.max_episode_length = self._wrapped_env.max_episode_length
        # Isaac Lab flattens each term's history independently; HIM expects full
        # observation frames concatenated from newest to oldest.
        self.reorder_history = bool(reorder_history)

        self.policy_term_slices, self.policy_one_step_term_slices = self._build_group_layout("policy")
        self.critic_term_slices, _ = self._build_group_layout("critic")
        self.policy_term_names = tuple(self.policy_one_step_term_slices.keys())
        self.critic_term_names = tuple(self.critic_term_slices.keys())
        self.teacher_target_terms = tuple(name for name in self.policy_term_names if name != "velocity_commands") + (
            "base_lin_vel",
        )

        inferred_one_step_obs = self._infer_num_one_step_obs()
        requested_one_step_obs = None if num_one_step_obs is None else int(num_one_step_obs)
        if inferred_one_step_obs is None and requested_one_step_obs is None:
            raise ValueError("Unable to infer HIM one-step actor observation dimension from the policy group.")
        if (
            inferred_one_step_obs is not None
            and requested_one_step_obs is not None
            and inferred_one_step_obs != requested_one_step_obs
        ):
            print(
                "[WARN] HIM policy one-step observation dim mismatch: "
                f"configured={requested_one_step_obs}, inferred={inferred_one_step_obs}. "
                "Using inferred value."
            )
        self.num_one_step_obs = (
            inferred_one_step_obs if inferred_one_step_obs is not None else requested_one_step_obs
        )
        if self.num_one_step_obs is None:
            raise ValueError("HIM one-step actor observation dimension is not available.")
        self.num_one_step_obs = int(self.num_one_step_obs)

        self.default_estimator_vel_slice = self._term_slice(self.critic_term_slices, "base_lin_vel")
        if self.default_estimator_vel_slice is None or (
            self.default_estimator_vel_slice[1] - self.default_estimator_vel_slice[0] != 3
        ):
            raise ValueError("HIM critic base_lin_vel must be an exactly 3-D observation term.")
        self.default_estimator_target_slices = self._infer_teacher_target_slices()
        self._terminal_critic_obs = None
        self._terminal_critic_valid = None
        self._last_dones = None
        self._capture_terminal = False

        actor_obs, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._cache_observations(actor_obs, critic_obs)
        self.num_obs = int(actor_obs.shape[-1])
        self.num_privileged_obs = None if critic_obs is None else int(critic_obs.shape[-1])
        self._validate_observation_layout()

        print(
            "[INFO] HIM observation manager dimensions: "
            f"policy_terms={self._term_dim_report('policy')}, "
            f"critic_terms={self._term_dim_report('critic')}; "
            f"policy_total={self.num_obs}, critic_total={self.num_privileged_obs}, "
            f"policy_one_step={self.num_one_step_obs}"
        )

        command_slice = self.policy_one_step_term_slices.get("velocity_commands")
        if command_slice is None or command_slice.stop - command_slice.start != 3:
            raise ValueError(
                "HIM requires velocity_commands to be exactly 3-D (vx, vy, yaw), "
                "without heading."
            )
        if self.num_privileged_obs != self.num_one_step_obs + 3:
            raise ValueError("HIM critic must contain the policy frame plus 3-D base_lin_vel.")
        if "actions" not in self.policy_one_step_term_slices:
            raise ValueError("HIM policy observation history must include the actions term.")

        if self.num_obs % self.num_one_step_obs != 0:
            raise ValueError(
                "Policy observation history is incompatible with HIM. "
                f"actor_obs_dim={self.num_obs}, one_step_dim={self.num_one_step_obs}."
            )
        self.history_length = self.num_obs // self.num_one_step_obs
        self._install_terminal_capture()

    def _validate_observation_layout(self):
        if not self.policy_term_slices or not self.policy_one_step_term_slices:
            raise ValueError("HIM requires complete policy observation-manager term metadata.")
        if not self.critic_term_slices:
            raise ValueError("HIM requires complete critic observation-manager term metadata.")

        policy_width = max(term_slice.stop for term_slice in self.policy_term_slices.values())
        critic_width = max(term_slice.stop for term_slice in self.critic_term_slices.values())
        if policy_width != self.num_obs or critic_width != self.num_privileged_obs:
            raise ValueError(
                "HIM observation-manager layout does not cover the returned observations: "
                f"policy={policy_width}/{self.num_obs}, critic={critic_width}/{self.num_privileged_obs}."
            )
        history_lengths = set()
        term_cfgs = self.unwrapped.observation_manager._group_obs_term_cfgs["policy"]
        for (name, term_slice), term_cfg in zip(self.policy_term_slices.items(), term_cfgs, strict=True):
            one_step_slice = self.policy_one_step_term_slices[name]
            total_dim = term_slice.stop - term_slice.start
            base_dim = one_step_slice.stop - one_step_slice.start
            if base_dim <= 0 or total_dim % base_dim != 0:
                raise ValueError(f"Invalid HIM policy history layout for term '{name}': {total_dim}/{base_dim}.")
            term_history = total_dim // base_dim
            configured_history = max(1, int(getattr(term_cfg, "history_length", 0) or 0))
            if term_history != configured_history:
                raise ValueError(
                    f"HIM policy term '{name}' history differs from observation-manager configuration: "
                    f"observed={term_history}, configured={configured_history}."
                )
            history_lengths.add(term_history)
        if len(history_lengths) != 1 or history_lengths != {self.num_obs // self.num_one_step_obs}:
            raise ValueError(f"HIM requires a uniform policy history length, got {sorted(history_lengths)}.")

    @property
    def unwrapped(self):
        return self._wrapped_env.unwrapped

    def reset(self):
        obs_dict, extras = self._wrapped_env.reset()
        actor_obs, critic_obs = self._split_observations(obs_dict)
        self._cache_observations(actor_obs, critic_obs)
        self._last_dones = None
        extras = dict(extras)
        if critic_obs is not None:
            extras.setdefault("critic_obs", critic_obs)
        return actor_obs, extras

    def step(self, actions: torch.Tensor):
        self._terminal_critic_obs = None
        self._terminal_critic_valid = None
        self._capture_terminal = True
        try:
            obs_dict, rewards, dones, extras = self._wrapped_env.step(actions)
        finally:
            self._capture_terminal = False
        actor_obs, critic_obs = self._split_observations(obs_dict)
        self._cache_observations(actor_obs, critic_obs)
        self._last_dones = dones.bool()
        extras = dict(extras)
        if critic_obs is not None:
            extras.setdefault("critic_obs", critic_obs)
        return actor_obs, rewards, dones, extras

    def get_estimator_next_critic_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """获取用于estimator训练的next observation和有效性mask。

        HIM修复：保留terminal transition的监督，不通过dones排除样本。
        关键改变：对于done环境，使用capture的terminal观测，并标记为有效。
        """
        next_obs = self.get_critic_observations().clone()
        valid = torch.ones(self.num_envs, device=next_obs.device, dtype=torch.bool)

        # 如果有terminal观测被捕捉（环境在reset前的观测），用它替换
        if self._terminal_critic_valid is not None:
            next_obs[self._terminal_critic_valid] = self._terminal_critic_obs[self._terminal_critic_valid]

        # ✅ 关键修改：保留所有terminal样本（包括done和not done的）
        # 旧逻辑会排除所有done的样本，现在改为全部保留
        # 即使done=True，只要被capture到terminal观测，也作为有效样本参与estimator训练
        if self._last_dones is not None and self._terminal_critic_valid is not None:
            # 对于done但被成功capture的环境，标记为有效
            # valid保持为True，让estimator学习terminal supervision
            pass  # 所有样本都保留valid=True，不主动排除任何done的样本

        return next_obs, valid

    def _install_terminal_capture(self):
        env = self.unwrapped
        self._original_reset_idx = env._reset_idx

        def capture_then_reset(env_ids):
            if not self._capture_terminal:
                return self._original_reset_idx(env_ids)
            # Isaac Lab resets done environments before returning from step.
            terminal = env.observation_manager.compute_group("critic", update_history=False)
            terminal = self._flatten_group_tensor(terminal)
            if self._terminal_critic_obs is None:
                self._terminal_critic_obs = torch.zeros_like(terminal)
                self._terminal_critic_valid = torch.zeros(self.num_envs, device=terminal.device, dtype=torch.bool)
            self._terminal_critic_obs[env_ids] = terminal[env_ids]
            self._terminal_critic_valid[env_ids] = True
            return self._original_reset_idx(env_ids)

        env._reset_idx = capture_then_reset

    def get_observations(self) -> torch.Tensor:
        actor_obs, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._cache_observations(actor_obs, critic_obs)
        return actor_obs

    def get_privileged_observations(self) -> torch.Tensor | None:
        if self._latest_critic_obs is not None:
            return self._latest_critic_obs
        _, critic_obs = self._split_observations(self._wrapped_env.get_observations())
        self._latest_critic_obs = critic_obs
        return critic_obs

    def get_critic_observations(self) -> torch.Tensor:
        critic_obs = self.get_privileged_observations()
        if critic_obs is None:
            raise RuntimeError(
                "HIM requires a dedicated 'critic' observation group; "
                "refusing to fall back to actor observations."
            )
        return critic_obs

    def _term_dim_report(self, group_name: str) -> list[dict[str, object]]:
        layout = self.policy_one_step_term_slices if group_name == "policy" else self.critic_term_slices
        return [
            {"name": name, "start": sl.start, "stop": sl.stop, "dim": sl.stop - sl.start}
            for name, sl in layout.items()
        ]

    def get_him_deployment_metadata(self) -> dict[str, object]:
        one_step_obs_dim = self.num_one_step_obs
        history_length = self.history_length
        return {
            "actor_obs_dim": int(self.num_obs),
            "critic_obs_dim": None if self.num_privileged_obs is None else int(self.num_privileged_obs),
            "one_step_obs_dim": one_step_obs_dim,
            "history_length": history_length,
            "history_indices_latest_to_oldest": list(range(history_length)),
            "history_layout": "time_major_current_to_oldest",
            "policy_term_names": list(self.policy_term_names),
            "policy_terms": [
                {
                    "name": term_name,
                    "dim": int(term_slice.stop - term_slice.start),
                    "start": int(term_slice.start),
                    "stop": int(term_slice.stop),
                }
                for term_name, term_slice in self.policy_one_step_term_slices.items()
            ],
            "critic_terms": self._term_dim_report("critic"),
            "critic_term_names": list(self.critic_term_names),
            "default_estimator_vel_slice": self.default_estimator_vel_slice,
            "default_estimator_target_slices": self.default_estimator_target_slices,
            "teacher_target_terms": list(self.teacher_target_terms),
            "observation_contract": self.get_observation_contract(),
        }

    @staticmethod
    def _contract_value(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, slice):
            return {"start": value.start, "stop": value.stop, "step": value.step}
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, Mapping):
            return {str(key): HIMVecEnvWrapper._contract_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (tuple, list)):
            return [HIMVecEnvWrapper._contract_value(item) for item in value]
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": {field.name: HIMVecEnvWrapper._contract_value(getattr(value, field.name)) for field in fields(value)},
            }
        if isinstance(value, partial):
            return {
                "func": HIMVecEnvWrapper._contract_value(value.func),
                "args": HIMVecEnvWrapper._contract_value(value.args),
                "keywords": HIMVecEnvWrapper._contract_value(value.keywords),
            }
        if callable(value):
            func = getattr(value, "__func__", value)
            if hasattr(func, "__qualname__"):
                return f"{func.__module__}.{func.__qualname__}"
            return f"{type(func).__module__}.{type(func).__qualname__}"
        # Older Isaac Lab config objects may not be dataclasses.
        return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))

    def get_observation_contract(self, legacy: bool = False) -> dict[str, object]:
        manager = self.unwrapped.observation_manager
        groups = {}
        for group in ("policy", "critic"):
            terms = []
            for name, cfg in zip(manager._group_obs_term_names[group], manager._group_obs_term_cfgs[group]):
                noise = cfg.noise
                term = {
                    "name": name,
                    "dim": self._shape_numel(manager.group_obs_term_dim[group][len(terms)]),
                    "scale": str(cfg.scale) if legacy else self._contract_value(cfg.scale),
                    "clip": str(cfg.clip) if legacy else self._contract_value(cfg.clip),
                    "noise": (None if noise is None else str(noise)) if legacy else self._contract_value(noise),
                    "params": str(cfg.params) if legacy else self._contract_value(cfg.params),
                }
                if not legacy:
                    term["func"] = self._contract_value(cfg.func)
                    term["modifiers"] = self._contract_value(cfg.modifiers)
                terms.append(term)
            groups[group] = terms
        contract = {
            "groups": groups,
            "history_length": self.history_length,
            "history_layout": "time_major_current_to_oldest",
            "policy_corruption": bool(self.unwrapped.cfg.observations.policy.enable_corruption),
            "critic_corruption": bool(self.unwrapped.cfg.observations.critic.enable_corruption),
            "teacher_target_terms": list(self.teacher_target_terms),
            "teacher_target_slices": [list(item) for item in self.default_estimator_target_slices],
            "velocity_slice": list(self.default_estimator_vel_slice),
        }
        if not legacy:
            contract["format_version"] = 2
        return contract

    def _cache_observations(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor | None):
        self._latest_actor_obs = actor_obs
        self._latest_critic_obs = critic_obs

    def _split_observations(self, observations) -> tuple[torch.Tensor, torch.Tensor | None]:
        actor_obs = self._extract_group_observations(observations, "policy")
        if actor_obs is None:
            raise KeyError("HIM requires a 'policy' observation group.")
        if self.reorder_history:
            actor_obs = self._reorder_policy_history(actor_obs)

        critic_obs = self._extract_group_observations(observations, "critic")
        if critic_obs is None:
            critic_obs = self._compute_group("critic")
        if critic_obs is None:
            raise KeyError("HIM requires a dedicated 'critic' observation group.")
        return actor_obs, critic_obs

    def _reorder_policy_history(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if not self.policy_term_slices or not self.policy_one_step_term_slices:
            raise ValueError("HIM requires policy observation-manager term metadata to reorder history.")
        policy_width = max(term_slice.stop for term_slice in self.policy_term_slices.values())
        if policy_width != actor_obs.shape[-1]:
            raise ValueError(
                f"HIM policy observation layout width {policy_width} does not match actor width {actor_obs.shape[-1]}."
            )

        term_histories = []
        history_lengths = set()
        for term_name, term_slice in self.policy_term_slices.items():
            one_step_slice = self.policy_one_step_term_slices[term_name]
            base_dim = one_step_slice.stop - one_step_slice.start
            total_dim = term_slice.stop - term_slice.start
            if base_dim <= 0 or total_dim % base_dim != 0:
                raise ValueError(
                    f"Invalid HIM policy history layout for term '{term_name}': "
                    f"total_dim={total_dim}, base_dim={base_dim}."
                )
            history_length = total_dim // base_dim
            history_lengths.add(history_length)
            term_histories.append(actor_obs[:, term_slice].reshape(actor_obs.shape[0], history_length, base_dim))

        if len(history_lengths) != 1:
            raise ValueError(f"HIM requires a uniform policy history length, got {sorted(history_lengths)}.")
        history_length = history_lengths.pop()
        if history_length <= 1:
            return actor_obs

        frames = []
        for history_index in range(history_length - 1, -1, -1):
            frames.append(torch.cat([term_history[:, history_index] for term_history in term_histories], dim=-1))
        return torch.cat(frames, dim=-1)

    def _extract_group_observations(self, observations, group_name: str) -> torch.Tensor | None:
        if observations is None:
            return None

        group_obs = None
        try:
            group_obs = observations[group_name]
        except Exception:
            if isinstance(observations, Mapping):
                group_obs = observations.get(group_name)

        if group_obs is None:
            return None
        return self._flatten_group_tensor(group_obs)

    def _compute_group(self, group_name: str) -> torch.Tensor | None:
        observation_manager = getattr(self.unwrapped, "observation_manager", None)
        if observation_manager is None:
            return None
        try:
            group_obs = observation_manager.compute_group(group_name=group_name)
        except Exception:
            return None
        return self._flatten_group_tensor(group_obs)

    @staticmethod
    def _flatten_group_tensor(group_obs) -> torch.Tensor:
        if not isinstance(group_obs, torch.Tensor):
            group_obs = torch.as_tensor(group_obs)
        if group_obs.ndim == 1:
            group_obs = group_obs.unsqueeze(0)
        elif group_obs.ndim > 2:
            group_obs = group_obs.flatten(start_dim=1)
        return group_obs

    def _infer_num_one_step_obs(self) -> int | None:
        if not self.policy_one_step_term_slices:
            return None
        return max(term_slice.stop for term_slice in self.policy_one_step_term_slices.values())

    def _build_group_layout(self, group_name: str) -> tuple[dict[str, slice], dict[str, slice]]:
        observation_manager = getattr(self.unwrapped, "observation_manager", None)
        if observation_manager is None:
            return {}, {}

        term_names = getattr(observation_manager, "_group_obs_term_names", {}).get(group_name, [])
        term_cfgs = getattr(observation_manager, "_group_obs_term_cfgs", {}).get(group_name, [])
        term_dims = observation_manager.group_obs_term_dim.get(group_name, [])
        if not term_names or not term_cfgs or not term_dims:
            raise ValueError(f"HIM {group_name} observation-manager term metadata is missing.")
        if len(term_names) != len(term_cfgs) or len(term_names) != len(term_dims):
            raise ValueError(
                f"HIM {group_name} observation-manager term metadata is incomplete: "
                f"names={len(term_names)}, configs={len(term_cfgs)}, dims={len(term_dims)}."
            )

        group_layout: dict[str, slice] = {}
        one_step_layout: dict[str, slice] = {}
        group_cursor = 0
        one_step_cursor = 0

        for term_name, term_cfg, term_dim in zip(term_names, term_cfgs, term_dims, strict=True):
            total_dim = self._shape_numel(term_dim)
            base_dim = self._infer_base_term_dim(term_cfg, total_dim)
            if term_name in group_layout or total_dim <= 0 or base_dim <= 0:
                raise ValueError(f"Invalid HIM {group_name} observation term '{term_name}': {total_dim}/{base_dim}.")
            group_layout[term_name] = slice(group_cursor, group_cursor + total_dim)
            one_step_layout[term_name] = slice(one_step_cursor, one_step_cursor + base_dim)
            group_cursor += total_dim
            one_step_cursor += base_dim

        return group_layout, one_step_layout

    def _infer_base_term_dim(self, term_cfg, total_dim: int) -> int:
        try:
            sample = term_cfg.func(self.unwrapped, **term_cfg.params)
            sample = self._flatten_group_tensor(sample)
            return int(sample.shape[-1])
        except Exception:
            history_length = int(getattr(term_cfg, "history_length", 0) or 0)
            if history_length <= 1 or total_dim % history_length != 0:
                return total_dim
            return total_dim // history_length

    def _infer_teacher_target_slices(self) -> tuple[tuple[int, int], ...]:
        missing = [name for name in self.teacher_target_terms if name not in self.critic_term_slices]
        if missing:
            raise ValueError(f"HIM teacher target terms missing from critic observations: {missing}")
        slices = tuple(self._term_slice(self.critic_term_slices, name) for name in self.teacher_target_terms)
        target_dim = sum(stop - start for start, stop in slices)
        if target_dim != self.num_one_step_obs:
            raise ValueError(f"HIM teacher target must be {self.num_one_step_obs}-D, got {target_dim}-D.")
        return slices

    @staticmethod
    def _term_slice(layout: dict[str, slice], term_name: str) -> tuple[int, int] | None:
        term_slice = layout.get(term_name)
        if term_slice is None:
            return None
        return (term_slice.start, term_slice.stop)

    @staticmethod
    def _shape_numel(shape) -> int:
        if isinstance(shape, int):
            return int(shape)
        if len(shape) == 0:
            return 1
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel

    def __getattr__(self, name: str):
        return getattr(self._wrapped_env, name)

    def close(self):
        self.unwrapped._reset_idx = self._original_reset_idx
        return self._wrapped_env.close()
