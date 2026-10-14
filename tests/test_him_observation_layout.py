# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "robot_lab"))

from robot_lab.him.him_on_policy_runner import HIMOnPolicyRunner
from robot_lab.him.him_vec_env_wrapper import HIMVecEnvWrapper
from robot_lab.him.observation_cfg import align_him_critic_observations


POLICY_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
)


class FakeObservationManager:
    def __init__(self, velocity_first=True, history_length=6):
        critic_terms = (
            [("base_lin_vel", 3), *POLICY_TERMS] if velocity_first else [*POLICY_TERMS, ("base_lin_vel", 3)]
        )
        self._group_obs_term_names = {
            "policy": [name for name, _ in POLICY_TERMS],
            "critic": [name for name, _ in critic_terms],
        }
        self._group_obs_term_cfgs = {}
        self.group_obs_term_dim = {}
        for group, terms in (("policy", POLICY_TERMS), ("critic", critic_terms)):
            self._group_obs_term_cfgs[group] = [
                SimpleNamespace(
                    func=lambda env, width=dim: torch.zeros(2, width),
                    params={},
                    history_length=history_length if group == "policy" else 0,
                )
                for _, dim in terms
            ]
            self.group_obs_term_dim[group] = [
                (dim * history_length if group == "policy" else dim,) for _, dim in terms
            ]


class FakeEnv:
    def __init__(self, manager):
        self.observation_manager = manager
        self._reset_idx = lambda env_ids: None


class FakeWrappedEnv:
    def __init__(self, env):
        self.unwrapped = env
        self.num_envs = 2
        self.num_actions = 12
        self.device = "cpu"
        self.max_episode_length = 100

    def get_observations(self):
        manager = self.unwrapped.observation_manager
        return {
            group: torch.zeros(2, sum(dim[0] for dim in manager.group_obs_term_dim[group]))
            for group in ("policy", "critic")
        }


class HimLayoutTests(unittest.TestCase):
    def make_wrapper(self, velocity_first=True, history_length=6, manager=None):
        manager = manager or FakeObservationManager(velocity_first, history_length)
        env = FakeEnv(manager)
        with patch("robot_lab.him.him_vec_env_wrapper.RslRlVecEnvWrapper", return_value=FakeWrappedEnv(env)):
            return HIMVecEnvWrapper(env, clip_actions=1.0, num_one_step_obs=45)

    def test_dog2_layout_and_history_reordering(self):
        wrapper = self.make_wrapper()
        self.assertEqual((wrapper.num_obs, wrapper.num_one_step_obs, wrapper.history_length), (270, 45, 6))
        self.assertEqual(wrapper.num_privileged_obs, 48)
        self.assertEqual(wrapper.default_estimator_vel_slice, (0, 3))

        actor = torch.zeros(1, 270)
        for term_index, term_slice in enumerate(wrapper.policy_term_slices.values()):
            dim = (term_slice.stop - term_slice.start) // 6
            for frame in range(6):
                actor[:, term_slice.start + frame * dim : term_slice.start + (frame + 1) * dim] = 10 * term_index + frame
        reordered = wrapper._reorder_policy_history(actor)
        for term_index, term_slice in enumerate(wrapper.policy_one_step_term_slices.values()):
            self.assertTrue(torch.all(reordered[:, term_slice] == 10 * term_index + 5))
            oldest_slice = slice(5 * 45 + term_slice.start, 5 * 45 + term_slice.stop)
            self.assertTrue(torch.all(reordered[:, oldest_slice] == 10 * term_index))

    def test_velocity_last_is_derived_from_live_layout(self):
        wrapper = self.make_wrapper(velocity_first=False)
        self.assertEqual(wrapper.default_estimator_vel_slice, (45, 48))
        self.assertEqual(sum(stop - start for start, stop in wrapper.default_estimator_target_slices), 45)

    def test_missing_velocity_or_incomplete_metadata_fails(self):
        manager = FakeObservationManager()
        manager._group_obs_term_names["critic"][0] = "not_velocity"
        with self.assertRaisesRegex(ValueError, "base_lin_vel"):
            self.make_wrapper(manager=manager)

        manager = FakeObservationManager()
        manager._group_obs_term_cfgs["critic"].pop()
        with self.assertRaisesRegex(ValueError, "metadata is incomplete"):
            self.make_wrapper(manager=manager)

    def test_malformed_velocity_and_observation_width_fail(self):
        manager = FakeObservationManager()
        manager.group_obs_term_dim["critic"][0] = (4,)
        manager._group_obs_term_cfgs["critic"][0].func = lambda env: torch.zeros(2, 4)
        with self.assertRaisesRegex(ValueError, "exactly 3-D"):
            self.make_wrapper(manager=manager)

        wrapper = self.make_wrapper()
        with self.assertRaisesRegex(ValueError, "does not match actor width"):
            wrapper._reorder_policy_history(torch.zeros(2, 269))

    def test_history_mismatch_fails_before_network_construction(self):
        manager = FakeObservationManager()
        manager._group_obs_term_cfgs["policy"][0].history_length = 5
        with self.assertRaisesRegex(ValueError, "observation-manager configuration"):
            self.make_wrapper(manager=manager)

        wrapper = self.make_wrapper(history_length=1)
        config = {"policy": {"history_length": 6}, "algorithm": {}, "num_steps_per_env": 24, "save_interval": 100}
        with self.assertRaisesRegex(ValueError, "configured=6, actual=1"):
            HIMOnPolicyRunner(wrapper, config)

    def test_velocity_slice_never_falls_back_to_configured_actions(self):
        runner = HIMOnPolicyRunner.__new__(HIMOnPolicyRunner)
        runner.env = SimpleNamespace(default_estimator_vel_slice=None, default_estimator_target_slices=((0, 45),))
        runner.policy_cfg = {"estimator_vel_slice": (45, 48)}
        with self.assertRaisesRegex(ValueError, "live, valid estimator_vel_slice"):
            runner._resolve_estimator_slices(48)

        runner.env.default_estimator_vel_slice = (0, 3)
        runner._resolve_estimator_slices(48)
        self.assertEqual(runner.policy_cfg["estimator_vel_slice"], (0, 3))

        runner.env.default_estimator_vel_slice = (0, 4)
        with self.assertRaisesRegex(ValueError, "exactly 3-D"):
            runner._resolve_estimator_slices(48)

    def test_estimator_supervision_uses_velocity_not_actions(self):
        from robot_lab.him.him_estimator import HIMEstimator

        for velocity_slice in ((0, 3), (45, 48)):
            with self.subTest(velocity_slice=velocity_slice):
                torch.manual_seed(7)
                estimator = HIMEstimator(
                    temporal_steps=1,
                    num_one_step_obs=45,
                    vel_slice=velocity_slice,
                    target_slices=((0, 45),),
                    enc_hidden_dims=(8, 4),
                    tar_hidden_dims=(8,),
                    num_prototypes=4,
                )
                with torch.no_grad():
                    for parameter in estimator.encoder.parameters():
                        parameter.zero_()
                critic_obs = torch.zeros(2, 48)
                critic_obs[:, velocity_slice[0] : velocity_slice[1]] = 2.0
                actions_slice = (45, 48) if velocity_slice == (0, 3) else (42, 45)
                critic_obs[:, actions_slice[0] : actions_slice[1]] = 50.0
                estimation_loss, _ = estimator.update(torch.zeros(2, 45), critic_obs)
                self.assertAlmostEqual(estimation_loss, 4.0, places=5)

    def test_him_critic_noise_alignment(self):
        noisy_term = SimpleNamespace(func=lambda env: None, noise=SimpleNamespace(n_min=-0.2, n_max=0.2))
        policy = SimpleNamespace(base_lin_vel=None, base_ang_vel=noisy_term)
        original_critic = SimpleNamespace(
            base_lin_vel=SimpleNamespace(scale=1.0, noise=None), base_ang_vel=None, enable_corruption=False
        )
        cfg = SimpleNamespace(observations=SimpleNamespace(policy=policy, critic=original_critic))
        align_him_critic_observations(cfg, SimpleNamespace(class_name="HIMOnPolicyRunner"))
        self.assertTrue(original_critic.enable_corruption)
        self.assertEqual(
            (original_critic.base_lin_vel.noise.n_min, original_critic.base_lin_vel.noise.n_max), (-0.1, 0.1)
        )
        self.assertEqual(original_critic.base_ang_vel.noise.n_min, noisy_term.noise.n_min)
        self.assertIsNot(original_critic.base_ang_vel, noisy_term)

        clean_critic = copy.deepcopy(original_critic)
        clean_critic.enable_corruption = False
        cfg.observations.critic = clean_critic
        align_him_critic_observations(cfg, SimpleNamespace(class_name="OnPolicyRunner"))
        self.assertFalse(clean_critic.enable_corruption)


if __name__ == "__main__":
    unittest.main()
