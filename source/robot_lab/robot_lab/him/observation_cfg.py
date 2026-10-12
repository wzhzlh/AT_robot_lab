# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

from isaaclab.utils.noise import AdditiveUniformNoiseCfg


def align_him_critic_observations(env_cfg, agent_cfg):
    if agent_cfg.class_name != "HIMOnPolicyRunner":
        return
    policy = env_cfg.observations.policy
    critic = env_cfg.observations.critic
    for name, term in vars(policy).items():
        if name == "base_lin_vel" or term is None or not hasattr(term, "func"):
            continue
        if hasattr(critic, name):
            setattr(critic, name, copy.deepcopy(term))
    critic.base_lin_vel.scale = 2.0
    critic.base_lin_vel.noise = AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1)
    critic.enable_corruption = True
