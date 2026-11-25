# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-ATDog-DogW1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ATDogDogW1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogW1FlatPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogW1FlatHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogW1FlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-ATDog-DogW1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ATDogDogW1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogW1RoughPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogW1RoughHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogW1RoughTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Slope-ATDog-DogW1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slope_env_cfg:ATDogDogW1SlopeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogW1SlopePPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogW1SlopeHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogW1SlopeTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Sand-ATDog-DogW1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sand_env_cfg:ATDogDogW1SandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogW1SandPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogW1SandHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogW1SandTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Stairs-ATDog-DogW1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:ATDogDogW1StairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogW1StairsPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogW1StairsHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogW1StairsTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Tires-ATDog-DogW1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tires_env_cfg:ATDogDogW1TiresEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDogDogW1TiresPPORunnerCfg",
        "rsl_rl_him_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_cfg:ATDogDogW1TiresHIMRunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:ATDogDogW1TiresTrainerCfg",
    },
)
