# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import ATDogDog2RoughEnvCfg
from .periodic_gait import configure_gait_rewards


ATDOG2_LOW_OBSTACLE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.85),
        "low_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.01, 0.025),
            platform_width=2.0,
            holes=False,
        ),
    },
)


@configclass
class ATDogDog2FlatEnvCfg(ATDogDog2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Short flat episodes: 250 control steps at 50 Hz.
        self.episode_length_s = 5.0

        # Keep flat-task resets very close to the nominal standing state.
        # Small pose perturbations help robustness without spawning the robot
        # in obviously unstable configurations.
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.1, 0.1),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Keep the ray caster for privileged critic terrain observations.
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        configure_gait_rewards(self, swing_height=0.07)
        self.rewards.standing_support.weight = 6.0
        self.rewards.standing_yaw_drift = RewTerm(
            func=mdp.standing_yaw_drift,
            weight=-4.0,
            params={"command_name": "base_velocity", "std": 0.15},
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ATDogDog2FlatEnvCfg":
            self.events.randomize_rigid_body_mass_base = None
            self.events.randomize_com_positions = None
            self.events.randomize_actuator_gains = None
            self.disable_zero_weight_rewards()


@configclass
class ATDogDog2FlatDomainRandEnvCfg(ATDogDog2FlatEnvCfg):
    """Low-obstacle robustness stage between flat locomotion and stairs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ATDOG2_LOW_OBSTACLE_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.1, 0.1)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.002, 0.002)
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.98, 1.02)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.98, 1.02)
        self.events.randomize_push_robot.interval_range_s = (15.0, 20.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        configure_gait_rewards(self, swing_height=0.08)
        self.rewards.standing_support.weight = 6.0
        self.disable_zero_weight_rewards()
