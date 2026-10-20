"""Shared ATDog2 gait clock and periodic foot lift configuration."""

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg


GAIT_CYCLE_TIME = 0.6


def configure_periodic_gait(env_cfg):
    env_cfg.observations.policy.phase = ObsTerm(func=mdp.phase, params={"cycle_time": GAIT_CYCLE_TIME})
    env_cfg.observations.critic.phase = ObsTerm(func=mdp.phase, params={"cycle_time": GAIT_CYCLE_TIME})
    env_cfg.rewards.periodic_foot_lift = RewTerm(
        func=mdp.PeriodicFootLiftReward,
        weight=40.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_names": ("FL_calf", "FR_calf", "RL_calf", "RR_calf"),
            "cycle_time": GAIT_CYCLE_TIME,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )


def configure_body_height_reward(env_cfg):
    foot_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    env_cfg.rewards.feet_height_body_exp = RewTerm(
        func=mdp.feet_height_body_exp,
        weight=100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=foot_names, preserve_order=True),
            "target_height": env_cfg.rewards.feet_height_body.params["target_height"],
            "std": 0.05,
        },
    )
