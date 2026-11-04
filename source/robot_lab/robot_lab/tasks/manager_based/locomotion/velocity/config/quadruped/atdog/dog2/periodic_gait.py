"""Shared ATDog2 gait clock and periodic foot lift configuration."""

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg


GAIT_CYCLE_TIME = 0.5


@configclass
class ATDog2PolicyCfg(ObsGroup):
    phase = ObsTerm(func=mdp.stairs_phase, params={"command_name": "base_velocity"})
    base_lin_vel = ObservationsCfg.PolicyCfg().base_lin_vel
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObservationsCfg.PolicyCfg().actions
    height_scan = ObservationsCfg.PolicyCfg().height_scan

    enable_corruption = True
    concatenate_terms = True


def configure_periodic_gait(env_cfg):
    env_cfg.observations.policy = ATDog2PolicyCfg()
    env_cfg.observations.critic.phase = ObsTerm(func=mdp.stairs_phase, params={"command_name": "base_velocity"})
    env_cfg.observations.policy.velocity_commands.scale = (1.0, 1.0, 0.25)
    env_cfg.observations.critic.velocity_commands.scale = (1.0, 1.0, 0.25)
    env_cfg.commands.base_velocity.class_type = mdp.PhaseVelocityCommand
    env_cfg.him_clean_critic = True
    configure_periodic_foot_lift_reward(env_cfg)


def configure_periodic_foot_lift_reward(env_cfg):
    env_cfg.rewards.periodic_foot_lift = RewTerm(
        func=mdp.PeriodicFootLiftReward,
        weight=50.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_names": ("FL_calf", "FR_calf", "RL_calf", "RR_calf"),
            "cycle_time": GAIT_CYCLE_TIME,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "phase_from_command": True,
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
            "std": 0.08,
        },
    )
