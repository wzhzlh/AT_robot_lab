"""Shared Dog2 observation and phase-free foot-contact configuration."""

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from .observations import configure_dog2_observations
from .rewards import (CommandedGaitReward, configure_dog2_phase_swing_reward, standing_contact,
                      standing_joint_velocity, standing_posture, standing_base_velocity)
from . import reference_rewards


FOOT_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
FOOT_OFFSETS = ((0.22, 0.0, 0.001), (0.22, 0.0, 0.0), (0.22, 0.0, 0.0), (0.22, 0.0, 0.0))


def configure_periodic_gait(env_cfg):
    env_cfg.observations.policy.phase = None
    env_cfg.observations.critic.phase = None
    env_cfg.observations.critic.height_scan = None
    env_cfg.observations.policy.velocity_commands.scale = (1.0, 1.0, 0.25)
    env_cfg.observations.critic.velocity_commands.scale = (1.0, 1.0, 0.25)
    env_cfg.him_clean_critic = True


def configure_gait_rewards(env_cfg, swing_height=0.15):
    configure_dog2_observations(env_cfg)
    configure_dog2_phase_swing_reward(env_cfg)
    defaults = RewardsCfg()
    for name in ("is_terminated", "applied_torque_limits", "feet_air_time", "feet_air_time_variance",
                 "feet_contact_without_cmd", "feet_slide", "feet_height", "feet_height_body",
                 "feet_gait", "diagonal_trot_contact_pattern"):
        setattr(env_cfg.rewards, name, getattr(defaults, name))
    rewards = env_cfg.rewards
    rewards.is_terminated.weight = -2.0
    rewards.applied_torque_limits.weight = -0.01
    rewards.applied_torque_limits.params["asset_cfg"].joint_names = env_cfg.joint_names
    rewards.feet_air_time.weight = 5.0
    rewards.feet_air_time.params["threshold"] = 0.25
    rewards.feet_air_time.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    rewards.feet_air_time_variance.weight = -6.0
    rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    rewards.feet_contact_without_cmd.weight = 10.0
    rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    rewards.feet_slide.weight = -4.0
    rewards.feet_slide.params["sensor_cfg"].body_names = [env_cfg.foot_link_name]
    rewards.feet_slide.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    stair = "Stairs" in type(env_cfg).__name__
    rough = "Rough" in type(env_cfg).__name__ or "Flat" in type(env_cfg).__name__
    rewards.feet_height.weight = -15.0 if rough else -10.0
    rewards.feet_height.params["target_height"] = 0.11 if stair else 0.08
    rewards.feet_height.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    rewards.feet_height_body.weight = -5.0 if rough else -4.0
    rewards.feet_height_body.params["target_height"] = -0.20 if stair else -0.15
    rewards.feet_height_body.params["asset_cfg"].body_names = [env_cfg.foot_link_name]
    rewards.feet_gait.weight = 5.0 if stair else 20.0 if rough else 12.0
    rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_calf", "RR_calf"), ("FR_calf", "RL_calf"))
    rewards.diagonal_trot_contact_pattern.weight = -2.0 if rough else -1.0
    rewards.diagonal_trot_contact_pattern.params["foot_names"] = FOOT_NAMES
    for name, value in vars(defaults).items():
        if value is not None and getattr(rewards, name, None) is None:
            setattr(rewards, name, value)
    task_name = type(env_cfg).__name__.lower()
    task = next((name for name in ("stairs", "sand", "slope", "bar") if name in task_name), "rough")
    getattr(reference_rewards, f"configure_{task}_rewards")(env_cfg)
    for name in ("feet_air_time", "feet_air_time_variance", "feet_contact_without_cmd", "feet_slide",
                 "feet_stumble", "contact_forces", "feet_contact"):
        getattr(rewards, name).params["sensor_cfg"].body_names = [".*_foot"]
    rewards.feet_slide.params["asset_cfg"].body_names = [".*_foot"]
    rewards.undesired_contacts.params["sensor_cfg"].body_names = ["^(?!.*_foot$).*"]
    rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
    rewards.feet_gait.params["command_threshold"] = 0.0
    rewards.diagonal_trot_contact_pattern.params["foot_names"] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    rewards.feet_contact_without_cmd.func = standing_contact
    rewards.feet_contact_without_cmd.weight = 10.0
    rewards.feet_gait.func = CommandedGaitReward
    rewards.standing_joint_motion = RewTerm(
        func=standing_joint_velocity,
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=env_cfg.joint_names)},
    )
    rewards.standing_posture = RewTerm(
        func=standing_posture, weight=-5.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=env_cfg.joint_names)},
    )
    rewards.standing_base_velocity = RewTerm(
        func=standing_base_velocity, weight=-2.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
