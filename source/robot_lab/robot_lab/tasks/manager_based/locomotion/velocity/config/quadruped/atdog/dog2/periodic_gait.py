"""ATDog2 fixed trot clock, unchanged policy layout, and phase-aware rewards."""

import copy

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg


FOOT_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
# Fixed foot-joint origins in dog2.urdf, in calf link frames. The merged
# collision mesh is a disk bounded by radius 0.0384 m and half-width 0.014 m.
FOOT_OFFSETS = ((0.22, 0.0, 0.001), (0.22, 0.0, 0.0), (0.22, 0.0, 0.0), (0.22, 0.0, 0.0))


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
    command_params = copy.deepcopy(vars(env_cfg.commands.base_velocity))
    command_params.pop("class_type")
    env_cfg.commands.base_velocity = mdp.PhaseVelocityCommandCfg(**command_params)
    env_cfg.him_clean_critic = True


def configure_gait_rewards(env_cfg, swing_height: float = 0.10):
    """Apply after terrain reward overrides, before zero-weight cleanup.

    Keep the 47-D policy contract; all gait terms consume the command clock.
    Penalties return nonnegative violations and therefore use negative weights.
    """
    for name in (
        "periodic_foot_lift", "feet_gait", "diagonal_trot_contact_pattern",
        "feet_height", "feet_height_body", "feet_height_body_exp",
        "feet_air_time", "feet_air_time_variance", "feet_contact_without_cmd",
    ):
        setattr(env_cfg.rewards, name, None)
    env_cfg.rewards.gait_contact = RewTerm(
        func=mdp.GaitContactPenalty,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "foot_names": FOOT_NAMES,
            "command_name": "base_velocity",
        },
    )
    env_cfg.rewards.swing_clearance = RewTerm(
        func=mdp.SwingClearancePenalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_NAMES), preserve_order=True),
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            "command_name": "base_velocity",
            "foot_offsets": FOOT_OFFSETS,
            "foot_radius": 0.0384,
            "foot_half_width": 0.014,
            "height": swing_height,
            "std": 0.10,
        },
    )
    env_cfg.rewards.standing_support = RewTerm(
        func=mdp.standing_support,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_NAMES), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_NAMES), preserve_order=True),
            "foot_offsets": FOOT_OFFSETS,
        },
    )
    env_cfg.rewards.standing_joint_motion = RewTerm(
        func=mdp.standing_joint_motion,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=env_cfg.joint_names, preserve_order=True),
        },
    )
    env_cfg.rewards.feet_slide = RewTerm(
        func=mdp.gait_foot_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_NAMES), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_NAMES), preserve_order=True),
            "foot_offsets": FOOT_OFFSETS,
        },
    )
