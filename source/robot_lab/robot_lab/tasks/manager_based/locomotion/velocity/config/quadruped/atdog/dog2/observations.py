"""CSDog1-style proprioceptive observations adapted to Dog2 joint order."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils import configclass

from .phase_observations import CSDog1Phase, previous_action, velocity_commands
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg


def csdog1_phase_term():
    return ObsTerm(func=CSDog1Phase, params={
        "command_name": "base_velocity", "cycle_time": 0.5, "stop_cycle_time": 0.5,
        "stop_duration": 0.5, "translation_threshold": 0.05, "yaw_threshold": 0.05,
    }, clip=(-1.0, 1.0), scale=1.0)


@configclass
class CSDog1PolicyCfg(ObsGroup):
    phase = csdog1_phase_term()
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObsTerm(func=previous_action, clip=(-100.0, 100.0), scale=1.0)
    enable_corruption = True
    concatenate_terms = True


def calf_vertical_contact_forces(env, sensor_cfg: SceneEntityCfg):
    """Return vertical contact force for each selected body."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]


def configure_dog2_observations(cfg):
    """Add the shared phase frame and available Dog2 critic privileges."""
    policy = CSDog1PolicyCfg()
    policy.base_ang_vel.scale = 0.25
    policy.base_ang_vel.clip = (-400.0, 400.0)
    policy.joint_vel.scale = 0.05
    policy.joint_vel.clip = (-2000.0, 2000.0)
    policy.velocity_commands.scale = (1.0, 1.0, 0.25)
    policy.velocity_commands.func = velocity_commands
    policy.velocity_commands.clip = None
    policy.joint_pos.params["asset_cfg"].joint_names = cfg.joint_names
    policy.joint_vel.params["asset_cfg"].joint_names = cfg.joint_names
    cfg.observations.policy = policy

    critic = cfg.observations.critic
    critic.phase = csdog1_phase_term()
    critic.base_ang_vel.scale = 0.25
    critic.base_ang_vel.clip = (-400.0, 400.0)
    critic.joint_vel.scale = 0.05
    critic.joint_vel.clip = (-2000.0, 2000.0)
    critic.velocity_commands.scale = (1.0, 1.0, 0.25)
    critic.velocity_commands.func = velocity_commands
    critic.velocity_commands.clip = None
    critic.actions.func = previous_action
    critic.joint_pos.params["asset_cfg"].joint_names = cfg.joint_names
    critic.joint_vel.params["asset_cfg"].joint_names = cfg.joint_names
    critic.calf_contact_forces = ObsTerm(
        func=calf_vertical_contact_forces,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True,
        )},
        scale=0.01,
    )
    if cfg.scene.terrain.terrain_type == "generator":
        critic.height_scan = ObservationsCfg.CriticCfg().height_scan


