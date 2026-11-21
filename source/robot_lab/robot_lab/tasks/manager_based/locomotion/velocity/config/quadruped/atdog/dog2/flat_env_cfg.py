# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
import torch
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import ATDogDog2RoughEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, RewardsCfg
from .observations import calf_vertical_contact_forces
from .phase_observations import CSDog1Phase, previous_action, velocity_commands
from .rewards import (CommandedGaitReward, PhaseSwingClearance, standing_contact,
                      standing_joint_velocity, standing_posture, standing_base_velocity, turning_two_foot_support)


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


class FlatTurnVelocityCommand(mdp.UniformThresholdVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        turn_in_place = torch.rand(len(env_ids), device=self.device) < 0.2
        self.vel_command_b[env_ids, :2] *= (~turn_in_place).unsqueeze(1)


def flat_csdog1_phase_term():
    return ObsTerm(func=CSDog1Phase, params={
        "command_name": "base_velocity", "cycle_time": 0.5, "stop_cycle_time": 0.5,
        "stop_duration": 0.5, "translation_threshold": 0.05, "yaw_threshold": 0.05,
    }, clip=(-1.0, 1.0), scale=1.0)


def flat_configure_dog2_phase_swing_reward(cfg):
    cfg.rewards.phase_swing_clearance = RewTerm(
        func=PhaseSwingClearance,
        # 摆动相足端高度误差惩罚
        weight=-200.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"], preserve_order=True,
            ),
            "command_name": "base_velocity",
            "cycle_time": 0.5,
            "stop_cycle_time": 0.5,
            "stop_duration": 0.5,
            "translation_threshold": 0.05,
            "yaw_threshold": 0.05,
            "phase_offset": 0.5,
            "stance_ratio": 0.55,
            "contact_smoothing": 0.07,
            "foot_radius": 0.02,
            "swing_height": 0.08,
            "turn_height_bonus": 0.02,
        },
    )


@configclass
class flat_CSDog1PolicyCfg(ObsGroup):
    phase = flat_csdog1_phase_term()
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObsTerm(func=previous_action, clip=(-100.0, 100.0), scale=1.0)
    enable_corruption = True
    concatenate_terms = True


def flat_configure_observations(cfg):
    """Add the shared phase frame and available Dog2 critic privileges."""
    policy = flat_CSDog1PolicyCfg()
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
    critic.phase = flat_csdog1_phase_term()
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


def flat_configure_task_reward_values(self):
    # 提前终止惩罚
    self.rewards.is_terminated.weight = -2.0
    # 机身竖直速度惩罚
    self.rewards.lin_vel_z_l2.weight = -10.0
    # 机身横滚和俯仰角速度惩罚
    self.rewards.ang_vel_xy_l2.weight = -6.0
    # 机身倾斜惩罚
    self.rewards.flat_orientation_l2.weight = -2.0
    # 机身高度误差惩罚
    self.rewards.base_height_l2.weight = -120.0
    self.rewards.base_height_l2.params['target_height'] = 0.33
    self.rewards.base_height_l2.params['asset_cfg'].body_names = [self.base_link_name]
    # 机身线加速度惩罚
    self.rewards.body_lin_acc_l2.weight = -0.0002
    self.rewards.body_lin_acc_l2.params['asset_cfg'].body_names = [self.base_link_name]
    # 关节力矩惩罚
    self.rewards.joint_torques_l2.weight = -2.5e-05
    # 关节速度惩罚
    self.rewards.joint_vel_l2.weight = 0
    # 关节加速度惩罚
    self.rewards.joint_acc_l2.weight = -1e-05
    # 关节位置限位惩罚
    self.rewards.joint_pos_limits.weight = -5.0
    # 关节力矩限位惩罚
    self.rewards.applied_torque_limits.weight = -0.01
    self.rewards.applied_torque_limits.params['asset_cfg'].joint_names = self.joint_names
    # 关节速度限位惩罚
    self.rewards.joint_vel_limits.weight = 0
    # 关节功率惩罚
    self.rewards.joint_power.weight = -4e-05
    # 静止命令下的站姿偏差惩罚
    self.rewards.stand_still.weight = -10.0
    # 关节位置偏离默认姿态惩罚
    self.rewards.joint_pos_penalty.weight = -4.0
    self.rewards.joint_pos_penalty.params['stand_still_scale'] = 4.0
    # 关节镜像不对称惩罚
    self.rewards.joint_mirror.weight = -1.0
    self.rewards.joint_mirror.params['mirror_joints'] = [['FR_(hip|thigh|calf).*', 'RL_(hip|thigh|calf).*'], ['FL_(hip|thigh|calf).*', 'RR_(hip|thigh|calf).*']]
    # 相邻动作变化惩罚
    self.rewards.action_rate_l2.weight = -2.0
    # 动作二阶变化惩罚
    self.rewards.action_smoothness_2_l2.weight = -1.0
    # 非足端接触惩罚
    self.rewards.undesired_contacts.weight = -20.0
    # 过大足端接触力惩罚
    self.rewards.contact_forces.weight = -0.00015
    # 水平线速度跟踪奖励
    self.rewards.track_lin_vel_xy_exp.weight = 80.0
    # 偏航角速度跟踪奖励
    self.rewards.track_ang_vel_z_exp.weight = 60.0
    # 足端腾空时间奖励
    self.rewards.feet_air_time.weight = 5.0
    self.rewards.feet_air_time.params['threshold'] = 0.2925
    # 足端腾空时间不一致惩罚
    self.rewards.feet_air_time_variance.weight = -6.0
    # 足端接触奖励
    self.rewards.feet_contact.weight = 0
    # 足端绊碰惩罚
    self.rewards.feet_stumble.weight = 0.0
    # 足端接触滑动惩罚
    self.rewards.feet_slide.weight = -5.0
    # 足端高度误差惩罚
    self.rewards.feet_height.weight = -0.0
    self.rewards.feet_height.params['target_height'] = 0.08
    self.rewards.feet_height.params['asset_cfg'].body_names = [self.foot_link_name]
    # 足端相对机身高度误差惩罚
    self.rewards.feet_height_body.weight = -0.0
    self.rewards.feet_height_body.params['target_height'] = -0.07
    self.rewards.feet_height_body.params['asset_cfg'].body_names = [self.foot_link_name]
    # 对角步态协调奖励
    self.rewards.feet_gait.weight = 20.0
    # 非对角步态接触惩罚
    self.rewards.diagonal_trot_contact_pattern.weight = -2.0
    # 机身直立奖励
    self.rewards.upward.weight = 15.0
    # 左右足间距奖励
    self.rewards.feet_distance_y_exp.weight = 20.0
    self.rewards.feet_distance_y_exp.params['stance_width'] = 0.18
    self.rewards.feet_distance_y_exp.params['std'] = 0.18
    self.rewards.feet_distance_y_exp.params['asset_cfg'].body_names = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']


def flat_configure_rewards(env_cfg):
    flat_configure_dog2_phase_swing_reward(env_cfg)
    defaults = RewardsCfg()
    for name in ("is_terminated", "applied_torque_limits", "feet_air_time", "feet_air_time_variance",
                 "feet_contact_without_cmd", "feet_slide", "feet_height", "feet_height_body",
                 "feet_gait", "diagonal_trot_contact_pattern"):
        setattr(env_cfg.rewards, name, getattr(defaults, name))
    rewards = env_cfg.rewards
    for name, value in vars(defaults).items():
        if value is not None and getattr(rewards, name, None) is None:
            setattr(rewards, name, value)
    flat_configure_task_reward_values(env_cfg)
    for name in ("feet_air_time", "feet_air_time_variance", "feet_contact_without_cmd", "feet_slide",
                 "feet_stumble", "contact_forces", "feet_contact"):
        getattr(rewards, name).params["sensor_cfg"].body_names = [".*_foot"]
    rewards.feet_slide.params["asset_cfg"].body_names = [".*_foot"]
    rewards.undesired_contacts.params["sensor_cfg"].body_names = ["^(?!.*_foot$).*"]
    rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
    rewards.feet_gait.params["command_threshold"] = 0.0
    rewards.diagonal_trot_contact_pattern.params["foot_names"] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    rewards.feet_contact_without_cmd.func = standing_contact
    # 静止命令下足端接触奖励
    rewards.feet_contact_without_cmd.weight = 10.0
    rewards.feet_gait.func = CommandedGaitReward
    # 原地转向时鼓励交替支撑，避免四脚持续贴地旋转。
    rewards.turning_two_foot_support = RewTerm(
        func=turning_two_foot_support,
        weight=3.0,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "yaw_threshold": 0.2, "translation_threshold": 0.05},
    )
    rewards.standing_joint_motion = RewTerm(
        func=standing_joint_velocity,
        # 静止命令下关节速度惩罚
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=env_cfg.joint_names)},
    )
    rewards.standing_posture = RewTerm(
        # 静止命令下姿态偏差惩罚
        func=standing_posture, weight=-5.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=env_cfg.joint_names)},
    )
    rewards.standing_base_velocity = RewTerm(
        # 静止命令下机身速度惩罚
        func=standing_base_velocity, weight=-2.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )

def flat_domain_rand_csdog1_phase_term():
    return ObsTerm(func=CSDog1Phase, params={
        "command_name": "base_velocity", "cycle_time": 0.5, "stop_cycle_time": 0.5,
        "stop_duration": 0.5, "translation_threshold": 0.05, "yaw_threshold": 0.05,
    }, clip=(-1.0, 1.0), scale=1.0)


def flat_domain_rand_configure_dog2_phase_swing_reward(cfg):
    cfg.rewards.phase_swing_clearance = RewTerm(
        func=PhaseSwingClearance,
        # 摆动相足端高度误差惩罚
        weight=-160.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"], preserve_order=True,
            ),
            "command_name": "base_velocity",
            "cycle_time": 0.5,
            "stop_cycle_time": 0.5,
            "stop_duration": 0.5,
            "translation_threshold": 0.05,
            "yaw_threshold": 0.05,
            "phase_offset": 0.5,
            "stance_ratio": 0.55,
            "contact_smoothing": 0.07,
            "foot_radius": 0.02,
            "swing_height": 0.135,
            "turn_height_bonus": 0.02,
        },
    )


@configclass
class flat_domain_rand_CSDog1PolicyCfg(ObsGroup):
    phase = flat_domain_rand_csdog1_phase_term()
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObsTerm(func=previous_action, clip=(-100.0, 100.0), scale=1.0)
    enable_corruption = True
    concatenate_terms = True


def flat_domain_rand_configure_observations(cfg):
    """Add the shared phase frame and available Dog2 critic privileges."""
    policy = flat_domain_rand_CSDog1PolicyCfg()
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
    critic.phase = flat_domain_rand_csdog1_phase_term()
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


def flat_domain_rand_configure_task_reward_values(self):
    # 提前终止惩罚
    self.rewards.is_terminated.weight = -2.0
    # 机身竖直速度惩罚
    self.rewards.lin_vel_z_l2.weight = -10.0
    # 机身横滚和俯仰角速度惩罚
    self.rewards.ang_vel_xy_l2.weight = -6.0
    # 机身倾斜惩罚
    self.rewards.flat_orientation_l2.weight = -2.0
    # 机身高度误差惩罚
    self.rewards.base_height_l2.weight = -30.0
    self.rewards.base_height_l2.params['target_height'] = 0.33
    self.rewards.base_height_l2.params['asset_cfg'].body_names = [self.base_link_name]
    # 机身线加速度惩罚
    self.rewards.body_lin_acc_l2.weight = -0.0002
    self.rewards.body_lin_acc_l2.params['asset_cfg'].body_names = [self.base_link_name]
    # 关节力矩惩罚
    self.rewards.joint_torques_l2.weight = -2.5e-05
    # 关节速度惩罚
    self.rewards.joint_vel_l2.weight = 0
    # 关节加速度惩罚
    self.rewards.joint_acc_l2.weight = -1e-05
    # 关节位置限位惩罚
    self.rewards.joint_pos_limits.weight = -5.0
    # 关节力矩限位惩罚
    self.rewards.applied_torque_limits.weight = -0.01
    self.rewards.applied_torque_limits.params['asset_cfg'].joint_names = self.joint_names
    # 关节速度限位惩罚
    self.rewards.joint_vel_limits.weight = 0
    # 关节功率惩罚
    self.rewards.joint_power.weight = -4e-05
    # 静止命令下的站姿偏差惩罚
    self.rewards.stand_still.weight = -10.0
    # 关节位置偏离默认姿态惩罚
    self.rewards.joint_pos_penalty.weight = -4.0
    self.rewards.joint_pos_penalty.params['stand_still_scale'] = 4.0
    # 关节镜像不对称惩罚
    self.rewards.joint_mirror.weight = -1.0
    self.rewards.joint_mirror.params['mirror_joints'] = [['FR_(hip|thigh|calf).*', 'RL_(hip|thigh|calf).*'], ['FL_(hip|thigh|calf).*', 'RR_(hip|thigh|calf).*']]
    # 相邻动作变化惩罚
    self.rewards.action_rate_l2.weight = -2.0
    # 动作二阶变化惩罚
    self.rewards.action_smoothness_2_l2.weight = -1.0
    # 非足端接触惩罚
    self.rewards.undesired_contacts.weight = -20.0
    # 过大足端接触力惩罚
    self.rewards.contact_forces.weight = -0.00015
    # 水平线速度跟踪奖励
    self.rewards.track_lin_vel_xy_exp.weight = 100.0
    # 偏航角速度跟踪奖励
    self.rewards.track_ang_vel_z_exp.weight = 80.0
    # 足端腾空时间奖励
    self.rewards.feet_air_time.weight = 5.0
    self.rewards.feet_air_time.params['threshold'] = 0.25
    # 足端腾空时间不一致惩罚
    self.rewards.feet_air_time_variance.weight = -6.0
    # 足端接触奖励
    self.rewards.feet_contact.weight = 0
    # 足端绊碰惩罚
    self.rewards.feet_stumble.weight = 0.0
    # 足端接触滑动惩罚
    self.rewards.feet_slide.weight = -5.0
    # 足端高度误差惩罚
    self.rewards.feet_height.weight = -15.0
    self.rewards.feet_height.params['target_height'] = 0.08
    self.rewards.feet_height.params['asset_cfg'].body_names = [self.foot_link_name]
    # 足端相对机身高度误差惩罚
    self.rewards.feet_height_body.weight = -5.0
    self.rewards.feet_height_body.params['target_height'] = -0.07
    self.rewards.feet_height_body.params['asset_cfg'].body_names = [self.foot_link_name]
    # 对角步态协调奖励
    self.rewards.feet_gait.weight = 20.0
    # 非对角步态接触惩罚
    self.rewards.diagonal_trot_contact_pattern.weight = -2.0
    # 机身直立奖励
    self.rewards.upward.weight = 20.0
    # 左右足间距奖励
    self.rewards.feet_distance_y_exp.weight = 5.0
    self.rewards.feet_distance_y_exp.params['stance_width'] = 0.15
    self.rewards.feet_distance_y_exp.params['std'] = 0.15
    self.rewards.feet_distance_y_exp.params['asset_cfg'].body_names = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']


def flat_domain_rand_configure_rewards(env_cfg):
    env_cfg.rewards.turning_two_foot_support = None
    flat_domain_rand_configure_dog2_phase_swing_reward(env_cfg)
    defaults = RewardsCfg()
    for name in ("is_terminated", "applied_torque_limits", "feet_air_time", "feet_air_time_variance",
                 "feet_contact_without_cmd", "feet_slide", "feet_height", "feet_height_body",
                 "feet_gait", "diagonal_trot_contact_pattern"):
        setattr(env_cfg.rewards, name, getattr(defaults, name))
    rewards = env_cfg.rewards
    for name, value in vars(defaults).items():
        if value is not None and getattr(rewards, name, None) is None:
            setattr(rewards, name, value)
    flat_domain_rand_configure_task_reward_values(env_cfg)
    for name in ("feet_air_time", "feet_air_time_variance", "feet_contact_without_cmd", "feet_slide",
                 "feet_stumble", "contact_forces", "feet_contact"):
        getattr(rewards, name).params["sensor_cfg"].body_names = [".*_foot"]
    rewards.feet_slide.params["asset_cfg"].body_names = [".*_foot"]
    rewards.undesired_contacts.params["sensor_cfg"].body_names = ["^(?!.*_foot$).*"]
    rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
    rewards.feet_gait.params["command_threshold"] = 0.0
    rewards.diagonal_trot_contact_pattern.params["foot_names"] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    rewards.feet_contact_without_cmd.func = standing_contact
    # 静止命令下足端接触奖励
    rewards.feet_contact_without_cmd.weight = 10.0
    rewards.feet_gait.func = CommandedGaitReward
    rewards.standing_joint_motion = RewTerm(
        func=standing_joint_velocity,
        # 静止命令下关节速度惩罚
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=env_cfg.joint_names)},
    )
    rewards.standing_posture = RewTerm(
        # 静止命令下姿态偏差惩罚
        func=standing_posture, weight=-5.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=env_cfg.joint_names)},
    )
    rewards.standing_base_velocity = RewTerm(
        # 静止命令下机身速度惩罚
        func=standing_base_velocity, weight=-2.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
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
        # Keep the ray caster for terrain-aware rewards; critic height_scan is disabled.
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.class_type = FlatTurnVelocityCommand
        flat_configure_observations(self)
        flat_configure_rewards(self)
        self.terminations.illegal_contact = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]), "threshold": 1.0},
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
        self.commands.base_velocity.class_type = mdp.UniformThresholdVelocityCommand
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
        flat_domain_rand_configure_observations(self)
        flat_domain_rand_configure_rewards(self)
        self.disable_zero_weight_rewards()
