# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    ObservationsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.atdog import AT_DOG_W1_CFG  # isort: skip
from .phase_observations import DogW1Phase
from .observations import joint_pos_rel_without_wheel
from .rewards import PhaseContactReward, PhaseSwingClearance, standing_contact


def rough_phase_term():
    return ObsTerm(
        func=DogW1Phase,
        params={
            "command_name": "base_velocity", "cycle_time": 0.5, "stop_cycle_time": 0.5,
            "stop_duration": 0.5, "translation_threshold": 0.05, "yaw_threshold": 0.05,
        },
        clip=(-1.0, 1.0), scale=1.0,
    )


@configclass
class ATDogDogW1RoughPolicyCfg(ObsGroup):
    # 59 values: phase(2), angular velocity(3), gravity(3), commands(3),
    # joint positions(16, wheel angles masked), joint velocities(16), actions(16).
    phase = rough_phase_term()
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObservationsCfg.PolicyCfg().actions
    enable_corruption = True
    concatenate_terms = True


@configclass
class ATDogDogW1RoughCriticCfg(ObsGroup):
    # Same ordering as dog2: privileged velocity first, phase after actions.
    # Retain ATDog's proprioceptive critic, now 62 values including phase.
    base_lin_vel = ObservationsCfg.CriticCfg().base_lin_vel
    base_ang_vel = ObservationsCfg.CriticCfg().base_ang_vel
    projected_gravity = ObservationsCfg.CriticCfg().projected_gravity
    velocity_commands = ObservationsCfg.CriticCfg().velocity_commands
    joint_pos = ObservationsCfg.CriticCfg().joint_pos
    joint_vel = ObservationsCfg.CriticCfg().joint_vel
    actions = ObservationsCfg.CriticCfg().actions
    phase = rough_phase_term()
    enable_corruption = False
    concatenate_terms = True


def rough_configure_observations(cfg):
    cfg.observations.policy = ATDogDogW1RoughPolicyCfg()
    cfg.observations.critic = ATDogDogW1RoughCriticCfg()
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.base_ang_vel.scale = 0.25
        group.joint_pos.scale = 1.0
        group.joint_vel.scale = 0.05
        group.joint_pos.func = joint_pos_rel_without_wheel
        group.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=cfg.wheel_joint_names, preserve_order=True,
        )
        for name in ("joint_pos", "joint_vel"):
            group_term = getattr(group, name)
            group_term.params["asset_cfg"].joint_names = cfg.joint_names
            group_term.params["asset_cfg"].preserve_order = True



def rough_configure_task_reward_values(cfg):
    """集中设置 rough 地形的奖励权重与参数。"""

    cfg.rewards.lin_vel_z_l2.weight = -30.0
    # 惩罚机身 x/y 角速度（roll/pitch 旋转速度），降低侧翻和点头抖动
    cfg.rewards.ang_vel_xy_l2.weight = -1.0
    # 惩罚机身姿态偏离水平（roll/pitch 倾斜角误差）
    cfg.rewards.flat_orientation_l2.weight = 0
    # 机身高度跟踪惩罚（当前关闭）
    cfg.rewards.base_height_l2.weight = 0
    cfg.rewards.base_height_l2.params["target_height"] = 0.35
    # 指定用 base 刚体计算该项（避免多 body 统计带来歧义）
    cfg.rewards.base_height_l2.params["asset_cfg"].body_names = [cfg.base_link_name]
    # 惩罚机身线加速度（平滑机身受力/运动），当前关闭
    cfg.rewards.body_lin_acc_l2.weight = 0
    cfg.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [cfg.base_link_name]

    # Joint penalties
    # 腿部力矩 L2 惩罚，控制能耗并抑制“暴力驱动”
    cfg.rewards.joint_torques_l2.weight = -2.5e-6
    cfg.rewards.joint_torques_l2.params["asset_cfg"].joint_names = cfg.leg_joint_names
    # 轮部力矩 L2 惩罚（单独项，当前关闭）
    cfg.rewards.joint_torques_wheel_l2.weight = -0.4
    cfg.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    # 腿部关节速度 L2 惩罚（当前关闭）
    cfg.rewards.joint_vel_l2.weight = 0
    cfg.rewards.joint_vel_l2.params["asset_cfg"].joint_names = cfg.leg_joint_names
    # 轮部关节速度 L2 惩罚（当前关闭）
    cfg.rewards.joint_vel_wheel_l2.weight = -2.5e-2
    cfg.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    # 腿部关节加速度 L2 惩罚，鼓励动作更平滑
    cfg.rewards.joint_acc_l2.weight = -2.5e-5
    cfg.rewards.joint_acc_l2.params["asset_cfg"].joint_names = cfg.leg_joint_names
    # 轮部关节加速度 L2 惩罚
    cfg.rewards.joint_acc_wheel_l2.weight = -2.5e-6
    cfg.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    # cfg.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
    # 腿部关节限位惩罚: 接近/触发关节上下限时强惩罚，防止打限位
    cfg.rewards.joint_pos_limits.weight = -5.0
    cfg.rewards.joint_pos_limits.params["asset_cfg"].joint_names = cfg.leg_joint_names
    # 轮关节速度上限惩罚，当前关闭
    cfg.rewards.joint_vel_limits.weight = -4.0
    cfg.rewards.joint_vel_limits.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    # 腿部关节功率惩罚（约束机械功输出），降低发热与电池消耗
    cfg.rewards.joint_power.weight = -2e-5
    cfg.rewards.joint_power.params["asset_cfg"].joint_names = cfg.leg_joint_names
    # 静止命令下站立惩罚项（鼓励“该停就停”）
    cfg.rewards.stand_still.weight = -2.0
    cfg.rewards.stand_still.params["asset_cfg"].joint_names = cfg.leg_joint_names
    # 关节位置正则惩罚（通常相对默认姿态/安全姿态），抑制异常构型
    cfg.rewards.joint_pos_penalty.weight = -4.0
    cfg.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = cfg.leg_joint_names
    # 轮速惩罚（配合接触与命令状态使用），当前关闭
    cfg.rewards.wheel_vel_penalty.weight = 0
    cfg.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    cfg.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    # 轮速方差惩罚: 约束四个轮子速度（绝对值）保持一致，提升轮式运动协调性
    cfg.rewards.wheel_vel_variance.weight = -0.2
    cfg.rewards.wheel_vel_variance.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    # 镜像对称惩罚: 约束对角腿对应关节的位置幅值相近，减少“偏腿”步态
    cfg.rewards.joint_mirror.weight = -2.0
    cfg.rewards.joint_mirror.params["mirror_joints"] = [
        ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
    ]

    # Action penalties
    # 动作变化率惩罚，抑制相邻时刻动作突变，提升控制平滑性与可部署性
    cfg.rewards.action_rate_l2.weight = -0.7

    # Contact sensor
    # 非足端 body 接触惩罚（如躯干/大腿触地），鼓励“只让脚接触地面”
    cfg.rewards.undesired_contacts.weight = -15.0
    cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{cfg.foot_link_name}).*"]
    # 足端接触力惩罚，避免落脚冲击过大
    cfg.rewards.contact_forces.weight = -1.5e-5
    cfg.rewards.contact_forces.params["sensor_cfg"].body_names = [cfg.foot_link_name]

    # Velocity-tracking rewards
    # 线速度追踪主奖励（xy 平面，指数型）
    cfg.rewards.track_lin_vel_xy_exp.weight = 50.0
    # 偏航角速度追踪奖励（绕 z 转向）
    cfg.rewards.track_ang_vel_z_exp.weight = 70.0

    # Others
    # 足端腾空时间奖励: 鼓励形成明确摆动相，避免拖脚
    cfg.rewards.feet_air_time.weight = 0.0  # Replaced by shared-phase contact/clearance targets.
    # 只在腾空时间超过阈值时开始计入（单位 s），避免“微小离地”刷分
    cfg.rewards.feet_air_time.params["threshold"] = 0.35
    cfg.rewards.feet_air_time.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    # 移动时按相位匹配各轮持续接触状态，满分为 1。
    cfg.rewards.feet_contact.weight = 5.0
    cfg.rewards.feet_contact.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    # 无速度命令时保持接触（鼓励原地稳定）
    cfg.rewards.feet_contact_without_cmd.weight = 2.0
    cfg.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    # 绊脚惩罚（当前关闭）
    cfg.rewards.feet_stumble.weight = 0
    cfg.rewards.feet_stumble.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    # 滑足惩罚: 脚着地后相对地面滑移越大，惩罚越大
    cfg.rewards.feet_slide.weight = -3.0
    cfg.rewards.feet_slide.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    cfg.rewards.feet_slide.params["asset_cfg"].body_names = [cfg.foot_link_name]
    # 足端绝对高度目标项
    cfg.rewards.feet_height.weight = 0.0  # Replaced by shared-phase contact/clearance targets.
    cfg.rewards.feet_height.params["target_height"] = 0.15
    cfg.rewards.feet_height.params["asset_cfg"].body_names = [cfg.foot_link_name]
    # 相对机身的足端高度惩罚（body frame）
    cfg.rewards.feet_height_body.weight = 0.0  # Replaced by shared-phase contact/clearance targets.
    cfg.rewards.feet_height_body.params["target_height"] = -0.20
    cfg.rewards.feet_height_body.params["asset_cfg"].body_names = [cfg.foot_link_name]
    # 原接触时间同步项关闭，由共享相位指定对角支撑。
    cfg.rewards.feet_gait.weight = 0.0  # Replaced by shared-phase contact/clearance targets.
    cfg.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))

    cfg.rewards.feet_air_time_variance.weight = 0.0  # Replaced by shared-phase contact/clearance targets.
    cfg.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [cfg.foot_link_name]     

    # 机身朝上约束奖励
    cfg.rewards.upward.weight = 20.0


def rough_configure_dog_w1_phase_swing_reward(cfg):
    """与 dog2 相同的相位奖励配置格式；轮心位置扣除实际轮半径。"""
    phase_params = cfg.observations.policy.phase.params
    cfg.rewards.phase_swing_clearance = RewTerm(
        func=PhaseSwingClearance,
        # 摆动相轮足高度误差惩罚
        weight=-160.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
            ),
            "command_name": phase_params["command_name"],
            "cycle_time": phase_params["cycle_time"],
            "stop_cycle_time": phase_params["stop_cycle_time"],
            "stop_duration": phase_params["stop_duration"],
            "translation_threshold": phase_params["translation_threshold"],
            "yaw_threshold": phase_params["yaw_threshold"],
            "phase_offset": 0.5,
            "stance_ratio": 0.55,
            "contact_smoothing": 0.07,
            "foot_radius": 0.08,
            "swing_height": 0.08,
            "turn_height_bonus": 0.02,
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner_base"),
        },
    )


def rough_configure_rewards(cfg):
    """按 dog2 顺序设置相位目标，再应用本任务奖励参数。"""
    rough_configure_dog_w1_phase_swing_reward(cfg)
    rough_configure_task_reward_values(cfg)
    # 接触与摆动共享相位、占空比、对角偏移和切换平滑参数。
    swing_params = cfg.rewards.phase_swing_clearance.params
    phase_params = dict(cfg.observations.policy.phase.params)
    cfg.rewards.feet_contact.func = PhaseContactReward
    cfg.rewards.feet_contact.params = {
        **phase_params,
        "sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
        ),
        "phase_offset": swing_params["phase_offset"],
        "stance_ratio": swing_params["stance_ratio"],
        "contact_smoothing": swing_params["contact_smoothing"],
    }
    cfg.rewards.feet_contact_without_cmd.func = standing_contact
    cfg.rewards.feet_contact_without_cmd.params = {
        "command_name": phase_params["command_name"],
        "translation_threshold": phase_params["translation_threshold"],
        "yaw_threshold": phase_params["yaw_threshold"],
        "sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
        ),
    }

@configclass
class ATDogDogW1ActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class ATDogDogW1RewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class ATDogDogW1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # HIM teacher uses clean privileged observations and metadata-derived slices.
    him_clean_critic = True
    actions: ATDogDogW1ActionsCfg = ATDogDogW1ActionsCfg()
    rewards: ATDogDogW1RewardsCfg = ATDogDogW1RewardsCfg()

    # 机身主刚体名称，用于:
    # 1) 传感器挂载（高度扫描器）
    # 2) 质量/质心/外力随机化时筛选 body
    # 3) 与 base 相关奖励项的 body 指定
    base_link_name = "base"
    # 轮足末端 body，用于接触相关奖励/惩罚筛选
    foot_link_name = ".*_foot"

    # fmt: off
    # 腿部关节顺序与动作/观测向量顺序保持一致，避免策略输入输出错位
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # 轮关节集合（速度控制）
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    # 全关节集合（观测使用）
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # 先继承父类默认配置，再按 ATDog 轮式地形任务覆写
        super().__post_init__()

        # ------------------------------Scene 场景与传感器------------------------------
        # 指定机器人资产，并放置到每个并行环境的 Robot prim 下
        self.scene.robot = AT_DOG_W1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 将高度扫描器挂到机身 base 上，保证地形感知参考系一致
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        rough_configure_observations(self)

        # ------------------------------Actions------------------------------
        # 髋关节幅度更小(0.125)，其余腿关节 0.25，轮关节速度缩放 5.0
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.02),
                "roll": (-0.8, 0.8), "pitch": (-0.8, 0.8), "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [f"^(?!.*{self.base_link_name}).*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # 奖励权重集中在 rough_configure_task_reward_values / rough_configure_rewards
        rough_configure_rewards(self)

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ATDogDogW1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
