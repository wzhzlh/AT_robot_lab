# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp as isaaclab_mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .periodic_gait import configure_periodic_gait
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, RewardsCfg
from .observations import calf_vertical_contact_forces
from .phase_observations import CSDog1Phase, previous_action, velocity_commands
from .rewards import (CommandedGaitReward, PhaseSwingClearance, standing_contact,
                      standing_joint_velocity, standing_posture, standing_base_velocity)

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets.atdog import AT_DOG2_CFG  # isort: skip

# 自定义台阶地形（倒金字塔上台阶）:
# - 台阶高 10cm
# - 台阶水平长度 30cm
# 说明:
# 使用 MeshInvertedPyramidStairsTerrainCfg（与 Isaac Lab 默认 rough 配置一致）
# 生成倒金字塔台阶，机器人可从低处向高处持续上台阶。
DOG2_ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.09, 0.11),#(最初0.07-0.13)
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.09, 0.11),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


def rough_csdog1_phase_term():
    return ObsTerm(func=CSDog1Phase, params={
        "command_name": "base_velocity", "cycle_time": 0.5, "stop_cycle_time": 0.5,
        "stop_duration": 0.5, "translation_threshold": 0.05, "yaw_threshold": 0.05,
    }, clip=(-1.0, 1.0), scale=1.0)


def rough_configure_dog2_phase_swing_reward(cfg):
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
class rough_CSDog1PolicyCfg(ObsGroup):
    phase = rough_csdog1_phase_term()
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObsTerm(func=previous_action, clip=(-100.0, 100.0), scale=1.0)
    enable_corruption = True
    concatenate_terms = True


def rough_configure_observations(cfg):
    """Add the shared phase frame and available Dog2 critic privileges."""
    policy = rough_CSDog1PolicyCfg()
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
    critic.phase = rough_csdog1_phase_term()
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


def rough_configure_task_reward_values(self):
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


def rough_configure_rewards(env_cfg):
    rough_configure_dog2_phase_swing_reward(env_cfg)
    defaults = RewardsCfg()
    for name in ("is_terminated", "applied_torque_limits", "feet_air_time", "feet_air_time_variance",
                 "feet_contact_without_cmd", "feet_slide", "feet_height", "feet_height_body",
                 "feet_gait", "diagonal_trot_contact_pattern"):
        setattr(env_cfg.rewards, name, getattr(defaults, name))
    rewards = env_cfg.rewards
    for name, value in vars(defaults).items():
        if value is not None and getattr(rewards, name, None) is None:
            setattr(rewards, name, value)
    rough_configure_task_reward_values(env_cfg)
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
class ATDogDog2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # 机身主刚体名称，用于:
    # 1) 传感器挂载（高度扫描器）
    # 2) 质量/质心/外力随机化时筛选 body
    # 3) 与 base 相关奖励项的 body 指定
    base_link_name = "base"
    # URDF 中的 foot link 通过固定关节连接，导入时（merge_fixed_joints=True）
    # 会并入父级 calf body，因此接触相关奖励/惩罚需要匹配末端 calf body。
    foot_link_name = ".*_calf"
    # fmt: off
    # 关节顺序与动作/观测向量顺序保持一致，避免策略输入输出错位
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # 先继承父类默认配置，再按 ATDog Dog2 粗糙地形任务覆写
        super().__post_init__()
        configure_periodic_gait(self)
        # 保持采样频率不变（sim.dt 与 decimation 沿用父类），仅将单回合时长设为 2s
        #self.episode_length_s = 2.0

        # ------------------------------Scene 场景与传感器------------------------------
        # 覆写默认 rough terrain：仅使用固定参数台阶地形
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = DOG2_ROUGH_TERRAIN_CFG
        # 指定机器人资产，并放置到每个并行环境的 Robot prim 下
        self.scene.robot = AT_DOG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 将高度扫描器挂到机身 base 上，保证地形感知参考系一致
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations 观测------------------------------
        # 各观测分量缩放系数:
        # - scale 越大，对应量在策略输入中的数值幅度越大
        # - 需结合网络归一化习惯，避免某一类信号过强/过弱
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 关闭策略侧 base_lin_vel 与 height_scan 观测（即不输入给 actor）
        # 说明: 这里常用于做“部分可观测”训练，迫使策略更依赖本体状态
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.velocity_commands.scale = (1.0, 1.0, 0.25)
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.velocity_commands.scale = (1.0, 1.0, 0.25)
        # 明确 joint_pos/joint_vel 仅采集 joint_names 中定义的关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # 添加姿态四元数观测，因为真实狗IMU能反馈角度（四元数形式）
        # self.observations.policy.base_orientation = ObsTerm(
        #     func=isaaclab_mdp.root_quat_w,
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )
        # # 添加关节力矩观测，因为电机能反馈力矩
        # self.observations.policy.joint_effort = ObsTerm(
        #     func=isaaclab_mdp.joint_effort,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)},
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-100.0, 100.0),
        #     scale=0.01,
        # )

        # ------------------------------Actions 动作------------------------------
        # 动作缩放:
        # - 所有关节使用相同的 0.25 动作缩放
        # 这里的 key 是正则，按关节名匹配后应用对应 scale
        self.actions.joint_pos.scale = {".*": 0.25}
        # 动作裁剪区间（非常宽），主要作为安全兜底防止异常值爆炸
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        # 限定动作控制关节集合，顺序与 joint_names 对齐
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events 随机化事件------------------------------
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.0)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
        }
        self.events.randomize_reset_joints.params["position_range"] = (0.9, 1.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_push_robot.interval_range_s = (15.0, 15.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        # reset 时随机化机身位姿与速度:
        # - pose_range: 初始位置/姿态扰动范围
        # - velocity_range: 初始线速度/角速度扰动范围
        # 目的: 提升鲁棒性，减少对单一起始状态过拟合
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.5, 0.5),
            },
        }
        # 仅随机化 base 的质量
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # 随机化除 base 外其他刚体质量（负向前瞻正则: 不匹配 base）
        # 仅随机化 base 质心位置
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # 外力/外力矩扰动施加到 base，模拟推搡/干扰
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards 奖励函数------------------------------
        # 以下保留原始说明；其中旧权重和启用状态以文件前方的任务奖励函数为准。
        # 约定:
        # - 正权重: 鼓励该行为（reward）-> 底层返回正值，策略争取高分
        # - 负权重: 惩罚该行为（penalty）-> 底层返回误差/能耗，策略避免扣分
        # - 权重为0: 本项不参与训练（后面可统一 disable）

        # General
        # 终止奖励（通常在 episode 提前结束时给固定惩罚/奖励）。
        # 提前终止时施加固定惩罚。

        # Root penalties
        # 惩罚机身 z 方向线速度，抑制“跳跃/颠簸”。
        # 绝对值增大 -> 更追求贴地平稳；过大可能抑制跨越障碍能力。
        # 惩罚机身 x/y 角速度（roll/pitch 旋转速度），降低侧翻和点头抖动。
        # 惩罚机身姿态偏离水平（roll/pitch 倾斜角误差）。
        # 与速度追踪和接触项共同约束机身姿态。
        # 机身高度跟踪惩罚: 鼓励 base 高度接近 target_height。
        # 当前权重为零，目标高度仅保留在配置中。
        # 目标机身高度（单位 m）。
        # 指定用 base 刚体计算该项（避免多 body 统计带来歧义）。
        # 惩罚机身线加速度（平滑机身受力/运动），当前关闭。
        # 该项常用于减小真实机器冲击与传感器饱和风险。

        # Joint penalties
        # 力矩 L2 惩罚，控制能耗并抑制“暴力驱动”。
        # 通常与 joint_power 配合，一个约束幅值，一个约束功率。
        # 关节速度 L2 惩罚，抑制关节甩动，当前关闭。
        # 关节加速度 L2 惩罚，鼓励动作更平滑、降低冲击。
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        # 关节限位惩罚: 接近/触发关节上下限时强惩罚，防止打限位。
        # 关节速度上限惩罚，当前关闭（可在硬件部署前再打开做保守化）。
        # 关节功率惩罚（约束机械功输出），降低发热与电池消耗。
        # 静止命令下站立惩罚项（鼓励“该停就停”）。
        # 权重绝对值越大，零速命令时越倾向快速收敛到稳态。
        # 关节位置正则惩罚（通常相对默认姿态/安全姿态），抑制异常构型。
        # 镜像对称惩罚: 约束对角腿运动统计相近，减少“偏腿”步态。
        # 指定镜像关节对:
        # - FR 对 RL
        # - FL 对 RR
        # 有助于形成更自然的对角步态模式。

        # Action penalties
        # 动作变化率惩罚，抑制相邻时刻动作突变，提升控制平滑性与可部署性。
        # 二阶动作差分惩罚，抑制高频来回修正导致的机身前后/上下抖动。

        # Contact sensor
        # 非足端 body 接触惩罚（如躯干/大腿触地），鼓励“只让脚接触地面”。
        # 足端接触力惩罚，避免落脚冲击过大。
        # 过大可能导致“轻触地”倾向，影响抓地与推进效率。

        # Velocity-tracking rewards
        # 线速度追踪主奖励（xy 平面，指数型）。
        # 常为 locomotion 核心驱动项，值越大越优先“跟得上命令”。
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.8
        # 偏航角速度追踪奖励（绕 z 转向），支持转向命令执行。
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.8

        # Others
        # 足接触奖励（可用于鼓励稳定支撑），当前关闭。
        # 绊脚/碰撞惩罚，当前关闭（可按地形难度逐步启用）。
        # 机身“向上”姿态奖励（保持重力反方向对齐），提升整体直立稳定性。


        rough_configure_observations(self)
        rough_configure_rewards(self)

        # 将权重为0的奖励项禁用，减少无效计算与配置噪声
        if self.__class__.__name__ == "ATDogDog2RoughEnvCfg": 
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations 终止条件------------------------------
        # End the episode when the base or a hip hits the ground; the environment resets it automatically.
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums 课程学习------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        # 关闭命令课程，直接使用固定命令范围
        self.curriculum.command_levels = None

        # ------------------------------Commands 命令范围------------------------------
        # 命令分布与桌面仓库的 CSDog1 rough 任务一致。
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
