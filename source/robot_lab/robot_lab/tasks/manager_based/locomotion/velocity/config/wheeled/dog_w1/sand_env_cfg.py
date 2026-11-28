# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import EventTermCfg as EventTerm
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
from .observations import dog_w1_joint_pos_scale, dog_w1_joint_vel_scale, joint_pos_rel_without_wheel
from .rewards import PhaseContactReward, PhaseSwingClearance, standing_contact, lateral_velocity_error, wheel_ground_slip, stationary_wheel_velocity
from .actions import DogW1WheelVelocityActionCfg
from .commands import DogW1VelocityCommand

DOG_OBSTACLE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.01,   # 建议更细，50mm障碍边缘更准确
    vertical_scale=0.005,    # 0.15m = 30个高度格
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "long_bars": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=1.0,
            platform_width=1.0,
            # 关闭课程随机：start=end，得到固定规格障碍
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=16,          # 每块子地形障碍数量，可调
                height=0.3,            # 150 mm
                size=(0.03, 1.6),       # (x方向厚度, y方向长度) -> 50mm厚 + 长条
                max_yx_angle=0.0,
                degrees=True,
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=16,
                height=0.3,
                size=(0.03, 1.6),
                max_yx_angle=0.0,
                degrees=True,
            ),
            abs_height_noise=(0.0, 0.0),
            rel_height_noise=(1.0, 1.0),
        ),
    },
)


def sand_phase_term():
    return ObsTerm(
        func=DogW1Phase,
        params={
            "command_name": "base_velocity", "cycle_time": 0.5, "stop_cycle_time": 0.5,
            "stop_duration": 0.5, "translation_threshold": 0.05, "yaw_threshold": 0.05,
        },
        clip=(-1.0, 1.0), scale=1.0,
    )


@configclass
class ATDogDogW1SandPolicyCfg(ObsGroup):
    # 59 values: phase(2), angular velocity(3), gravity(3), commands(3),
    # joint positions(16, wheel angles masked), joint velocities(16), actions(16).
    phase = sand_phase_term()
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObservationsCfg.PolicyCfg().actions
    enable_corruption = True
    concatenate_terms = True


@configclass
class ATDogDogW1SandCriticCfg(ObsGroup):
    # Same ordering as dog2: privileged velocity first, phase after actions.
    # Retain ATDog's proprioceptive critic, now 62 values including phase.
    base_lin_vel = ObservationsCfg.CriticCfg().base_lin_vel
    base_ang_vel = ObservationsCfg.CriticCfg().base_ang_vel
    projected_gravity = ObservationsCfg.CriticCfg().projected_gravity
    velocity_commands = ObservationsCfg.CriticCfg().velocity_commands
    joint_pos = ObservationsCfg.CriticCfg().joint_pos
    joint_vel = ObservationsCfg.CriticCfg().joint_vel
    actions = ObservationsCfg.CriticCfg().actions
    phase = sand_phase_term()
    enable_corruption = False
    concatenate_terms = True


def sand_configure_observations(cfg):
    cfg.observations.policy = ATDogDogW1SandPolicyCfg()
    cfg.observations.critic = ATDogDogW1SandCriticCfg()
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


def sand_configure_task_reward_values(cfg):
    """沙地障碍奖励权重：步态目标与 flat 一致，地面高度由扫描器提供。"""
    rewards = cfg.rewards

    # --- 终止与机身姿态：先保证不摔倒、不弹跳、不过度倾斜 ---
    rewards.is_terminated.weight = -2.0  # 提前终止固定罚，避免策略故意摔倒重开
    rewards.lin_vel_z_l2.weight = -10.0  # 抑制机身 z 向跳动
    rewards.ang_vel_xy_l2.weight = -3.0  # 抑制 roll/pitch 角速度，减少点头与侧翻
    rewards.flat_orientation_l2.weight = -30.0  # 机身偏离水平越强，惩罚越大（防止前低后高）
    rewards.base_height_l2.weight = -500.0  # 强高度跟踪，目标 0.36m
    rewards.base_height_l2.params["target_height"] = 0.36
    rewards.base_height_l2.params["asset_cfg"].body_names = [cfg.base_link_name]
    rewards.body_lin_acc_l2.weight = -0.0002  # 轻微抑制机身线加速度抖动
    rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [cfg.base_link_name]

    # --- 能耗与平滑：腿关节保留摆动幅度，轮关节只做轻约束 ---
    rewards.joint_torques_l2.weight = -1e-6  # 腿力矩成本，量级极小以免压制摆动
    rewards.joint_torques_l2.params["asset_cfg"].joint_names = cfg.leg_joint_names
    rewards.joint_torques_wheel_l2.weight = -0.1  # 轮力矩成本，避免空转烧电机
    rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    rewards.joint_vel_l2.weight = 0.0  # 腿速惩罚关闭，摆腿需要快速响应
    rewards.joint_vel_l2.params["asset_cfg"].joint_names = cfg.leg_joint_names
    rewards.joint_vel_wheel_l2.weight = -0.005  # 轻微轮速惩罚，不阻碍速度跟踪
    rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    rewards.joint_acc_l2.weight = -1.5e-5  # 腿加速度 L2，动作更平滑易部署
    rewards.joint_acc_l2.params["asset_cfg"].joint_names = cfg.leg_joint_names
    rewards.joint_acc_wheel_l2.weight = -2.5e-6  # 轮加速度 L2，抑制轮速突变
    rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    rewards.joint_pos_limits.weight = -5.0  # 靠近关节限位时强惩罚
    rewards.joint_pos_limits.params["asset_cfg"].joint_names = cfg.leg_joint_names
    rewards.joint_vel_limits.weight = -4.0  # 轮速超限惩罚
    rewards.joint_vel_limits.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    rewards.joint_power.weight = -1e-6  # 机械功率成本，控制发热
    rewards.joint_power.params["asset_cfg"].joint_names = cfg.leg_joint_names

    # --- 站立与姿态正则 ---
    rewards.stand_still.weight = -10.0  # 零命令时惩罚腿部无谓动作
    rewards.stand_still.params["asset_cfg"].joint_names = cfg.leg_joint_names
    rewards.joint_pos_penalty.weight = -10.0  # 强约束关节回到默认姿态附近
    rewards.joint_pos_penalty.params["asset_cfg"].joint_names = cfg.leg_joint_names
    rewards.wheel_vel_penalty.weight = 0.0  # 通用轮速惩罚关闭，交给零命令专用项
    rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    rewards.wheel_vel_zero_cmd_l2.weight = -20.0  # 零命令时惩罚轮速，配合 parking brake 站稳
    rewards.wheel_vel_zero_cmd_l2.func = stationary_wheel_velocity
    rewards.wheel_vel_zero_cmd_l2.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=cfg.wheel_joint_names, preserve_order=True),
        **{key: cfg.observations.policy.phase.params[key]
           for key in ("command_name", "translation_threshold", "yaw_threshold")},
    }
    rewards.wheel_vel_variance.weight = 0.0  # 不强制四轮同速，转向需要差速
    rewards.wheel_vel_variance.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    rewards.joint_mirror.weight = 0.0  # 不强制镜像，允许侧向步态
    rewards.joint_mirror.params["mirror_joints"] = [
        ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
    ]

    # --- 动作平滑与接触安全 ---
    rewards.action_rate_l2.weight = -1.0  # 抑制相邻动作突变，降低部署抖动
    rewards.undesired_contacts.weight = -15.0  # 非足端接触强惩罚
    rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{cfg.foot_link_name}).*"]
    rewards.contact_forces.weight = -1.5e-5  # 足端落地冲击惩罚
    rewards.contact_forces.params["sensor_cfg"].body_names = [cfg.foot_link_name]

    # --- 速度跟踪：主任务奖励，权重最高 ---
    rewards.track_lin_vel_xy_exp.weight = 80.0  # xy 线速度跟踪主奖励
    rewards.track_ang_vel_z_exp.weight = 75.0  # 偏航角速度跟踪奖励

    # --- 步态：旧的腾空/足高项关闭，统一由相位目标驱动 ---
    rewards.feet_air_time.weight = 0.0  # 由相位接触目标取代
    rewards.feet_air_time.params["threshold"] = 0.35
    rewards.feet_air_time.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_contact.weight = 15.0  # 相位匹配的支撑/摆动奖励，权重大以稳定步态
    rewards.feet_contact.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_contact_without_cmd.weight = 4.0  # 零命令时鼓励四轮支撑
    rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_stumble.weight = 0.0  # 绊脚项关闭
    rewards.feet_stumble.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_slide.weight = -5.0  # 轮底滑移惩罚，按接触门控
    rewards.feet_slide.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_slide.params["asset_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_height.weight = -0.0  # 绝对足高项关闭
    rewards.feet_height.params["target_height"] = 0.07
    rewards.feet_height.params["asset_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_height_body.weight = -0.0  # 相对机身足高项关闭
    rewards.feet_height_body.params["target_height"] = -0.29
    rewards.feet_height_body.params["asset_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_gait.weight = 0.0  # 原同步步态项关闭
    rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
    rewards.feet_air_time_variance.weight = 0.0  # 腾空方差项关闭
    rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.upward.weight = 10.0  # 保持机身朝上


def sand_configure_dog_w1_phase_swing_reward(cfg):
    """摆动相抬轮目标：轮心位置扣除实测轮半径；地形任务用高度扫描器取地面高度。"""
    phase_params = cfg.observations.policy.phase.params
    cfg.rewards.phase_swing_clearance = RewTerm(
        func=PhaseSwingClearance,
        weight=-2000.0,  # 摆动相高度误差惩罚，权重大保证抬轮干净
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
            "phase_offset": 0.5,  # FL/RR 与 FR/RL 相差半周期
            "stance_ratio": 0.55,  # 支撑相占比
            "contact_smoothing": 0.07,  # 接触判定平滑窗口
            "foot_radius": 0.0765,  # 实测轮半径
            "swing_height": 0.10,  # 摆动相目标离地高度
            "turn_height_bonus": 0.03,  # 原地转向时额外抬轮
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner_base"),  # 地形任务用高度扫描器，平地不用
        },
    )


def sand_configure_rewards(cfg):
    """组装本任务奖励：相位目标 -> 权重 -> 接触/滑移，再追加任务专属项。"""
    sand_configure_dog_w1_phase_swing_reward(cfg)
    sand_configure_task_reward_values(cfg)

    cfg.rewards.lateral_velocity_error = RewTerm(
        func=lateral_velocity_error, weight=-20.0,  # 密集横向误差，避免指数奖励在大误差处梯度消失
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    # 平面轮底切向滑移：只有轮子真的接触地面时才计入。
    cfg.rewards.feet_slide.func = wheel_ground_slip
    cfg.rewards.feet_slide.params = {
        "asset_cfg": SceneEntityCfg(
            "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
        ),
        "sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
        ),
        "wheel_radius": cfg.rewards.phase_swing_clearance.params["foot_radius"],
    }

    swing_params = cfg.rewards.phase_swing_clearance.params
    phase_params = dict(cfg.observations.policy.phase.params)
    cfg.rewards.feet_contact.func = PhaseContactReward
    cfg.rewards.feet_contact.params = {
        **phase_params,
        "sensor_cfg": SceneEntityCfg(
            "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
        ),
        "centered": True,  # 每个轮子 +1/-1，避免四轮着地只得半分
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

    joint_vel = DogW1WheelVelocityActionCfg(
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
class ATDogDogW1SandEnvCfg(LocomotionVelocityRoughEnvCfg):
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
        # 地形生成器：长条箱体障碍
        self.scene.terrain.terrain_generator = DOG_OBSTACLE_TERRAIN_CFG
        # 将高度扫描器挂到机身 base 上，保证地形感知参考系一致
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        sand_configure_observations(self)

        self.commands.base_velocity.class_type = DogW1VelocityCommand
        for key in ("command_name", "translation_threshold", "yaw_threshold"):
            setattr(self.actions.joint_vel, key, self.observations.policy.phase.params[key])

        # ------------------------------Actions------------------------------
        # 髋关节幅度更小(0.125)，其余腿关节 0.25，轮关节速度缩放 5.0
        self.actions.joint_pos.scale = dog_w1_joint_pos_scale()
        self.actions.joint_vel.scale = dog_w1_joint_vel_scale()
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

        # 与 flat 一致的外力/外力矩扰动：reset 时对 base 施加 ±10 N / ±10 N·m 随机 wrench
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)
        # 与 flat 一致的周期性速度扰动（推搡）：每 15-20 s 把 base 的 xy 速度设为 ±0.2 m/s
        self.events.randomize_push_robot = EventTerm(
            func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(15.0, 20.0),
            params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
        )

        # ------------------------------Rewards------------------------------
        # 奖励权重在本文件对应的 *_configure_task_reward_values 中逐项标注
        sand_configure_rewards(self)

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ATDogDogW1SandEnvCfg":
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
