"""Wall climb on flat ground, keeping the Flat HIM observation/action interface."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from .flat_env_cfg import ATDogDogW1FlatEnvCfg, ATDogDogW1RewardsCfg
from .rewards import (
    PhaseContactReward, PhaseSwingClearance, standing_contact, lateral_velocity_error,
    wheel_ground_slip, stationary_wheel_velocity, standing_joint_velocity,
    standing_posture, standing_base_velocity,
)
from .wall_response import WallContactState, wall_motion_component, wall_response_failed
from .wall_rewards import WallTraversalReward, wall_success, wall_out_of_lane, wall_clearance_error, wall_base_clearance


def wall_configure_task_reward_values(cfg):
    """Wall 基础奖励的独立配方；不从 Flat 读取权重。"""
    rewards = cfg.rewards

    # --- 终止与机身姿态：先保证不摔倒、不弹跳、不过度倾斜 ---
    rewards.is_terminated.weight = -2.0  # 提前终止固定罚，避免策略故意摔倒重开
    rewards.lin_vel_z_l2.weight = -1.0  # 抑制机身 z 向跳动
    rewards.ang_vel_xy_l2.weight = -1.0  # 抑制 roll/pitch 角速度，减少点头与侧翻
    rewards.flat_orientation_l2.weight = -1.0  # 机身偏离水平越强，惩罚越大
    rewards.base_height_l2.weight = 0.0  # 强高度跟踪，目标 0.36m
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
    rewards.stand_still.weight = 0.0  # 由 standing_joint_motion / standing_posture 分别约束
    rewards.stand_still.params["asset_cfg"].joint_names = cfg.leg_joint_names
    rewards.joint_pos_penalty.weight = -0.5  # 保留轻姿态正则，给受扰后的调整留空间
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

    # Dog2 distance_y convention requires FL, FR, RL, RR (+Y,-Y,+Y,-Y).
    # Nominal dog_w1 wheel centers are y ~= +/-0.22 m (not dog2 calf origins).
    rewards.feet_distance_y_exp.weight = 20.0
    rewards.feet_distance_y_exp.params = {
        "stance_width": 0.44, "std": 0.18,
        "asset_cfg": SceneEntityCfg(
            "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
        ),
    }

    # --- 动作平滑与接触安全 ---
    rewards.action_rate_l2.weight = -1.0  # 抑制相邻动作突变，降低部署抖动
    rewards.undesired_contacts.weight = -15.0  # 非足端接触强惩罚
    rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{cfg.foot_link_name}).*"]
    rewards.contact_forces.weight = -1.5e-5  # 足端落地冲击惩罚
    rewards.contact_forces.params["sensor_cfg"].body_names = [cfg.foot_link_name]

    # --- 速度跟踪：主任务奖励，权重最高 ---
    rewards.track_lin_vel_xy_exp.weight = 20.0  # xy 线速度跟踪主奖励
    rewards.track_ang_vel_z_exp.weight = 10.0  # 偏航角速度跟踪奖励

    # --- 步态：旧的腾空/足高项关闭，统一由相位目标驱动 ---
    rewards.feet_air_time.weight = 0.0  # 由相位接触目标取代
    rewards.feet_air_time.params["threshold"] = 0.35
    rewards.feet_air_time.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_contact.weight = 3.0  # 相位匹配的支撑/摆动奖励，权重大以稳定步态
    rewards.feet_contact.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_contact_without_cmd.weight = 4.0  # 零命令时鼓励四轮支撑
    rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_stumble.weight = 0.0  # 绊脚项关闭
    rewards.feet_stumble.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.feet_slide.weight = 0.0  # 轮底滑移惩罚，按接触门控
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
    rewards.upward.weight = 10.0

def wall_configure_dog_w1_phase_swing_reward(cfg):
    """摆动相抬轮目标：轮心位置扣除实测轮半径；地形任务用高度扫描器取地面高度。"""
    phase_params = cfg.observations.policy.phase.params
    cfg.rewards.phase_swing_clearance = RewTerm(
        func=PhaseSwingClearance,
        weight=0.0,  # Wall 关闭地面摆动高度约束，由墙附近的间隙目标替代
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
            "foot_radius": 0.0865,  # 实测轮半径
            "swing_height": 0.11,  # 摆动相目标离地高度
            "turn_height_bonus": 0.04,  # 原地转向时额外抬轮
            "terrain_sensor_cfg": None,  # 地形任务用高度扫描器，平地不用
        },
    )

def wall_configure_rewards(cfg):
    """组装本任务奖励：相位目标 -> 权重 -> 接触/滑移，再追加任务专属项。"""
    wall_configure_dog_w1_phase_swing_reward(cfg)
    wall_configure_task_reward_values(cfg)

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

    # Dog2-style standing recovery: penalize residual movement while allowing correction.
    cfg.rewards.standing_joint_motion = RewTerm(
        func=standing_joint_velocity, weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=cfg.leg_joint_names)},
    )
    cfg.rewards.standing_posture = RewTerm(
        func=standing_posture, weight=-5.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=cfg.leg_joint_names)},
    )
    cfg.rewards.standing_base_velocity = RewTerm(
        func=standing_base_velocity, weight=-2.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )

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

    # Wall-specific objectives. All tunable constants are local to this file.
    feet = SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
    cfg.rewards.wall_task = RewTerm(
        func=WallTraversalReward, weight=1.,
        params={"asset_cfg": feet, "wall_x": cfg.wall_x,
                "wall_thickness": cfg.wall_thickness, "wall_length": cfg.wall_length,
                    "progress_scale": 3.0, "success_bonus": 10.0,
                    "progress_limit": 0.6, "base_margin": 0.35, "foot_margin": 0.08,
                    "lane_margin": 0.3, "upright_threshold": 0.94,
                    "max_angular_speed": 0.6, "max_linear_speed": 0.8,
                    "min_base_height": 0.22, "max_base_height": 0.55,
                    "max_foot_height": 0.18, "stable_duration": 0.3},
    )
    cfg.rewards.wall_foot_clearance = RewTerm(
        func=wall_clearance_error, weight=-80.,
        params={"asset_cfg": feet, "wall_x": cfg.wall_x, "wall_thickness": cfg.wall_thickness,
                "wall_height": cfg.wall_height,
                "foot_radius": .0765, "near_margin": .12, "clearance_margin": .03},
    )
    cfg.rewards.wall_base_clearance = RewTerm(
        func=wall_base_clearance, weight=-50., params={"wall_x": cfg.wall_x, "wall_height": cfg.wall_height, "overlap_half_width": .28, "clearance_margin": .08},
    )


    # Filtered-contact response. State/reward constants are local to this file.
    # Exactly one body per filtered sensor: Isaac Lab supports one-to-many filtering.
    cfg.rewards.wall_response_hip = RewTerm(
        func=WallContactState, weight=1.0,
        params={
            "feet_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True),
            "hip_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"], preserve_order=True),
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True),
            "wheel_sensors": ["wall_contact_FL_foot", "wall_contact_FR_foot", "wall_contact_RL_foot", "wall_contact_RR_foot"],
            "trigger_sensors": ["wall_contact_FL_thigh", "wall_contact_FL_calf", "wall_contact_FL_foot",
                                "wall_contact_FR_thigh", "wall_contact_FR_calf", "wall_contact_FR_foot", "wall_contact_base"],
            "base_sensor": "wall_contact_base",
            "wall_x": cfg.wall_x, "wall_height": cfg.wall_height,
            "wall_thickness": cfg.wall_thickness, "wall_length": cfg.wall_length,
            "foot_radius": cfg.rewards.wall_foot_clearance.params["foot_radius"],
            "clearance_margin": 0.03,
            "contact_threshold": 5.0,       # N, filtered wall contacts only
            "response_timeout": 6.0,        # s after first valid wall contact
            "max_base_contact_time": 0.5,   # s of sustained body pushing
            "base_margin": 0.35, "foot_margin": 0.08,
            "lane_half_width": 0.55, "near_margin": 0.2,
            "target_speed": 0.4,           # m/s, not raw unsigned joint speed
            "hip_outward_signs": [1., -1., -1., 1.],  # FL, FR, RL, RR URDF signs
            "hip_target": 0.15, "hip_sigma": 0.12,  # rad, bounded target
            "lift_sigma": 0.06,            # m
            "wheel_speed_sigma": 0.2,      # m/s
            "collision_force_cap": 50.0,   # N
            **dict(cfg.observations.policy.phase.params),
            "phase_offset": cfg.rewards.feet_contact.params["phase_offset"],
            "stance_ratio": cfg.rewards.feet_contact.params["stance_ratio"],
        },
    )
    cfg.rewards.wall_response_lift = RewTerm(func=wall_motion_component, weight=3.0, params={"component": "lift"})
    cfg.rewards.wall_response_wheel = RewTerm(func=wall_motion_component, weight=1.0, params={"component": "wheel"})
    cfg.rewards.wall_response_progress = RewTerm(func=wall_motion_component, weight=3.0, params={"component": "progress"})
    cfg.rewards.wall_response_clear = RewTerm(func=wall_motion_component, weight=4.0, params={"component": "clear"})
    cfg.rewards.wall_response_slip = RewTerm(func=wall_motion_component, weight=-1.0, params={"component": "slip"})
    cfg.rewards.wall_response_collision = RewTerm(func=wall_motion_component, weight=-0.1, params={"component": "collision"})
    cfg.rewards.wall_response_recovery = RewTerm(func=wall_motion_component, weight=2.0, params={"component": "recovery"})


@configclass
class ATDogDogW1WallEnvCfg(ATDogDogW1FlatEnvCfg):
    """Fixed 30 cm wall. No additional actor/critic inputs in this first experiment."""

    # 墙体参数统一在这里修改，单位 m；重启训练/创建环境后生效。
    wall_height = 0.30     # 墙高
    wall_thickness = 0.08  # 前进方向厚度
    wall_length = 1.8      # 横向长度
    wall_x = 1.0          # 相对环境原点的墙体中心 X

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.
        self.scene.env_spacing = 5.
        self.scene.wall = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Wall",
            spawn=sim_utils.CuboidCfg(
                size=(self.wall_thickness, self.wall_length, self.wall_height),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.55, .35, .15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(self.wall_x, 0., self.wall_height / 2)),
        )
        for body in ("FL_thigh", "FL_calf", "FL_foot", "FR_thigh", "FR_calf", "FR_foot", "RL_foot", "RR_foot", "base"):
            setattr(self.scene, "wall_contact_" + body, ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + body,
                update_period=0.0, history_length=1,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Wall"],
            ))
        command = self.commands.base_velocity
        command.heading_command = True
        command.heading_control_stiffness = 1.
        command.rel_heading_envs = 1.
        command.rel_standing_envs = 0.
        command.resampling_time_range = (12., 12.)
        command.ranges.lin_vel_x = (.25, .4)
        command.ranges.lin_vel_y = (0., 0.)
        command.ranges.ang_vel_z = (-.4, .4)
        command.ranges.heading = (0., 0.)
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-.02, .02), "y": (-.02, .02), "z": (0., .01),
            "roll": (-.03, .03), "pitch": (-.03, .03), "yaw": (-.03, .03),
        }
        # First learn climbing without persistent wrenches or timed kicks.
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_com_positions = None
        # Rebuild all reward terms from Wall's own recipe, discarding Flat mutations.
        self.rewards = ATDogDogW1RewardsCfg()
        wall_configure_rewards(self)
        # Allow brief base-wall contact to trigger response; sustained pushing fails.
        # Non-wheel contact penalties remain. Tilt and response timeout still terminate.
        self.terminations.illegal_contact = None
        self.terminations.wall_response_failed = DoneTerm(func=wall_response_failed)
        self.terminations.bad_orientation.params["limit_angle"] = 1.4
        self.terminations.wall_success = DoneTerm(func=wall_success, time_out=True)
        self.terminations.wall_out_of_lane = DoneTerm(func=wall_out_of_lane, params={"half_width": .55, "min_x": -.5, "max_x": 2.5})
        self.disable_zero_weight_rewards()
