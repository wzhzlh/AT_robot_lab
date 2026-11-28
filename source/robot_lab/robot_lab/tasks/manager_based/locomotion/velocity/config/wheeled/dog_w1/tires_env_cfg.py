# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp  # noqa: F401
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from robot_lab.assets.atdog import AT_DOG_W1_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

from .actions import DogW1WheelVelocityActionCfg
from .commands import DogW1VelocityCommand
from .observations import dog_w1_joint_pos_scale, dog_w1_joint_vel_scale, joint_pos_rel_without_wheel
from .phase_observations import DogW1Phase
from .rewards import PhaseContactReward, PhaseSwingClearance, standing_contact, lateral_velocity_error, wheel_ground_slip, stationary_wheel_velocity, base_height_above_support_l2
from .tires_traversal import (
    TireObstacleClimbReward, TireTraversalReward, TIRE_OUTER_RADIUS, TIRE_INNER_RADIUS, TIRE_SPACING, TIRE_HEIGHT,
)


def tire_array_terrain(
    difficulty: float, cfg: MeshPlaneTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a dense field of randomly offset 2 by 2 flat annular tire arrays."""
    meshes, origin = mesh_terrains.flat_terrain(difficulty, cfg)
    outer_radius = TIRE_OUTER_RADIUS
    inner_radius = TIRE_INNER_RADIUS
    tire_width = TIRE_HEIGHT  # 与遇障抬腿目标共享实际轮胎高度
    segments = 12
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    z_bottom = 0.0
    z_top = tire_width
    ring_vertices = np.vstack(
        [
            np.column_stack((outer_radius * np.cos(angles), outer_radius * np.sin(angles), np.full(segments, z_bottom))),
            np.column_stack((outer_radius * np.cos(angles), outer_radius * np.sin(angles), np.full(segments, z_top))),
            np.column_stack((inner_radius * np.cos(angles), inner_radius * np.sin(angles), np.full(segments, z_bottom))),
            np.column_stack((inner_radius * np.cos(angles), inner_radius * np.sin(angles), np.full(segments, z_top))),
        ]
    )
    faces = []
    for index in range(segments):
        nxt = (index + 1) % segments
        outer_bottom, outer_top = index, segments + index
        inner_bottom, inner_top = 2 * segments + index, 3 * segments + index
        outer_bottom_next, outer_top_next = nxt, segments + nxt
        inner_bottom_next, inner_top_next = 2 * segments + nxt, 3 * segments + nxt
        faces.extend(
            [
                (outer_bottom, outer_bottom_next, outer_top_next),
                (outer_bottom, outer_top_next, outer_top),
                (inner_bottom, inner_top_next, inner_bottom_next),
                (inner_bottom, inner_top, inner_top_next),
                (outer_top, outer_top_next, inner_top_next),
                (outer_top, inner_top_next, inner_top),
                (outer_bottom, inner_bottom_next, outer_bottom_next),
                (outer_bottom, inner_bottom, inner_bottom_next),
            ]
        )
    tire_mesh = trimesh.Trimesh(vertices=ring_vertices, faces=np.asarray(faces), process=True)

    tile_center_x, tile_center_y = origin[:2]
    rng = np.random.default_rng(np.random.randint(0, 2**32, dtype=np.uint32))
    grid_spacing = 1.18
    grid_offsets = (np.arange(6) - 2.5) * grid_spacing
    array_centers = []
    for offset_x in grid_offsets:
        for offset_y in grid_offsets:
            if abs(offset_x) < 1.0 and abs(offset_y) < 1.0:
                continue
            array_centers.append(
                (
                    tile_center_x + offset_x + rng.uniform(-0.02, 0.02),
                    tile_center_y + offset_y + rng.uniform(-0.02, 0.02),
                )
            )
    rng.shuffle(array_centers)
    cfg.group_centers = [(x - tile_center_x, y - tile_center_y) for x, y in array_centers]

    for array_center_x, array_center_y in array_centers:
        for row in range(2):
            for column in range(2):
                center_x = array_center_x + (column - 0.5) * TIRE_SPACING
                center_y = array_center_y + (row - 0.5) * TIRE_SPACING
                tire = tire_mesh.copy()
                tire.apply_translation((center_x, center_y, 0.0))
                meshes.append(tire)
    return meshes, origin


@configclass
class MeshTireArrayTerrainCfg(MeshPlaneTerrainCfg):
    function = tire_array_terrain
    group_centers: list = []


class TireTerrainGenerator(terrain_gen.TerrainGenerator):
    def _get_terrain_mesh(self, difficulty, cfg):
        if self.cfg.use_cache:
            raise ValueError("Tire traversal metadata requires terrain caching to be disabled")
        terrain_cfg = cfg.copy()
        meshes, origin = terrain_cfg.function(difficulty, terrain_cfg)
        self._tire_group_centers = terrain_cfg.group_centers
        mesh = trimesh.util.concatenate(meshes)
        transform = np.eye(4)
        transform[:2, 3] = -np.asarray(terrain_cfg.size) * 0.5
        mesh.apply_transform(transform)
        return mesh, origin + transform[:3, 3]

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        if not self.cfg.group_centers:
            self.cfg.group_centers = [[None for _ in range(self.cfg.num_cols)] for _ in range(self.cfg.num_rows)]
        self.cfg.group_centers[row][col] = self._tire_group_centers
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


@configclass
class TireTerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    class_type = TireTerrainGenerator
    group_centers: list = []


DOG_W1_TIRE_ARRAY_TERRAIN_CFG = TireTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "tire_array": MeshTireArrayTerrainCfg(proportion=1.0),
    },
)


def tires_phase_term():
    return ObsTerm(
        func=DogW1Phase,
        params={
            "command_name": "base_velocity", "cycle_time": 0.5, "stop_cycle_time": 0.5,
            "stop_duration": 0.5, "translation_threshold": 0.05, "yaw_threshold": 0.05,
        },
        clip=(-1.0, 1.0), scale=1.0,
    )


@configclass
class ATDogDogW1TiresPolicyCfg(ObsGroup):
    # 59 values: phase(2), angular velocity(3), gravity(3), commands(3),
    # joint positions(16, wheel angles masked), joint velocities(16), actions(16).
    phase = tires_phase_term()
    base_ang_vel = ObservationsCfg.PolicyCfg().base_ang_vel
    projected_gravity = ObservationsCfg.PolicyCfg().projected_gravity
    velocity_commands = ObservationsCfg.PolicyCfg().velocity_commands
    joint_pos = ObservationsCfg.PolicyCfg().joint_pos
    joint_vel = ObservationsCfg.PolicyCfg().joint_vel
    actions = ObservationsCfg.PolicyCfg().actions
    enable_corruption = True
    concatenate_terms = True


@configclass
class ATDogDogW1TiresCriticCfg(ObsGroup):
    # Same ordering as dog2: privileged velocity first, phase after actions.
    # Retain ATDog's proprioceptive critic, now 62 values including phase.
    base_lin_vel = ObservationsCfg.CriticCfg().base_lin_vel
    base_ang_vel = ObservationsCfg.CriticCfg().base_ang_vel
    projected_gravity = ObservationsCfg.CriticCfg().projected_gravity
    velocity_commands = ObservationsCfg.CriticCfg().velocity_commands
    joint_pos = ObservationsCfg.CriticCfg().joint_pos
    joint_vel = ObservationsCfg.CriticCfg().joint_vel
    actions = ObservationsCfg.CriticCfg().actions
    phase = tires_phase_term()
    enable_corruption = False
    concatenate_terms = True


def tires_configure_observations(cfg):
    cfg.observations.policy = ATDogDogW1TiresPolicyCfg()
    cfg.observations.critic = ATDogDogW1TiresCriticCfg()
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


def tires_configure_task_reward_values(cfg):
    """轮胎穿越奖励权重：共享 flat 步态基线，额外追加越障与站宽项。"""
    rewards = cfg.rewards

    # --- 终止与机身姿态：先保证不摔倒、不弹跳、不过度倾斜 ---
    rewards.is_terminated.weight = -2.0  # 提前终止固定罚，避免策略故意摔倒重开
    rewards.lin_vel_z_l2.weight = -10.0  # 抑制机身 z 向跳动
    rewards.ang_vel_xy_l2.weight = -3.0  # 抑制 roll/pitch 角速度，减少点头与侧翻
    rewards.flat_orientation_l2.weight = -100.0  # 机身偏离水平越强，惩罚越大（防止前低后高）
    # 爬轮胎时 base 的世界高度会上升：改为相对"支撑轮底部"跟踪，
    # 不再依赖 3x3 射线(0.1m 采样窗)去猜地面高度——轮胎中间是空洞，射线很容易打到地面。
    rewards.base_height_l2.weight = -1050.0  # 强高度跟踪，目标仍是离支撑面 0.37m
    rewards.base_height_l2.func = base_height_above_support_l2
    rewards.base_height_l2.params["target_height"] = 0.370
    rewards.base_height_l2.params["asset_cfg"].body_names = [cfg.base_link_name]
    rewards.base_height_l2.params["wheel_cfg"] = SceneEntityCfg(
        "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
    )
    rewards.base_height_l2.params["contact_sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
    )
    rewards.base_height_l2.params["wheel_radius"] = 0.0765
    rewards.base_height_l2.params["sensor_cfg"] = None  # 高度扫描器不再参与本项
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
    rewards.joint_pos_penalty.weight = -30.0  # 强约束关节回到默认姿态附近
    rewards.joint_pos_penalty.params["asset_cfg"].joint_names = cfg.leg_joint_names
    rewards.wheel_vel_penalty.weight = 0.0  # 通用轮速惩罚关闭，交给零命令专用项
    rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [cfg.foot_link_name]
    rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = cfg.wheel_joint_names
    rewards.wheel_vel_zero_cmd_l2.weight = -10.0  # 零命令时惩罚轮速，配合 parking brake 站稳
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
    rewards.undesired_contacts.weight = -105.0  # 非足端接触强惩罚
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

    # --- 轮胎穿越任务专属权重：地形需要更宽松的姿态与更平滑的动作 ---
    rewards.applied_torque_limits.weight = -0.001  # 限制实际输出力矩，保护轮胎与电机
    rewards.applied_torque_limits.params["asset_cfg"].joint_names = cfg.joint_names
    rewards.stand_still.weight = -4.0  # 轮胎上允许轻微姿态调整，站立约束比 flat 放宽
    rewards.joint_pos_penalty.weight = -1.0  # 关节偏离默认姿态约束放宽，便于越障
    rewards.action_smoothness_2_l2.weight = -0.3  # 二阶动作平滑，减少轮胎上的冲击
    # stance_width 是左右轮心的全距，函数内部会 /2 得到每侧目标 ±stance_width/2。
    # dog_w1 轮心 nominal y≈±0.22，所以应写 0.44；写成 0.22 等于奖励把腿收窄一半。
    # side_sign 按 FL,FR,RL,RR 的奇偶分配 +Y/-Y，必须 preserve_order 才能保证顺序。
    rewards.feet_distance_y_exp.weight = 300.0  # 鼓励足够站宽，避免轮胎之间翻倒
    rewards.feet_distance_y_exp.params = {
        "stance_width": 0.48,  # 全距 0.48 = 每侧 ±0.24，与 dog_w1 nominal 轮距一致
        "std": 0.18,  # 与 flat 一致；0.22 太宽会让窄站姿也拿到高分
        "asset_cfg": SceneEntityCfg(
            "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
        ),
    }


def tires_configure_dog_w1_phase_swing_reward(cfg):
    """摆动相抬轮目标：轮心位置扣除实测轮半径；地形任务用高度扫描器取地面高度。"""
    phase_params = cfg.observations.policy.phase.params
    cfg.rewards.phase_swing_clearance = RewTerm(
        func=PhaseSwingClearance,
        weight=-8000.0,  # 摆动相高度误差惩罚，权重大保证抬轮干净
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
            "swing_height": 0.185,  # 摆动相目标离地高度
            "turn_height_bonus": 0.020,  # 原地转向时额外抬轮
            "terrain_sensor_cfg": SceneEntityCfg("height_scanner_base"),  # 地形任务用高度扫描器，平地不用
        },
    )


def tires_configure_rewards(cfg):
    """组装本任务奖励：相位目标 -> 权重 -> 接触/滑移，再追加任务专属项。"""
    tires_configure_dog_w1_phase_swing_reward(cfg)
    tires_configure_task_reward_values(cfg)

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

    # 轮胎穿越完成奖励：奖励持续前进穿过轮胎阵列，而不是原地站立。
    cfg.rewards.tire_traversal = RewTerm(
        func=TireTraversalReward,
        weight=30.0,
        params={"command_name": "base_velocity", "half_extent": 0.55, "min_speed": 0.2},
    )

    # 遇障响应：每轮每个轮胎每回合仅触发一次，有效期和累计奖励均受限。
    cfg.rewards.tire_obstacle_climb = RewTerm(
        func=TireObstacleClimbReward,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
            ),
            "joint_cfg": SceneEntityCfg(
                "robot", joint_names=["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"], preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True,
            ),
            **phase_params,  # 与观测共用相位时钟及全部参数
            "phase_offset": swing_params["phase_offset"],
            "stance_ratio": swing_params["stance_ratio"],
            "ahead_only": True,
            "approach_distance": 0.12,  # 沿指令方向前探，与真实轮胎环相交
            "response_duration": 0.8,  # 每轮每个轮胎触发后最多引导 0.8 秒
            "reward_budget": 0.4,  # 单轮积分上限；乘 weight / 4 后最多 1.5 分/轮/轮胎
            "speed_reference": 22.0,  # 有方向的目标轮速，超过目标同样扣减得分
            "wheel_radius": 0.0765,
            "speed_weight": 0.6,
            "lift_weight": 2.0,
            "lift_clearance": 0.03,  # 摆动相轮底目标 = 实际轮胎顶面 + 3 cm
            "lift_std": 0.05,  # 高度目标容差，过度抬高也降低得分
            "progress_threshold": 0.02,  # 机身实际速度、逐步位移均须超过该值 m/s
            "progress_reference": 0.2,  # 推进门控归一化速度 m/s
            "contact_threshold": 1.0,  # 持续接触且接触合力 > 1 N 才奖励驱动
            "slip_std": 0.3,  # 轮面滚动速度与轮心推进速度的误差容差 m/s
        },
    )
    # Keep the existing swing penalty consistent with obstacle lift guidance.
    cfg.rewards.phase_swing_clearance.params["max_ground_clearance"] = (
        TIRE_HEIGHT + cfg.rewards.tire_obstacle_climb.params["lift_clearance"]
    )



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
class ATDogDogW1TiresEnvCfg(LocomotionVelocityRoughEnvCfg):
    # HIM teacher uses clean privileged observations and metadata-derived slices.
    him_clean_critic = True
    actions: ATDogDogW1ActionsCfg = ATDogDogW1ActionsCfg()
    rewards: ATDogDogW1RewardsCfg = ATDogDogW1RewardsCfg()

    base_link_name = "base"
    foot_link_name = ".*_foot"

    # fmt: off
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene 场景与传感器------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = DOG_W1_TIRE_ARRAY_TERRAIN_CFG
        self.scene.robot = AT_DOG_W1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        tires_configure_observations(self)

        self.commands.base_velocity.class_type = DogW1VelocityCommand
        for key in ("command_name", "translation_threshold", "yaw_threshold"):
            setattr(self.actions.joint_vel, key, self.observations.policy.phase.params[key])

        # ------------------------------Actions------------------------------
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
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
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
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [self.foot_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.8, 2.2)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.6, 1.8)

        # ------------------------------Rewards------------------------------
        tires_configure_rewards(self)
        if self.__class__.__name__ == "ATDogDogW1TiresEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels = None
        self.curriculum.terrain_levels = None

        # ------------------------------Commands------------------------------
        # 保持默认速度命令范围，不额外缩放。
