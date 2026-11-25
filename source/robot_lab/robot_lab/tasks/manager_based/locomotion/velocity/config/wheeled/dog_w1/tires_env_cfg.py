# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp  # noqa: F401
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from robot_lab.assets.atdog import AT_DOG_W1_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .stairs_env_cfg import (
    ATDogDogW1ActionsCfg,
    ATDogDogW1RewardsCfg,
    stairs_configure_observations,
    stairs_configure_rewards,
)
from .tires_traversal import TireTraversalReward


def tire_array_terrain(
    difficulty: float, cfg: MeshPlaneTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a dense field of randomly offset 2 by 2 flat annular tire arrays."""
    meshes, origin = mesh_terrains.flat_terrain(difficulty, cfg)
    outer_radius = 0.275
    inner_radius = 0.165
    tire_width = 0.142
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
                center_x = array_center_x + (column - 0.5) * 0.55
                center_y = array_center_y + (row - 0.5) * 0.55
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


tires_configure_observations = stairs_configure_observations


def tires_configure_rewards(cfg):
    """Use the stairs reward set, then add the tire-traversal completion reward."""
    stairs_configure_rewards(cfg)
    cfg.rewards.tire_traversal = RewTerm(
        func=TireTraversalReward,
        weight=30.0,
        params={"command_name": "base_velocity", "half_extent": 0.55, "min_speed": 0.2},
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

        # ------------------------------Actions------------------------------
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
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
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
