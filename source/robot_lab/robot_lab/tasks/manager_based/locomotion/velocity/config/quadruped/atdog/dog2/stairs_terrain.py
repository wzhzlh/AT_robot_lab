"""Stair patches with explicit flat/stair metadata for mixed training."""

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass


class ATDog2StairsTerrainGenerator(terrain_gen.TerrainGenerator):
    def _generate_random_terrains(self):
        count = self.cfg.num_rows * self.cfg.num_cols
        flat_count = round(count * self.cfg.flat_ratio)
        up_count = round((count - flat_count) * self.cfg.up_ratio)
        patch_types = np.array(
            [0] * flat_count + [1] * up_count + [2] * (count - flat_count - up_count), dtype=np.int8
        )
        self.np_rng.shuffle(patch_types)
        self.cfg.flat_patch_mask = (patch_types == 0).tolist()
        patches = list(self.cfg.sub_terrains.values())
        for index, kind in enumerate(patch_types):
            row, col = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            mesh, origin = self._get_terrain_mesh(0.0, patches[int(kind)])
            self._add_sub_terrain(mesh, origin, row, col, patches[int(kind)])


@configclass
class ATDog2StairsTerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    class_type = ATDog2StairsTerrainGenerator
    flat_ratio: float = 0.30
    up_ratio: float = 0.65
    flat_patch_mask: list[bool] = []
