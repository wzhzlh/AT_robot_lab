"""Stair patches with explicit flat/stair metadata for mixed training."""

import copy
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

    def _generate_curriculum_terrains(self):
        """Generate curriculum rows using configured heights and terrain ratios."""
        proportions = np.array([self.cfg.flat_ratio, (1 - self.cfg.flat_ratio) * self.cfg.up_ratio,
                                (1 - self.cfg.flat_ratio) * (1 - self.cfg.up_ratio)])
        sub_indices = [int(np.min(np.where(col / self.cfg.num_cols + 0.001 < np.cumsum(proportions))))
                       for col in range(self.cfg.num_cols)]
        configs = list(self.cfg.sub_terrains.values())
        flat_mask = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=bool)
        for col, sub_index in enumerate(sub_indices):
            for row in range(self.cfg.num_rows):
                base_cfg = configs[sub_index]
                cfg = copy.deepcopy(base_cfg)
                is_flat = sub_index == 0
                if hasattr(cfg, "step_height_range"):
                    fraction = row / max(self.cfg.num_rows - 1, 1)
                    low, high = cfg.step_height_range
                    height = low + (high - low) * fraction
                    cfg.step_height_range = (height, height)
                flat_mask[row, col] = is_flat
                mesh, origin = self._get_terrain_mesh(row / max(self.cfg.num_rows - 1, 1), cfg)
                self._add_sub_terrain(mesh, origin, row, col, cfg)
        self.cfg.flat_patch_mask = flat_mask.reshape(-1).tolist()


@configclass
class ATDog2StairsTerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    class_type = ATDog2StairsTerrainGenerator
    flat_ratio: float = 0.30
    up_ratio: float = 0.65
    flat_patch_mask: list[bool] = []
