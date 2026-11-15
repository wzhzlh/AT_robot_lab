"""Check stage terrain geometry, safe promotions, and campaign acceptance."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class Terrain:
    def _get_terrain_mesh(self, difficulty, cfg):
        return cfg, None

    def _add_sub_terrain(self, mesh, origin, row, col, cfg):
        self.records.append((row, col, cfg))


class CampaignTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/reinforcement_learning/rsl_rl/stairs_stage.py'
        tree = ast.parse(path.read_text())
        self.ns = dict(TerrainGenerator=Terrain, np=np, torch=torch, copy=copy)
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) or isinstance(n, ast.FunctionDef) and n.name == 'stable_curriculum']
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), self.ns)

    def test_height_bounds_and_flat_metadata(self):
        terrain = self.ns['StageTerrain']()
        terrain.cfg = NS(num_rows=3, num_cols=4, sub_terrains={
            'flat': NS(proportion=.25), 'stairs': NS(proportion=.75, step_height_range=(.07, .11)),
        })
        terrain.records = []
        terrain._generate_curriculum_terrains()
        self.assertEqual(terrain.cfg.flat_patch_mask, [True, False, False, False] * 3)
        for row, col, cfg in terrain.records:
            if col:
                self.assertAlmostEqual(cfg.step_height_range[0], [.07, .09, .11][row])
                self.assertEqual(cfg.step_height_range[0], cfg.step_height_range[1])

    def test_falls_never_promote_and_top_row_never_wraps(self):
        updates = []
        terrain = NS(terrain_levels=torch.tensor([0, 0, 2]), max_terrain_level=3)
        terrain.update_env_origins = lambda ids, up, down: updates.append((up.clone(), down.clone()))
        command = NS(_had_translation=torch.ones(3, dtype=torch.bool),
                     _start_valid=torch.ones(3, dtype=torch.bool),
                     metrics={'stair_progress': torch.ones(3) * 2., 'stair_height_change': torch.ones(3) * .3})
        env = NS(num_envs=3, device='cpu', scene=NS(terrain=terrain),
                 command_manager=NS(get_term=lambda name: command),
                 termination_manager=NS(terminated=torch.tensor([True, False, False])))
        for _ in range(2):
            self.ns['stable_curriculum'](env, torch.arange(3))
        self.assertEqual(updates[0][0].tolist(), [False, False, False])
        self.assertEqual(updates[1][0].tolist(), [False, True, False])
        self.assertEqual(updates[1][1].tolist(), [True, False, False])

    def test_acceptance_needs_all_seeds_and_sufficient_samples(self):
        path = ROOT / 'scripts/reinforcement_learning/rsl_rl/stairs_supervisor.py'
        tree = ast.parse(path.read_text())
        ns = {}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'passed'], type_ignores=[]), str(path), 'exec'), ns)
        good = [dict(episodes=120, success_rate=.95, fall_rate=.03) for _ in range(3)]
        self.assertTrue(ns['passed'](good))
        good[0]['fall_rate'] = .1
        self.assertFalse(ns['passed'](good))
        for entry in good:
            entry.update(episodes=10, fall_rate=.01)
        self.assertFalse(ns['passed'](good))


if __name__ == '__main__':
    unittest.main()
