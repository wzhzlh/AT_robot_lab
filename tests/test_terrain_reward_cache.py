"""Exercise the real terrain reward classes on CPU without starting Isaac Sim."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cache_test_gait", ROOT / "source/robot_lab/robot_lab/gait.py")
gait = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gait
spec.loader.exec_module(gait)


class ManagerTermBase:
    def __init__(self, cfg, env):
        self._env = env


class Scene(dict):
    @property
    def sensors(self):
        return self


class TerrainRewardCacheTests(unittest.TestCase):
    def setUp(self):
        self.rays = []

        def raycast(starts, directions, **kwargs):
            # An analytic 5 cm step makes the footprint straddle an edge.
            hits = starts.clone()
            hits[..., 2] = torch.where(starts[..., 0] >= 0.0, 0.05, 0.0)
            normals = torch.zeros_like(hits)
            normals[..., 2] = 1.0
            self.rays.append(hits)
            return hits, None, normals, None

        namespace = dict(
            torch=torch, ManagerTermBase=ManagerTermBase, RewTerm=object, ManagerBasedRLEnv=object,
            foot_geometry=gait.foot_geometry, swing_clearance_error=gait.swing_clearance_error,
            stair_edge_error=gait.stair_edge_error, raycast_mesh=raycast,
            RayCaster=SimpleNamespace(meshes={"terrain": object()}),
        )
        path = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
        tree = ast.parse(path.read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ("SwingClearancePenalty", "StairLandingPenalty")]
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
        asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3])
        scanner = SimpleNamespace(data=None, cfg=SimpleNamespace(mesh_prim_paths=["terrain"]))
        positions = torch.zeros(2, 4, 3)
        positions[..., 2] = 0.15
        quaternions = torch.zeros(2, 4, 4)
        quaternions[..., 0] = 1.0
        data = SimpleNamespace(body_link_pos_w=positions, body_link_quat_w=quaternions,
                               heading_w=torch.zeros(2))
        clock = gait.TrotClock(2, "cpu", gait.TrotClockCfg())
        command = torch.tensor([[0.4, 0, 0], [0, 0.4, 0]])
        clock.update(command, 0.02)
        self.env = SimpleNamespace(
            device="cpu", common_step_counter=1,
            scene=Scene({"robot": SimpleNamespace(data=data), "height_scanner": scanner,
                         "contact_forces": SimpleNamespace(data=SimpleNamespace(current_contact_time=torch.ones(2, 4)))}),
            command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(gait_clock=clock),
                                            get_command=lambda name: command),
        )
        common = dict(asset_cfg=asset_cfg, terrain_sensor_cfg=SimpleNamespace(name="height_scanner"),
                      command_name="base_velocity", foot_offsets=((0.0, 0.0, 0.0),) * 4,
                      foot_radius=0.0384, foot_half_width=0.014)
        self.swing_params = dict(common, height=0.12, std=0.06, terrain_sample_radius=0.0384,
                                 lookahead_distance=0.08, deficit_only=True)
        self.landing_params = dict(common, sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3]),
                                   edge_tolerance=0.015)
        self.swing = namespace["SwingClearancePenalty"](SimpleNamespace(params=self.swing_params), self.env)
        self.landing = namespace["StairLandingPenalty"](SimpleNamespace(params=self.landing_params), self.env)

    def test_shared_query_matches_independent_loaded_edge_reward(self):
        self.swing(self.env, **self.swing_params)
        shared = self.landing(self.env, **self.landing_params)
        self.assertEqual(len(self.rays), 1)
        self.env._gait_terrain_cache = None
        independent = self.landing(self.env, **self.landing_params)
        self.assertEqual(len(self.rays), 2)
        torch.testing.assert_close(self.rays[0][..., :5, :], self.rays[1], rtol=0, atol=0)
        torch.testing.assert_close(shared, independent, rtol=0, atol=0)
        torch.testing.assert_close(shared, torch.ones(2))

    def test_step_reset_and_radius_changes_invalidate_cache(self):
        for change in ("step", "reset", "radius"):
            with self.subTest(change=change):
                self.swing(self.env, **self.swing_params)
                if change == "step":
                    self.env.common_step_counter += 1
                elif change == "reset":
                    self.swing.reset()
                    self.assertIsNone(self.env._gait_terrain_cache)
                else:
                    entry = self.env._gait_terrain_cache
                    self.env._gait_terrain_cache = (*entry[:2], 0.01, entry[3])
                before = len(self.rays)
                self.landing(self.env, **self.landing_params)
                self.assertEqual(len(self.rays), before + 1)


if __name__ == "__main__":
    unittest.main()
