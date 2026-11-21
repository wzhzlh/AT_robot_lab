import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


PATH = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2/rewards.py"


class PhaseTerrainTest(unittest.TestCase):
    def test_clearance_is_relative_to_each_foot_terrain_and_invalid_rays_are_masked(self):
        tree = ast.parse(PATH.read_text())
        node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "PhaseSwingClearance")

        class Manager:
            def __init__(self, cfg, env):
                pass

        class Clock:
            def __init__(self, cfg, env):
                pass

            def __call__(self, env, *args):
                return torch.tensor([[0., -1.]]).repeat(env.num_envs, 1)

        ground = torch.tensor([[0., 0.11, 0.22, 0.33], [0., 0.11, 0.22, float('inf')]])

        def raycast(starts, directions, **kwargs):
            hits = starts.clone()
            hits[..., 2] = ground
            normals = torch.zeros_like(hits)
            normals[..., 2] = 1.
            return hits, None, normals, None

        namespace = {"torch": torch, "math": math, "ManagerTermBase": Manager, "CSDog1Phase": Clock,
                     "SceneEntityCfg": object, "RayCaster": SimpleNamespace(meshes={"ground": object()}),
                     "raycast_mesh": raycast, "quat_apply": lambda quat, offsets: torch.zeros_like(offsets),
                     "moving_command": lambda command: command[:, 0] != 0}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATH), "exec"), namespace)
        terrain = SimpleNamespace(cfg=SimpleNamespace(mesh_prim_paths=["ground"]))
        positions = torch.zeros(2, 4, 3)
        positions[0, :, 2] = torch.tensor([0., 0.11, 0.22, 0.33]) + 0.07
        positions[1, :, 2] = torch.tensor([0., 0.11, 0.22, 0.33]) + 0.07
        robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=positions, body_quat_w=torch.zeros(2, 4, 4),
                                                     root_pos_w=torch.ones(2, 3)))
        class Scene(dict):
            pass
        scene = Scene(robot=robot)
        scene.sensors = {"terrain": terrain}
        command = torch.tensor([[0.5, 0., 0.], [0., 0., 0.]])
        env = SimpleNamespace(num_envs=2, device="cpu", scene=scene,
                              command_manager=SimpleNamespace(get_command=lambda _: command))
        asset = SimpleNamespace(name="robot", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"],
                                body_ids=[0, 1, 2, 3])
        sensor = SimpleNamespace(name="terrain")
        term = namespace["PhaseSwingClearance"](SimpleNamespace(params={"asset_cfg": asset,
                                                                        "terrain_sensor_cfg": sensor}), env)
        result = term(env, asset, "base_velocity", 0.5, 0.5, 0.5, 0.05, 0.05, 0.5, 0.55, 0.07,
                      0.02, 0.10, 0.015, terrain_sensor_cfg=sensor)
        torch.testing.assert_close(term.debug_metrics["foot_clearance"][0], torch.full((4,), 0.05))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result[1].item(), 0.)
        self.assertFalse(term.debug_metrics["terrain_valid"][1, 3].item())


if __name__ == "__main__":
    unittest.main()
