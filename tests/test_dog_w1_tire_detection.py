"""Exercise actual detector geometry without starting Isaac Sim."""
import ast
import math
from pathlib import Path
import unittest

from test_dog_w1_gait import torch

PATH = Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1/tires_traversal.py'
tree = ast.parse(PATH.read_text())
nodes = [n for n in tree.body if isinstance(n, ast.Assign) or
         isinstance(n, ast.FunctionDef) and n.name == 'tire_approach_mask']
ns = {'torch': torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), ns)


class TireDetectionTest(unittest.TestCase):
    def hit(self, xy, command=(1., 0.), heading=0., reach=.12, origin=(0., 0.)):
        wheel = torch.tensor([[[xy[0] + origin[0], xy[1] + origin[1]]]])
        return bool(ns['tire_approach_mask'](
            wheel, torch.tensor([[origin]]), torch.tensor([command]), torch.tensor([heading]),
            reach, .0765, .05)[0, 0])

    def test_front_edge_not_group_center_radius(self):
        self.assertTrue(self.hit((-.45, 0.)))
        self.assertFalse(self.hit((-.49, 0.)))

    def test_hole_requires_reachable_material(self):
        self.assertFalse(self.hit((0., 0.), reach=.05))
        self.assertTrue(self.hit((0., 0.), reach=.12))

    def test_behind_side_and_no_translation(self):
        self.assertFalse(self.hit((.30, 0.)))
        self.assertFalse(self.hit((-.30, .40)))
        self.assertFalse(self.hit((-.30, 0.), command=(0., 0.)))

    def test_heading_reverse_lateral_and_world_offset(self):
        self.assertTrue(self.hit((0., -.45), heading=math.pi / 2))
        self.assertTrue(self.hit((.45, 0.), command=(-1., 0.)))
        self.assertTrue(self.hit((0., -.45), command=(0., 1.)))
        self.assertTrue(self.hit((-.45, 0.), origin=(12., -7.)))


if __name__ == '__main__':
    unittest.main()
