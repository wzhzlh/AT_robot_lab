import ast
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2/tires_traversal.py"
tree = ast.parse(PATH.read_text())
node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "TireTraversalTracker")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATH), "exec"), namespace)
Tracker = namespace["TireTraversalTracker"]


class TireTraversalTest(unittest.TestCase):
    def setUp(self):
        self.tracker = Tracker(1, 1, "cpu")
        self.centers = torch.zeros(1, 1, 2)

    def step(self, x, y=0., command=(0.4, 0.), failed=False):
        return self.tracker.update(torch.tensor([[x, y]]), torch.tensor([command]), self.centers,
                                   0.55, 0.2, torch.tensor([failed])).item()

    def test_complete_once_and_reset(self):
        self.assertFalse(self.step(-0.7))
        self.assertFalse(self.step(-0.5))
        self.assertFalse(self.step(0.0))
        self.assertTrue(self.step(0.6))
        self.assertFalse(self.step(-0.7))
        self.assertFalse(self.step(-0.5))
        self.assertFalse(self.step(0.6))
        self.tracker.reset([0])
        self.assertFalse(self.step(-0.7))
        self.assertFalse(self.step(-0.5))
        self.assertTrue(self.step(0.6))

    def test_stand_bypass_and_termination_do_not_count(self):
        self.step(-0.7, command=(0., 0.))
        self.step(-0.5, command=(0., 0.))
        self.assertFalse(self.step(0.6, command=(0., 0.)))
        self.tracker.reset([0])
        self.step(-0.7, 0.8)
        self.step(-0.5, 0.8)
        self.assertFalse(self.step(0.6, 0.8))
        self.tracker.reset([0])
        self.step(-0.7)
        self.step(-0.5)
        self.assertFalse(self.step(0.0, 0.8))
        self.assertFalse(self.step(0.6))
        self.tracker.reset([0])
        self.step(-0.7)
        self.step(-0.5)
        self.assertFalse(self.step(0.6, failed=True))
