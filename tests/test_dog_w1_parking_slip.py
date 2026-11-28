"""Wheel tread kinematics and parking brake regression checks."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from test_dog_w1_gait import load_namespace, make_env, torch


TASK = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1"


class ParkingSlipTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_namespace()
        if not hasattr(torch, "cross"):
            torch.cross = lambda a, b, dim: np.cross(a, b, axis=dim)
        tree = ast.parse((TASK / "actions.py").read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(tree, "actions.py", "exec"), self.ns)

    def close(self, value, expected):
        np.testing.assert_allclose(np.asarray(value), np.asarray(expected), atol=1e-6)

    def test_both_wheel_signs_roll_without_slip_and_spin_is_detected(self):
        # Mirrored left/right spin, identical +X wheel-center translation.
        position = torch.tensor([[[0., .2, .08], [0., -.2, .08]]])
        velocity = torch.tensor([[[.8, 0., 0.], [.8, 0., 0.]]])
        angular = torch.tensor([[[0., 10., 0.], [0., 10., 0.]]])
        func = self.ns["wheel_contact_velocity"]
        self.close(func(position, position, velocity, angular, .08), torch.zeros(1, 2, 3))
        self.close(func(position, position, torch.zeros(1, 2, 3), angular, .08)[..., 0], [[-.8, -.8]])
        self.close(func(position, position, velocity, torch.zeros(1, 2, 3), .08)[..., 0], [[.8, .8]])

    def test_com_offset_is_accounted_for(self):
        # COM shifted +1 cm in X, angular Z=2 gives a +Y COM velocity .02.
        center = torch.tensor([[[0., 0., .08]]])
        com = torch.tensor([[[.01, 0., .08]]])
        result = self.ns["wheel_contact_velocity"](
            center, com, torch.tensor([[[0., .02, 0.]]]), torch.tensor([[[0., 0., 2.]]]), .08,
        )
        self.close(result, [[[0., 0., 0.]]])

    def test_stop_captures_angle_holds_releases_and_recaptures(self):
        step = self.ns["parking_brake_step"]
        pos = torch.tensor([[1., 2.]])
        vel = torch.zeros(1, 2)
        moving = torch.tensor([False])
        locked, hold, target, effort = step(pos, vel, moving, torch.tensor([False]),
                                           torch.zeros(1, 2), torch.tensor([[5., -5.]]), 20., 2., 15.)
        self.close(hold, pos)
        self.close(target, [[0., 0.]])
        self.close(effort, [[0., 0.]])
        locked, hold, target, effort = step(pos + .1, vel, moving, locked, hold, target, 20., 2., 15.)
        self.close(hold, pos)
        self.close(effort, [[-2., -2.]])
        _, _, _, effort = step(pos + 10., vel, moving, locked, hold, target, 20., 2., 15.)
        self.close(effort, [[-15., -15.]])
        locked, hold, target, effort = step(pos, vel, torch.tensor([True]), locked, hold,
                                           torch.tensor([[4., -4.]]), 20., 2., 15.)
        self.close(target, [[4., -4.]])
        self.close(effort, [[0., 0.]])
        _, hold, _, _ = step(pos + 2., vel, moving, locked, hold, target, 20., 2., 15.)
        self.close(hold, pos + 2.)

    def test_stationary_gate_allows_pure_yaw_and_lateral_motion(self):
        env = make_env()
        env.scene["robot"].data.joint_vel = torch.tensor([[2., -2.], [2., -2.]])
        cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])
        env.command[:] = 0.
        self.close(self.ns["stationary_wheel_velocity"](env, cfg, "base_velocity"), [4., 4.])
        env.command[0, 2] = .3
        env.command[1, 1] = .3
        self.close(self.ns["stationary_wheel_velocity"](env, cfg, "base_velocity"), [0., 0.])

    def test_dog_command_preserves_yaw_without_changing_other_tasks(self):
        class Upstream:
            def _update_command(self):
                self.vel_command_b[:] = self.test_command
        class Existing(Upstream):
            def _update_command(self):
                super()._update_command()
                self.vel_command_b[:, 2] = 0.
        tree = ast.parse((TASK / "commands.py").read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        ns = {"UniformVelocityCommand": Upstream, "UniformThresholdVelocityCommand": Existing}
        exec(compile(tree, "commands.py", "exec"), ns)
        command = ns["DogW1VelocityCommand"]()
        command.vel_command_b = torch.zeros(2, 3)
        command.test_command = torch.tensor([[0., 0., .5], [0., 0., 0.]])
        command._update_command()
        self.close(command.vel_command_b, command.test_command)


if __name__ == "__main__":
    unittest.main()
