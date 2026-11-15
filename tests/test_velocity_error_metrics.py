"""Exercise actual command metrics without starting the simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class BaseCommand:
    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        result = {key: value[ids].mean().item() for key, value in self.metrics.items()}
        for value in self.metrics.values():
            value[ids] = 0
        return result


class VelocityErrorTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PhaseVelocityCommand')
        ns = dict(mdp=SimpleNamespace(UniformVelocityCommand=BaseCommand))
        code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
        ns['torch'] = torch
        exec(compile(ast.fix_missing_locations(code), str(path), 'exec'), ns)
        self.command = object.__new__(ns['PhaseVelocityCommand'])
        c = self.command
        c.num_envs = 2
        c.device = 'cpu'
        c.command_counter = torch.zeros(2, dtype=torch.long)
        c._resample = lambda ids: None
        c.vel_command_b = torch.zeros(2, 3)
        c.robot = SimpleNamespace(data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([[.3, .4, 0.], [.3, .4, 0.]]),
            root_ang_vel_b=torch.tensor([[0., 0., -.2], [0., 0., -.2]]),
        ))
        c._velocity_error_sum = torch.zeros(2, 2)
        c._velocity_error_samples = torch.zeros(2, dtype=torch.long)
        c.metrics = dict(error_vel_xy=torch.zeros(2), error_vel_yaw=torch.zeros(2))
        c.gait_clock = SimpleNamespace(reset=lambda ids: None)

    def test_short_and_long_episodes_have_same_constant_error(self):
        c = self.command
        for _ in range(10):
            c._update_metrics()
        first = c.reset(torch.tensor([0]))
        for _ in range(10):
            c._update_metrics()
        second = c.reset(torch.tensor([1]))
        for result in (first, second):
            self.assertAlmostEqual(float(result['error_vel_xy']), .5)
            self.assertAlmostEqual(float(result['error_vel_yaw']), .2)
        self.assertEqual(c._velocity_error_samples.tolist(), [10, 0])

    def test_variable_errors_and_full_reset(self):
        c = self.command
        c._update_metrics()
        c.robot.data.root_lin_vel_b.zero_()
        c.robot.data.root_ang_vel_b.zero_()
        c._update_metrics()
        result = c.reset()
        self.assertAlmostEqual(float(result['error_vel_xy']), .25)
        self.assertAlmostEqual(float(result['error_vel_yaw']), .1)
        self.assertEqual(c._velocity_error_samples.tolist(), [0, 0])
        self.assertEqual(c.reset()['error_vel_xy'], 0.)

    def test_standing_mask_matches_upstream_selection_and_clock_input(self):
        c = self.command
        c.cfg = SimpleNamespace(heading_command=False)
        c.is_standing_env = torch.tensor([False, True])
        c.vel_command_b.copy_(torch.tensor([[.4, -.1, .3], [.5, .2, -.2]]))
        expected = c.vel_command_b.clone()
        expected[c.is_standing_env.nonzero().flatten()] = 0.
        captured = []
        c._env = SimpleNamespace(step_dt=.02)
        c.gait_clock = SimpleNamespace(update=lambda command, dt: captured.append((command.clone(), dt)))
        c._update_command()
        torch.testing.assert_close(c.vel_command_b, expected, rtol=0, atol=0)
        torch.testing.assert_close(captured[0][0], expected, rtol=0, atol=0)
        self.assertEqual(captured[0][1], .02)

    def test_reset_preserves_reduction_and_resampling_order(self):
        c = self.command
        c._update_metrics()
        expected = {key: value[1:].mean().item() for key, value in c.metrics.items()}
        c.command_counter[:] = 5
        calls = []
        def resample(ids):
            calls.append(ids.tolist())
            self.assertEqual(c.command_counter[ids].tolist(), [0])
            self.assertTrue(all(value[ids].eq(0).all() for value in c.metrics.values()))
            c.command_counter[ids] += 1
        c._resample = resample
        result = c.reset(torch.tensor([1]))
        self.assertEqual(calls, [[1]])
        self.assertEqual(c.command_counter.tolist(), [5, 1])
        for key, value in result.items():
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(value.item(), expected[key])


if __name__ == '__main__':
    unittest.main()
