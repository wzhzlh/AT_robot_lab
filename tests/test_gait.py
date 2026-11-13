"""Numerical gait regressions; run with a PyTorch interpreter, no simulator."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import torch

# Load the deployment reference directly, avoiding robot_lab's simulator registration.
GAIT_PATH = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/gait.py"
spec = importlib.util.spec_from_file_location("atdog2_gait_reference", GAIT_PATH)
gait = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gait
spec.loader.exec_module(gait)


class GaitTests(unittest.TestCase):
    def test_slower_default_clock_and_quiet_joint_gating(self):
        clock = gait.TrotClock(1, "cpu", gait.TrotClockCfg())
        for _ in range(35):
            clock.update(torch.tensor([[0.5, 0, 0]]), 0.02)
        torch.testing.assert_close(clock.observation(), torch.tensor([[0.0, 1.0]]), atol=4e-6, rtol=0)
        velocity = torch.ones(3, 12) * 0.25
        delta = torch.ones(3, 12) * 0.025
        penalty = gait.standing_motion_error(velocity, delta, torch.tensor([False, True, True]))
        torch.testing.assert_close(penalty, torch.tensor([0.5, 0, 0]))
        torch.testing.assert_close(
            gait.standing_motion_error(torch.zeros_like(velocity), torch.zeros_like(delta), torch.zeros(3, dtype=torch.bool)),
            torch.zeros(3),
        )

    def test_fixed_frequency_wrap_and_encoding(self):
        clock = gait.TrotClock(2, "cpu", gait.TrotClockCfg(cycle_time=0.5))
        command = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.4]])
        for _ in range(25):
            clock.update(command, 0.02)
        self.assertTrue(torch.all(clock.moving))
        torch.testing.assert_close(clock.observation(), torch.tensor([[0.0, 1.0], [0.0, 1.0]]), atol=3e-6, rtol=0)
        # Reading targets/observations must not advance the clock.
        before = clock.phase.clone()
        clock.foot_targets()
        clock.observation()
        torch.testing.assert_close(before, clock.phase)

    def test_independent_linear_and_angular_hysteresis(self):
        clock = gait.TrotClock(2, "cpu", gait.TrotClockCfg(cycle_time=0.5))
        for speed, yaw, expected in ((0.06, 0.08, False), (0.09, 0.11, True),
                                     (0.06, 0.08, True), (0.03, 0.04, False)):
            clock.update(torch.tensor([[speed, 0, 0], [0, 0, yaw]]), 0.02)
            self.assertEqual(clock.moving.tolist(), [expected, expected])

    def test_stop_at_next_landing_and_restart_without_jump(self):
        # Sweep every starting phase, including both swing groups and double support.
        clock = gait.TrotClock(1000, "cpu", gait.TrotClockCfg(cycle_time=0.5))
        clock.phase[:] = torch.arange(1000) / 1000
        clock.translating[:] = True
        original = clock.phase.clone()
        for _ in range(14):
            previous = clock.phase.clone()
            clock.update(torch.zeros(1000, 3), 0.02)
            delta = torch.remainder(clock.phase - previous, 1.0)
            self.assertTrue(torch.all(delta <= 0.040001))
        self.assertFalse(clock.active.any())
        _, swing, _ = clock.foot_targets()
        self.assertTrue(torch.all(swing < 1e-5))
        # Do not erase the half-cycle landing to phase zero.
        self.assertTrue(torch.any(clock.phase >= 0.5))
        frozen = clock.phase.clone()
        clock.update(torch.zeros(1000, 3), 0.02)
        torch.testing.assert_close(frozen, clock.phase)
        clock.update(torch.tensor([[0.5, 0, 0]]).expand(1000, -1), 0.02)
        torch.testing.assert_close(torch.remainder(clock.phase - frozen, 1), torch.full((1000,), 0.04))
        self.assertTrue(torch.any(original != frozen))

    def test_partial_reset_does_not_change_other_envs(self):
        clock = gait.TrotClock(3, "cpu", gait.TrotClockCfg(cycle_time=0.5))
        clock.update(torch.ones(3, 3), 0.02)
        clock.reset(torch.tensor([1]))
        torch.testing.assert_close(clock.phase, torch.tensor([0.04, 0, 0.04]))
        self.assertEqual(clock.moving.tolist(), [True, False, True])

    def test_contact_degenerate_patterns_and_double_support(self):
        cfg = gait.TrotClockCfg(cycle_time=0.5)
        phase = torch.arange(1000) / 1000
        stance, swing, _ = gait.trot_targets(phase, cfg)
        active = torch.ones(1000, dtype=torch.bool)
        ideal = stance > 0
        correct = gait.contact_violation(ideal, stance, swing, active)
        torch.testing.assert_close(correct, torch.zeros_like(correct))
        for pattern in ((1, 1, 1, 1), (0, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0)):
            contact = torch.tensor(pattern).expand(1000, -1)
            penalty = gait.contact_violation(contact, stance, swing, active)
            self.assertGreater(penalty.mean().item(), 0.2)
        opposite = gait.contact_violation(~ideal, stance, swing, active)
        self.assertGreater(opposite.mean().item(), 0.7)
        stance, swing, _ = gait.trot_targets(torch.tensor([0.05, 0.55]), cfg)
        double_support = gait.contact_violation(torch.ones(2, 4), stance, swing, torch.ones(2))
        torch.testing.assert_close(double_support, torch.zeros(2))

    def test_clearance_phase_locality_terrain_invariance_invalid_hits(self):
        phase = torch.tensor([0.3])
        _, swing, progress = gait.trot_targets(phase, gait.TrotClockCfg(cycle_time=0.5))
        target = 0.08 * torch.sin(math.pi * progress).square()
        valid = torch.ones(1, 4, dtype=torch.bool)
        def error(clearance, validity=valid):
            return gait.swing_clearance_error(clearance, progress, swing, validity, torch.ones(1), 0.08, 0.04)
        torch.testing.assert_close(error(target), torch.zeros(1))
        self.assertGreater(error(torch.zeros_like(target)).item(), 0.1)
        changed_stance = target + 10.0 * (swing == 0)
        torch.testing.assert_close(error(changed_stance), torch.zeros(1))
        raised_terrain = target + 0.7
        torch.testing.assert_close(error(raised_terrain - 0.7), error(target), atol=1e-5, rtol=0)
        torch.testing.assert_close(error(torch.full_like(target, float("inf")), ~valid), torch.zeros(1))

    def test_foot_offset_rotates_and_sole_extent_uses_orientation(self):
        positions = torch.zeros(1, 2, 3)
        quats = torch.tensor([[[1.0, 0, 0, 0], [math.sqrt(0.5), 0, math.sqrt(0.5), 0]]])
        offsets = torch.tensor([[0.22, 0, 0], [0.22, 0, 0]])
        center, extent = gait.foot_geometry(positions, quats, offsets, 0.0384, 0.014)
        torch.testing.assert_close(center, torch.tensor([[[0.22, 0, 0], [0, 0, -0.22]]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(extent, torch.tensor([[0.014, 0.0384]]), atol=1e-6, rtol=0)

    def test_configuration_validation(self):
        for kwargs in ({"cycle_time": 0}, {"duty_factor": 0.5}, {"transition_width": 0.3},
                       {"linear_enter": 0.01, "linear_exit": 0.02}):
            with self.assertRaises(ValueError):
                gait.TrotClockCfg(**kwargs)

    def test_deployment_metadata_restores_nondefault_parameters(self):
        cfg = gait.TrotClockCfg(cycle_time=0.7, duty_factor=0.65, angular_enter=0.2)
        reference = gait.TrotClock(2, "cpu", cfg)
        deploy = gait.TrotClock.from_metadata(2, "cpu", reference.metadata())
        self.assertEqual(deploy.cfg, cfg)
        for values in ((0.5, 0, 0), (0, 0, 0.3), (0, 0, 0)):
            for _ in range(20):
                command = torch.tensor([values, values], dtype=torch.float32)
                reference.update(command, 0.02)
                deploy.update(command, 0.02)
                torch.testing.assert_close(reference.observation(), deploy.observation())


if __name__ == "__main__":
    unittest.main()
