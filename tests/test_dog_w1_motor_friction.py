"""Measured friction torque and actuator integration checks, no simulator required."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTUATORS = ROOT / "source/robot_lab/robot_lab/actuators.py"


class MotorFrictionTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(ACTUATORS.read_text())
        names = {"directional_friction_torque", "_measured_friction", "FrictionDelayedPDActuator",
                 "FrictionImplicitActuator", "FrictionDelayedPDActuatorCfg", "FrictionImplicitActuatorCfg"}
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]

        class Base:
            def compute(self, action, joint_pos, joint_vel):
                return action

        self.ns = {"torch": SimpleNamespace(where=np.where, sign=np.sign),
                   "StreamDelayedPDActuator": Base, "ImplicitActuator": Base,
                   "DelayedPDActuatorCfg": object, "ImplicitActuatorCfg": object,
                   "configclass": lambda cls: cls}
        exec(compile(tree, str(ACTUATORS), "exec"), self.ns)
        self.cfg = self.ns["FrictionDelayedPDActuatorCfg"]()

    def test_measured_values_resist_both_directions_without_zero_speed_bias(self):
        class Velocity(np.ndarray):
            def abs(self):
                return np.abs(self)
        speed = np.array([-10., -1., 0., 1., 10.]).view(Velocity)
        torque = self.ns["_measured_friction"](self.cfg, speed)
        np.testing.assert_allclose(torque, [.334516608, .2244339648, 0., -.2420997756, -.338458983])
        self.assertTrue(np.all(torque * speed <= 0.))
        for name in ("FrictionDelayedPDActuatorCfg", "FrictionImplicitActuatorCfg"):
            cfg = self.ns[name]()
            self.assertEqual(cfg.friction_a_positive, .0107065786)
            self.assertEqual(cfg.friction_b_positive, .231393197)
            self.assertEqual(cfg.friction_a_negative, .0122314048)
            self.assertEqual(cfg.friction_b_negative, .21220256)

    def test_both_drives_add_resistance_without_changing_motor_diagnostics(self):
        class Velocity(np.ndarray):
            def abs(self):
                return np.abs(self)
        speed = np.array([[1., -1.]]).view(Velocity)
        for name in ("FrictionDelayedPDActuator", "FrictionImplicitActuator"):
            motor = self.ns[name]()
            motor.cfg = self.cfg
            motor.applied_effort = np.array([[2., -2.]])
            action = SimpleNamespace(joint_efforts=motor.applied_effort.copy(), joint_velocities=np.zeros((1, 2)))
            result = motor.compute(action, np.zeros((1, 2)), speed)
            np.testing.assert_allclose(result.joint_efforts, [[1.7579002244, -1.7755660352]])
            np.testing.assert_array_equal(motor.applied_effort, [[2., -2.]])
            np.testing.assert_array_equal(result.joint_velocities, np.zeros((1, 2)))

    def test_only_dog_w1_selects_measured_friction_drives(self):
        path = ROOT / "source/robot_lab/robot_lab/assets/atdog.py"
        tree = ast.parse(path.read_text())
        matches = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if not name.startswith("AT_DOG"):
                continue
            actuators = next((kw.value for kw in node.value.keywords if kw.arg == "actuators"), None)
            if actuators is None:
                continue
            matches[name] = [ast.unparse(value.func) for value in actuators.values]
        self.assertEqual(matches["AT_DOG_W1_CFG"].count("FrictionDelayedPDActuatorCfg"), 12)
        self.assertEqual(matches["AT_DOG_W1_CFG"].count("FrictionImplicitActuatorCfg"), 1)
        for name, types in matches.items():
            if name != "AT_DOG_W1_CFG":
                self.assertFalse(any(t.startswith("Friction") for t in types), name)


if __name__ == "__main__":
    unittest.main()
