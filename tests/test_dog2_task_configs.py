import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2"
TASKS = ("flat", "flat_domain_rand", "rough", "stairs", "sand", "slope", "bar", "tires")


class TaskConfigTest(unittest.TestCase):
    def test_tires_uses_terrain_relative_swing_without_world_height_penalty(self):
        tree = ast.parse((ROOT / "tires_env_cfg.py").read_text())
        swing = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "tires_configure_dog2_phase_swing_reward")
        reward = swing.body[0].value
        params = next(keyword.value for keyword in reward.keywords if keyword.arg == "params")
        values = {ast.literal_eval(key): value for key, value in zip(params.keys, params.values)}
        self.assertEqual(ast.literal_eval(values["terrain_sensor_cfg"].args[0]), "height_scanner_base")
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "tires_configure_task_reward_values")
        weights = [node.value for node in function.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and ast.unparse(node.targets[0]) == "self.rewards.feet_height.weight"]
        self.assertEqual(len(weights), 1)
        self.assertEqual(ast.literal_eval(weights[0]), 0.0)

    def test_stairs_world_direction_is_preserved_after_turning(self):
        path = ROOT / "stairs_env_cfg.py"
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "StairsTurnVelocityCommand")

        class BaseCommand:
            def _update_command(self):
                pass

        namespace = {"mdp": SimpleNamespace(StairsVelocityCommand=BaseCommand), "torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        command = namespace["StairsTurnVelocityCommand"]()
        command._env = SimpleNamespace(_stair_flat_mask=torch.tensor([False, False, False, True]))
        command.robot = SimpleNamespace(data=SimpleNamespace(heading_w=torch.tensor([0., math.pi / 2, math.pi, 1.])))
        command._travel_direction = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 0.]])
        command.vel_command_b = torch.tensor([[0.4, 0., 0.], [0.4, 0., 0.], [0., 0., 0.5], [0.2, 0.3, 0.]])
        command._update_command()
        torch.testing.assert_close(command.vel_command_b, torch.tensor(
            [[0.4, 0., 0.], [0., -0.4, 0.], [0., 0., 0.5], [0.2, 0.3, 0.]]), atol=1e-6, rtol=0.)
        command.robot.data.heading_w[1] = math.pi
        command._update_command()
        torch.testing.assert_close(command.vel_command_b[1, :2], torch.tensor([-0.4, 0.]), atol=1e-6, rtol=0.)

    def test_stairs_disables_world_frame_feet_height(self):
        tree = ast.parse((ROOT / "stairs_env_cfg.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "stairs_configure_task_reward_values")
        weights = [node.value for node in function.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and ast.unparse(node.targets[0]) == "self.rewards.feet_height.weight"]
        self.assertEqual(len(weights), 1)
        self.assertEqual(ast.literal_eval(weights[0]), 0.0)

    def test_task_reward_weights_are_not_assigned_in_environment_before_configuration(self):
        for task in ("rough", "stairs", "sand", "slope", "bar", "tires"):
            with self.subTest(task=task):
                tree = ast.parse((ROOT / f"{task}_env_cfg.py").read_text())
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                and node.name == f"{task}_configure_task_reward_values")
                weights = {node.targets[0].value.attr for node in function.body
                           if isinstance(node, ast.Assign) and len(node.targets) == 1
                           and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == "weight"}
                env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                                 and node.name.endswith(task.capitalize() + "EnvCfg"))
                init = next(node for node in env_class.body if isinstance(node, ast.FunctionDef)
                            and node.name == "__post_init__")
                for node in init.body:
                    if isinstance(node, ast.Assign) and len(node.targets) == 1:
                        target = node.targets[0]
                        if (isinstance(target, ast.Attribute) and target.attr == "weight"
                                and isinstance(target.value, ast.Attribute)
                                and isinstance(target.value.value, ast.Attribute)
                                and target.value.value.attr == "rewards"):
                            self.assertNotIn(target.value.attr, weights)

    def test_each_task_owns_observation_and_reward_configuration(self):
        for task in TASKS:
            with self.subTest(task=task):
                path = ROOT / f"{'flat' if task == 'flat_domain_rand' else task}_env_cfg.py"
                tree = ast.parse(path.read_text())
                definitions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                self.assertTrue({f"{task}_{name}" for name in ("configure_observations", "configure_rewards",
                                 "configure_task_reward_values", "csdog1_phase_term",
                                 "configure_dog2_phase_swing_reward", "CSDog1PolicyCfg")} <= definitions)
                self.assertNotIn("_task_cfg import", path.read_text())

    def test_environment_classes_use_explicit_task_configs(self):
        for task in ("rough", "stairs", "sand", "slope", "bar", "tires"):
            with self.subTest(task=task):
                source = (ROOT / f"{task}_env_cfg.py").read_text()
                self.assertIn(f"{task}_configure_observations(self)", source)
                self.assertIn(f"{task}_configure_rewards(self)", source)
        source = (ROOT / "flat_env_cfg.py").read_text()
        for task in ("flat", "flat_domain_rand"):
            self.assertIn(f"{task}_configure_observations(self)", source)
            self.assertIn(f"{task}_configure_rewards(self)", source)


if __name__ == "__main__":
    unittest.main()
