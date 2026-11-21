import ast
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2"


def moving_command(command, translation_threshold=0.05, yaw_threshold=0.05):
    return ((torch.linalg.vector_norm(command[:, :2], dim=-1) >= translation_threshold)
            | (command[:, 2].abs() >= yaw_threshold))


class ManagerTerm:
    def __init__(self, cfg, env):
        pass


class StandingTest(unittest.TestCase):
    def test_phase_holds_zero_immediately_after_stop(self):
        managers = ModuleType("isaaclab.managers")
        managers.ManagerTermBase = ManagerTerm
        managers.ObservationTermCfg = object
        actions = ModuleType("isaaclab.envs.mdp.actions")
        actions.JointPositionAction = object
        spec = importlib.util.spec_from_file_location("dog2_phase", ROOT / "phase_observations.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"isaaclab.managers": managers, "isaaclab.envs.mdp.actions": actions}):
            spec.loader.exec_module(module)
        command = torch.zeros(2, 3)
        env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.02, common_step_counter=0,
                              command_manager=SimpleNamespace(get_command=lambda _: command))
        phase = module.CSDog1Phase(None, env)
        params = dict(command_name="base_velocity", cycle_time=0.5, stop_cycle_time=0.5,
                      stop_duration=0.5, translation_threshold=0.05, yaw_threshold=0.05)
        for step in range(30):
            env.common_step_counter = step
            torch.testing.assert_close(phase(env, **params), torch.tensor([[0., 1.], [0., 1.]]))
        command[0, 0] = 0.5
        env.common_step_counter += 1
        phase(env, **params)
        command.zero_()
        torch.testing.assert_close(phase(env, **params), torch.tensor([[0., 1.], [0., 1.]]))
        for _ in range(26):
            env.common_step_counter += 1
            result = phase(env, **params)
        torch.testing.assert_close(result, torch.tensor([[0., 1.], [0., 1.]]))

    def test_standing_contact_rewards_sustained_contact_not_touchdowns(self):
        tree = ast.parse((ROOT / "rewards.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "standing_contact")
        namespace = {"torch": torch, "SceneEntityCfg": object, "moving_command": moving_command}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "rewards.py"), "exec"), namespace)
        command = torch.tensor([[0., 0., 0.], [0.2, 0., 0.]])
        sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=torch.tensor([[1., 1., 0., 1.], [1., 1., 1., 1.]])))
        robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]])))
        class Scene(dict):
            pass
        scene = Scene(robot=robot)
        scene.sensors = {"contact": sensor}
        env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: command))
        cfg = SimpleNamespace(name="contact", body_ids=[0, 1, 2, 3])
        for _ in range(2):
            torch.testing.assert_close(namespace["standing_contact"](env, "base_velocity", cfg), torch.tensor([3., 0.]))

    def test_stationary_velocity_penalty_only_applies_to_small_commands(self):
        tree = ast.parse((ROOT / "rewards.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "standing_joint_velocity")
        namespace = {"torch": torch, "SceneEntityCfg": object, "moving_command": moving_command}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "rewards.py"), "exec"), namespace)
        command = torch.tensor([[0., 0., 0.], [0.2, 0., 0.], [0., 0., 0.1]])
        robot = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.tensor([[2., -2.], [2., -2.], [2., -2.]])))
        env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(get_command=lambda _: command))
        cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])
        torch.testing.assert_close(namespace["standing_joint_velocity"](env, "base_velocity", cfg), torch.tensor([4., 0., 0.]))

    def test_gait_is_disabled_at_zero_command_even_when_base_moves(self):
        tree = ast.parse((ROOT / "rewards.py").read_text())
        class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CommandedGaitReward")
        class GaitReward:
            def __call__(self, env, *args, **kwargs):
                return torch.ones(2)
        namespace = {"torch": torch, "mdp": SimpleNamespace(GaitReward=GaitReward), "moving_command": moving_command}
        exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(ROOT / "rewards.py"), "exec"), namespace)
        term = namespace["CommandedGaitReward"]()
        term.command_name, term.command_threshold = "base_velocity", 0.1
        command = torch.tensor([[0., 0., 0.], [0.2, 0., 0.]])
        env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _: command))
        torch.testing.assert_close(term(env, None, "base_velocity", None, None, 0.1, None, None, None),
                                   torch.tensor([0., 1.]))

    def test_stationary_posture_and_base_velocity_are_masked(self):
        tree = ast.parse((ROOT / "rewards.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("standing_posture", "standing_base_velocity")]
        namespace = {"torch": torch, "SceneEntityCfg": object, "moving_command": moving_command}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / "rewards.py"), "exec"), namespace)
        command = torch.tensor([[0., 0., 0.], [0., 0.05, 0.]])
        data = SimpleNamespace(joint_pos=torch.ones(2, 2), default_joint_pos=torch.zeros(2, 2),
                               root_lin_vel_b=torch.ones(2, 3), root_ang_vel_b=torch.ones(2, 3))
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                              command_manager=SimpleNamespace(get_command=lambda _: command))
        cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])
        torch.testing.assert_close(namespace["standing_posture"](env, "base_velocity", cfg), torch.tensor([1., 0.]))
        torch.testing.assert_close(namespace["standing_base_velocity"](env, "base_velocity", cfg), torch.tensor([6., 0.]))

    def test_turn_support_only_rewards_two_contacts_during_in_place_turn(self):
        tree = ast.parse((ROOT / "rewards.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "turning_two_foot_support")
        namespace = {"torch": torch, "SceneEntityCfg": object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "rewards.py"), "exec"), namespace)
        command = torch.tensor([[0., 0., 0.5], [0., 0., 0.5], [0., 0., 0.], [0.3, 0., 0.5]])
        contact = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.], [1., 1., 0., 0.], [1., 1., 0., 0.]])
        sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=contact))
        robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=torch.tensor([[0., 0., -1.]] * 4)))
        class Scene(dict):
            pass
        scene = Scene(robot=robot)
        scene.sensors = {"contact": sensor}
        env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda _: command))
        cfg = SimpleNamespace(name="contact", body_ids=[0, 1, 2, 3])
        result = namespace["turning_two_foot_support"](env, "base_velocity", cfg, 0.2, 0.05)
        torch.testing.assert_close(result, torch.tensor([1., 0., 0., 0.]))


if __name__ == "__main__":
    unittest.main()
