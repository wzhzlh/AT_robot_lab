"""Numerical checks for the actual dog_w1 clock/reward code without Isaac Sim.

Uses CPU torch when installed, otherwise a small NumPy tensor adapter for the
operations used here; simulator wiring is checked by check_dog_w1_sim.py.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:
    class Tensor(np.ndarray):
        def float(self):
            return self.astype(float)

        def clamp(self, min=None, max=None):
            return np.clip(self, min, max).view(Tensor)

        def clamp_min(self, minimum):
            return self.clamp(min=minimum)

        def square(self):
            return self * self

        def abs(self):
            return np.abs(self)

        def sum(self, dim=None, **kwargs):
            return np.asarray(super().sum(axis=kwargs.pop("axis", dim), keepdims=kwargs.pop("keepdim", False), **kwargs)).view(Tensor)

        def mean(self, dim=None, **kwargs):
            return np.asarray(super().mean(axis=dim, **kwargs)).view(Tensor)

        def all(self, dim=None, **kwargs):
            return np.asarray(super().all(axis=dim, **kwargs)).view(Tensor)

        def detach(self):
            return self

        def clone(self):
            return self.copy()

        def expand_as(self, other):
            return np.broadcast_to(self, other.shape).view(Tensor)

    def tensor(value, **kwargs):
        kwargs.pop("device", None)
        return np.array(value, **kwargs).view(Tensor)

    torch = SimpleNamespace(
        Tensor=Tensor, tensor=tensor, pi=math.pi, long=np.int64, bool=bool,
        zeros=lambda *shape, device=None, dtype=None: np.zeros(shape, dtype=dtype).view(Tensor),
        full=lambda shape, value, device=None, dtype=None: np.full(shape, value, dtype=dtype).view(Tensor),
        ones_like=lambda value, dtype=None: np.ones_like(value, dtype=dtype),
        zeros_like=np.zeros_like, maximum=np.maximum, where=np.where, isfinite=np.isfinite,
        stack=lambda values, dim: np.stack(values, axis=dim).view(Tensor), sin=np.sin, cos=np.cos, sigmoid=lambda value: (1.0 / (1.0 + np.exp(-value))),
        linalg=SimpleNamespace(vector_norm=lambda value, dim: np.linalg.norm(value, axis=dim).view(Tensor)),
    )

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1"


class Manager:
    def __init__(self, cfg, env):
        pass


class Scene(dict):
    pass


def load_namespace():
    ns = {"torch": torch, "math": math, "ManagerTermBase": Manager,
          "SceneEntityCfg": object, "ObservationTermCfg": object, "ManagerBasedRLEnv": object,
          "Sequence": list, "RayCaster": SimpleNamespace(meshes={"ground": object()})}
    for name in ("phase_observations.py", "rewards.py"):
        tree = ast.parse((TASK / name).read_text())
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign))]
        exec(compile(tree, name, "exec"), ns)
    return ns


def make_env():
    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(
        projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]),
        body_pos_w=torch.zeros(2, 4, 3), root_pos_w=torch.tensor([[0., 0., .35], [0., 0., .35]]),
    )))
    scene.env_origins = torch.zeros(2, 3)
    scene.sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(current_contact_time=torch.zeros(2, 4)))}
    command = torch.tensor([[.4, 0., 0.], [0., 0., .4]])
    env = SimpleNamespace(scene=scene, command=command, num_envs=2, device="cpu", step_dt=.02, common_step_counter=0)
    env.command_manager = SimpleNamespace(get_command=lambda _: env.command)
    return env


PARAMS = dict(command_name="base_velocity", cycle_time=.5, stop_cycle_time=.5, stop_duration=.5,
              translation_threshold=.05, yaw_threshold=.05)


class DogW1GaitTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_namespace()
        self.env = make_env()
        self.sensor = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3],
                                      body_names=self.ns["FOOT_ORDER"])
        self.asset = SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3], body_names=self.ns["FOOT_ORDER"])

    def assertClose(self, actual, expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)

    def test_clock_shared_between_reward_policy_critic_and_partial_reset(self):
        clock_type = self.ns["DogW1Phase"]
        clocks = [clock_type(None, self.env) for _ in range(3)]
        for step in range(3):
            self.env.common_step_counter = step
            values = [clock(self.env, **PARAMS) for clock in clocks]
            self.assertClose(values[0], values[1])
            self.assertClose(values[0], values[2])
            self.assertClose(clocks[0].phase, [.04 * (step + 1)] * 2)
        clocks[1].reset([0])
        self.assertClose(clocks[0].phase, [0., .12])
        self.env.common_step_counter += 1
        clocks[0](self.env, **PARAMS)
        self.assertClose(clocks[0].phase, [.04, .16])
        self.env.command[:] = 0.
        self.assertClose(clocks[0](self.env, **PARAMS), [[0., 1.], [0., 1.]])
        self.env.command[:, 0] = .4
        self.env.common_step_counter += 1
        clocks[2](self.env, **PARAMS)
        self.assertClose(clocks[2].phase, [.04, .04])
        with self.assertRaises(ValueError):
            clocks[0](self.env, **{**PARAMS, "cycle_time": .7})

    def test_phase_targets_swap_diagonals_and_have_sine_clearance(self):
        phase = torch.tensor([.275, .775])
        stance, target = self.ns["phase_gait_targets"](phase, .55, .08)
        self.assertClose(stance, [[1, 0, 0, 1], [0, 1, 1, 0]])
        self.assertClose(target, [[0., .08, .08, 0.], [.08, 0., 0., .08]])

    def test_dog2_style_contact_smoothing_and_phase_offset_are_effective(self):
        desired, _ = self.ns["phase_gait_targets"](torch.tensor([.549, .55, .551]), .55, .08,
                                                  phase_offset=.5, contact_smoothing=.07)
        self.assertClose(desired[1, 0], .5)
        self.assertGreater(float(desired[0, 0]), float(desired[2, 0]))
        self.assertLess(float(desired[0, 0] - desired[2, 0]), .1)
        stance, _ = self.ns["phase_gait_targets"](torch.tensor([0.]), .55, .08, phase_offset=.75)
        self.assertClose(stance, [[1, 0, 0, 1]])

    def test_contact_rewards_correct_diagonal_and_persistent_contact(self):
        cfg = SimpleNamespace(params={"sensor_cfg": self.sensor})
        reward = self.ns["PhaseContactReward"](cfg, self.env)
        reward.clock.phase[:] = .275
        reward.clock._last_step[:] = self.env.common_step_counter
        contact = self.env.scene.sensors["contact_forces"].data.current_contact_time
        contact[:] = torch.tensor([[.1, 0., 0., .1], [.1, 0., 0., .1]])
        for _ in range(3):
            self.assertClose(reward(self.env, self.sensor, **PARAMS, stance_ratio=.55), [1., 1.])
        # Same support count, wrong diagonal must not earn a full reward.
        contact[:] = torch.tensor([[0., .1, .1, 0.], [0., .1, .1, 0.]])
        self.assertLess(float(np.asarray(reward(self.env, self.sensor, **PARAMS, stance_ratio=.55)).max()), 1e-4)
        contact[:] = .1
        np.testing.assert_allclose(np.asarray(reward(self.env, self.sensor, **PARAMS, stance_ratio=.55)), [.5, .5], atol=1e-5)
        self.env.command[:] = 0.
        self.assertClose(reward(self.env, self.sensor, **PARAMS, stance_ratio=.55), [0., 0.])

    def test_flat_centered_contact_removes_permanent_ground_support_bonus(self):
        reward = self.ns["PhaseContactReward"](SimpleNamespace(params={"sensor_cfg": self.sensor}), self.env)
        reward.clock.phase[:] = .275
        reward.clock._last_step[:] = self.env.common_step_counter
        contact = self.env.scene.sensors["contact_forces"].data.current_contact_time
        contact[:] = .1
        grounded = reward(self.env, self.sensor, **PARAMS, stance_ratio=.55, centered=True)
        self.assertLess(float(np.asarray(grounded).max()), .01)
        contact[:] = torch.tensor([[.1, 0., 0., .1], [.1, 0., 0., .1]])
        correct = reward(self.env, self.sensor, **PARAMS, stance_ratio=.55, centered=True)
        self.assertGreater(float(np.asarray(correct).min()), .99)
        contact[:] = torch.tensor([[0., .1, .1, 0.], [0., .1, .1, 0.]])
        wrong = reward(self.env, self.sensor, **PARAMS, stance_ratio=.55, centered=True)
        self.assertLess(float(np.asarray(wrong).max()), -.99)

    def test_dense_lateral_error_is_independent_of_forward_error(self):
        data = self.env.scene["robot"].data
        data.root_lin_vel_b = torch.tensor([[10., 0., 0.], [0., 0., 0.]])
        self.env.command = torch.tensor([[0., 1., 0.], [0., -1., 0.]])
        error = self.ns["lateral_velocity_error"]
        self.assertClose(error(self.env, "base_velocity", self.asset), [1., 1.])
        data.root_lin_vel_b[:, 1] = torch.tensor([.5, -.5])
        self.assertClose(error(self.env, "base_velocity", self.asset), [.25, .25])
        data.root_lin_vel_b[:, 1] = self.env.command[:, 1]
        self.assertClose(error(self.env, "base_velocity", self.asset), [0., 0.])

    def test_standing_contact_counts_continuous_support_and_turning_is_moving(self):
        contact = self.env.scene.sensors["contact_forces"].data.current_contact_time
        contact[:] = .1
        self.env.command[:] = 0.
        reward = self.ns["standing_contact"]
        for _ in range(3):
            self.assertClose(reward(self.env, "base_velocity", self.sensor), [4., 4.])
            contact[:] += .02
        self.env.command[:, 2] = .1
        self.assertClose(reward(self.env, "base_velocity", self.sensor), [0., 0.])

    def test_swing_reward_uses_wheel_radius_and_each_wheels_terrain_height(self):
        terrain = SimpleNamespace(name="height_scanner_base")
        self.env.scene.sensors[terrain.name] = SimpleNamespace(cfg=SimpleNamespace(mesh_prim_paths=["ground"]))
        cfg = SimpleNamespace(params={"asset_cfg": self.asset, "terrain_sensor_cfg": terrain})
        reward = self.ns["PhaseSwingClearance"](cfg, self.env)
        reward.clock.phase[:] = .275
        reward.clock._last_step[:] = self.env.common_step_counter
        ground = torch.tensor([[0., .1, .2, .3], [0., .1, .2, .3]])
        _, target = self.ns["phase_gait_targets"](reward.clock.phase, .55, .08)
        self.env.scene["robot"].data.body_pos_w[:, :, 2] = ground + .08 + target

        def raycast(starts, directions, **kwargs):
            hits, normals = starts.clone(), torch.zeros_like(starts)
            hits[:, :, 2] = ground
            normals[:, :, 2] = 1.
            return hits, None, normals, None

        self.ns["raycast_mesh"] = raycast
        kwargs = dict(**PARAMS, stance_ratio=.55, swing_height=.08, foot_radius=.08, terrain_sensor_cfg=terrain)
        self.assertClose(reward(self.env, self.asset, **kwargs), [0., 0.])
        self.env.scene["robot"].data.body_pos_w[:, [1, 2], 2] -= .08
        self.assertClose(reward(self.env, self.asset, **kwargs), [.08**2, .08**2])
        self.env.command[:] = 0.
        self.assertClose(reward(self.env, self.asset, **kwargs), [0., 0.])

    def test_turn_height_bonus_only_increases_turning_wheels_lift(self):
        reward = self.ns["PhaseSwingClearance"](SimpleNamespace(params={"asset_cfg": self.asset}), self.env)
        reward.clock.phase[:] = .275
        reward.clock._last_step[:] = self.env.common_step_counter
        positions = self.env.scene["robot"].data.body_pos_w
        positions[:, :, 2] = .08
        positions[:, [1, 2], 2] += .08
        kwargs = dict(**PARAMS, stance_ratio=.55, swing_height=.08, foot_radius=.08)
        self.assertClose(reward(self.env, self.asset, **kwargs), [0., 0.])
        self.assertClose(reward(self.env, self.asset, **kwargs, turn_height_bonus=.02), [0., .02**2])

    def test_standing_recovery_terms_stop_penalizing_after_motion_command(self):
        data = self.env.scene["robot"].data
        data.joint_pos = torch.tensor([[.2]*4, [.2]*4])
        data.default_joint_pos = torch.zeros(2, 4)
        data.joint_vel = torch.tensor([[.5]*4, [.5]*4])
        data.root_lin_vel_b = torch.tensor([[.2, 0., 0.], [.2, 0., 0.]])
        data.root_ang_vel_b = torch.tensor([[0., .3, 0.], [0., .3, 0.]])
        self.asset.joint_ids = [0, 1, 2, 3]
        self.env.command[:] = 0.
        for name, expected in (("standing_posture", .04), ("standing_joint_velocity", .25),
                               ("standing_base_velocity", .13)):
            self.assertClose(self.ns[name](self.env, "base_velocity", self.asset), [expected]*2)
        self.env.command[:, 2] = .2
        for name in ("standing_posture", "standing_joint_velocity", "standing_base_velocity"):
            self.assertClose(self.ns[name](self.env, "base_velocity", self.asset), [0., 0.])

    def test_invalid_terrain_rays_are_masked(self):
        terrain = SimpleNamespace(name="height_scanner_base")
        self.env.scene.sensors[terrain.name] = SimpleNamespace(cfg=SimpleNamespace(mesh_prim_paths=["ground"]))
        reward = self.ns["PhaseSwingClearance"](
            SimpleNamespace(params={"asset_cfg": self.asset, "terrain_sensor_cfg": terrain}), self.env,
        )
        reward.clock.phase[:] = .275
        reward.clock._last_step[:] = self.env.common_step_counter

        def raycast(starts, directions, **kwargs):
            hits, normals = starts.clone(), torch.zeros_like(starts)
            hits[:] = float("inf")
            normals[:, :, 2] = 1.
            return hits, None, normals, None

        self.ns["raycast_mesh"] = raycast
        value = reward(self.env, self.asset, **PARAMS, stance_ratio=.55,
                       swing_height=.08, foot_radius=.08, terrain_sensor_cfg=terrain)
        self.assertClose(value, [0., 0.])

    def test_each_task_configures_same_clock_and_its_own_gait_parameters(self):
        # Execute each terrain's configuration functions with simple config objects.
        # This catches bad parameter names and accidental use of old reward terms.
        for terrain in ("flat", "rough", "slope", "sand", "stairs"):
            tree = ast.parse((TASK / f"{terrain}_env_cfg.py").read_text())
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name in (f"{terrain}_configure_task_reward_values", f"{terrain}_configure_rewards",
                                        f"{terrain}_configure_dog_w1_phase_swing_reward")]
            term_names = {n.attr for function in functions for n in ast.walk(function)
                          if isinstance(n, ast.Attribute) and ast.unparse(n.value) in ("cfg.rewards", "rewards")}
            term_names.update(("feet_air_time", "feet_air_time_variance", "feet_gait", "feet_height", "feet_height_body"))
            rewards = SimpleNamespace(**{
                name: SimpleNamespace(weight=0., params={
                    "asset_cfg": SimpleNamespace(), "sensor_cfg": SimpleNamespace(),
                }) for name in term_names
            })
            cfg = SimpleNamespace(
                rewards=rewards, leg_joint_names=["legs"], wheel_joint_names=["wheels"],
                foot_link_name=".*_foot", base_link_name="base",
                observations=SimpleNamespace(policy=SimpleNamespace(phase=SimpleNamespace(params=PARAMS))),
            )
            ns = {**self.ns, "RewTerm": SimpleNamespace, "SceneEntityCfg": lambda name, **kw: SimpleNamespace(name=name, **kw)}
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(TASK / f"{terrain}_env_cfg.py"), "exec"), ns)
            ns[f"{terrain}_configure_rewards"](cfg)
            self.assertIs(rewards.feet_contact.func, self.ns["PhaseContactReward"])
            self.assertGreater(rewards.feet_contact.weight, 0.)
            if terrain == "flat":
                self.assertTrue(rewards.feet_contact.params["centered"])
            if terrain == "flat":
                self.assertLess(rewards.lateral_velocity_error.weight, 0.)
                distance = rewards.feet_distance_y_exp
                self.assertEqual(distance.params["stance_width"], .44)
                self.assertEqual(distance.params["asset_cfg"].body_names, self.ns["FOOT_ORDER"])
                self.assertTrue(distance.params["asset_cfg"].preserve_order)
                self.assertEqual(rewards.standing_base_velocity.weight, -2.)
                self.assertEqual(rewards.standing_posture.weight, -5.)

                self.assertEqual(rewards.feet_slide.params["wheel_radius"],
                                 rewards.phase_swing_clearance.params["foot_radius"])
                self.assertIs(rewards.feet_slide.func, self.ns["wheel_ground_slip"])
                self.assertIs(rewards.wheel_vel_zero_cmd_l2.func, self.ns["stationary_wheel_velocity"])
                self.assertEqual(rewards.wheel_vel_variance.weight, 0.)

            self.assertIs(rewards.feet_contact_without_cmd.func, self.ns["standing_contact"])
            self.assertLess(rewards.phase_swing_clearance.weight, 0.)
            for key in PARAMS:
                self.assertEqual(rewards.feet_contact.params[key], PARAMS[key])
                self.assertEqual(rewards.phase_swing_clearance.params[key], PARAMS[key])
            for key in ("phase_offset", "stance_ratio", "contact_smoothing"):
                self.assertEqual(rewards.feet_contact.params[key], rewards.phase_swing_clearance.params[key])
            reference_keys = {"phase_offset", "stance_ratio", "contact_smoothing", "foot_radius",
                              "swing_height", "turn_height_bonus"}
            self.assertTrue(reference_keys <= rewards.phase_swing_clearance.params.keys())
            self.assertEqual(rewards.feet_contact.params["sensor_cfg"].body_names, self.ns["FOOT_ORDER"])
            for name in ("feet_air_time", "feet_air_time_variance", "feet_gait", "feet_height", "feet_height_body"):
                self.assertEqual(getattr(rewards, name).weight, 0., (terrain, name))
            self.assertEqual(rewards.phase_swing_clearance.params["terrain_sensor_cfg"] is None, terrain == "flat")


if __name__ == "__main__":
    unittest.main()
