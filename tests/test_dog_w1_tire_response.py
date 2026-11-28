"""Numerical regression of the real bounded tire reward (CPU/NumPy, no simulator)."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1'


class T(np.ndarray):
    def float(self): return self.astype(float)
    def clamp(self, min=None, max=None): return np.clip(self, min, max).view(T)
    def clamp_min(self, v): return self.clamp(min=v)
    def square(self): return self * self
    def abs(self): return np.abs(self)
    def sum(self, dim=None, **kw): return np.asarray(np.asarray(self).sum(axis=kw.pop('axis', dim), **kw)).view(T)
    def mean(self, dim=None, **kw): return np.asarray(np.asarray(self).mean(axis=kw.pop('axis', dim), **kw)).view(T)
    def any(self, dim=None, **kw): return np.asarray(np.asarray(self).any(axis=kw.pop('axis', dim), **kw)).view(T)
    def argmin(self, dim=None, **kw): return np.asarray(np.asarray(self).argmin(axis=kw.pop('axis', dim), **kw)).view(T)
    def new_tensor(self, v): return tensor(v)
    def flatten(self, start_dim, end_dim):
        return self.reshape(*self.shape[:start_dim], -1, *self.shape[end_dim + 1:])
    def copy_(self, value): self[:] = value


def tensor(v, **kw):
    kw.pop('device', None)
    return np.array(v, **kw).view(T)


def wrap(fn):
    return lambda *a, **kw: np.asarray(fn(*a, **kw)).view(T)


def quat_apply(q, v):
    return v + 2 * np.cross(q[..., 1:], np.cross(q[..., 1:], v) + q[..., :1] * v)


torch = NS(as_tensor=tensor, float=float, bool=bool, long=np.int64,
           zeros=lambda *s, **kw: tensor(np.zeros(s), **kw),
           full=lambda s, v, **kw: tensor(np.full(s, v), **kw),
           arange=lambda n, **kw: tensor(np.arange(n), **kw),
           zeros_like=wrap(np.zeros_like), full_like=wrap(np.full_like),
           minimum=wrap(np.minimum), maximum=wrap(np.maximum), where=wrap(np.where),
           cos=wrap(np.cos), sin=wrap(np.sin), exp=wrap(np.exp),
           stack=lambda vs, dim: tensor(np.stack(vs, axis=dim)),
           linalg=NS(vector_norm=lambda v, dim: tensor(np.linalg.norm(v, axis=dim))))


class Clock:
    def __init__(self, cfg, env): self.phase = tensor([.75])
    def __call__(self, *a): pass
    def reset(self, ids=None): self.phase[:] = 0


ns = dict(torch=torch, math=math, quat_apply=quat_apply, DogW1Phase=Clock,
          ManagerTermBase=type('Base', (), {'__init__': lambda self, cfg, env: None}))
for file, names in [('rewards.py', {'phase_gait_targets'}),
                    ('tires_traversal.py', {'tire_approach_mask', 'TireObstacleClimbReward'})]:
    nodes = [n for n in ast.parse((PATH / file).read_text()).body
             if isinstance(n, ast.Assign) and file == 'tires_traversal.py'
             or isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), file, 'exec'), ns)


class Scene(dict): pass


class ResponseTest(unittest.TestCase):
    def setUp(self):
        bodies = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self.p = dict(asset_cfg=NS(name='robot', body_names=bodies, body_ids=[0, 1, 2, 3], preserve_order=True),
                      joint_cfg=NS(joint_names=[b + '_joint' for b in bodies], joint_ids=[0, 1, 2, 3], preserve_order=True),
                      sensor_cfg=NS(name='contact', body_names=bodies, body_ids=[0, 1, 2, 3], preserve_order=True),
                      command_name='base_velocity', cycle_time=.5, stop_cycle_time=.5, stop_duration=.5,
                      translation_threshold=.05, yaw_threshold=.05, stance_ratio=.55,
                      speed_reference=.2 / .0765, response_duration=.12, reward_budget=.025)
        pos = tensor([[[-.675, -.275, .2515], [3., 3., .1], [3., 3., .1], [3., 3., .1]]])
        self.data = NS(body_pos_w=pos, heading_w=tensor([0.]), root_pos_w=tensor([[0., 0., .3]]),
                       root_lin_vel_w=tensor([[.2, 0., 0.]]), body_lin_vel_w=tensor([[[.2, 0., 0.]] * 4]),
                       body_quat_w=tensor([[[2**-.5, -2**-.5, 0., 0.]] * 4]),
                       joint_vel=tensor([[.2 / .0765] * 4]))
        self.contact = NS(current_contact_time=tensor([[1.] * 4]), net_forces_w=tensor([[[0., 0., 10.]] * 4]))
        scene = Scene(robot=NS(data=self.data))
        scene.env_origins = tensor([[0., 0., 0.]])
        scene.terrain = NS(cfg=NS(terrain_generator=NS(group_centers=[[[[0., 0.]]]])),
                           terrain_levels=tensor([0], dtype=int), terrain_types=tensor([0], dtype=int),
                           env_origins=scene.env_origins)
        scene.sensors = {'contact': NS(data=self.contact)}
        self.command = tensor([[.2, 0., 0.]])
        self.env = NS(scene=scene, num_envs=1, device='cpu', step_dt=.02,
                      command_manager=NS(get_command=lambda _: self.command))
        self.reward = ns['TireObstacleClimbReward'](NS(params=self.p), self.env)

    def step(self, **extra):
        self.data.root_pos_w[:, 0] += .004
        return float(self.reward(self.env, **(self.p | extra))[0])

    def test_budget_timeout_no_retrigger_and_reset(self):
        total = sum(self.step() * .02 for _ in range(30))
        self.assertGreater(total, 0)
        self.assertLessEqual(total, .025 / 4 + 1.e-9)
        self.assertEqual(self.step(), 0)
        self.reward.reset()
        self.reward.clock.phase[:] = .75
        self.assertGreater(self.step(), 0)

    def test_drive_requires_contact_progress_and_correct_direction(self):
        self.assertEqual(self.step(lift_weight=0), 0)  # first sample has no displacement history
        self.assertGreater(self.step(lift_weight=0), 0)
        self.contact.current_contact_time[:] = 0
        self.assertEqual(self.step(lift_weight=0), 0)
        self.contact.current_contact_time[:] = 1
        self.data.joint_vel *= -1
        self.assertEqual(self.step(lift_weight=0), 0)
        self.data.joint_vel *= -1
        self.data.root_lin_vel_w[:] = 0
        self.assertEqual(self.step(lift_weight=0), 0)

    def test_stance_excess_height_and_crossed_close_reward(self):
        self.reward.clock.phase[:] = .2
        self.assertEqual(self.step(speed_weight=0), 0)
        self.reward.clock.phase[:] = .75
        high = self.step(speed_weight=0)
        self.assertGreater(high, 0)
        self.data.body_pos_w[0, 0, 2] += .3
        self.assertLess(self.step(speed_weight=0), high * .01)
        self.data.body_pos_w[0, 0, 0] = 1.0
        self.assertEqual(self.step(), 0)

    def test_timeout_cannot_be_rearmed_by_leaving_and_returning(self):
        self.p['reward_budget'] = 100.0  # Timeout, rather than budget, must close this window.
        for _ in range(8): self.step(speed_weight=0)
        self.assertEqual(self.step(), 0)
        self.data.body_pos_w[0, 0, 0] = -2.0
        self.step()
        self.data.body_pos_w[0, 0, 0] = -.675
        self.assertEqual(self.step(), 0)

    def test_no_translation_cancels_window(self):
        self.step()
        self.command[:] = 0
        self.assertEqual(self.step(), 0)
        self.command[0, 0] = .2
        self.assertEqual(self.step(), 0)


if __name__ == '__main__': unittest.main()
