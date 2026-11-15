"""Run the actual Dog2 reward classes against analytic terrain without Isaac Sim."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'source/robot_lab/robot_lab'
spec = importlib.util.spec_from_file_location('stair_test_gait', BASE / 'gait.py')
gait = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gait
spec.loader.exec_module(gait)


class Manager:
    def __init__(self, cfg, env):
        self._env = env


class Scene(dict):
    @property
    def sensors(self):
        return self


class StairRewardTests(unittest.TestCase):
    def make(self, yaw=0., command=(.5, 0.), invalid=False, descending=False):
        world_direction = torch.tensor([torch.cos(torch.tensor(yaw)) * command[0] - torch.sin(torch.tensor(yaw)) * command[1],
                                        torch.sin(torch.tensor(yaw)) * command[0] + torch.cos(torch.tensor(yaw)) * command[1]])
        world_direction /= world_direction.norm()
        calls = []

        def raycast(starts, rays, **kwargs):
            calls.append(1)
            hits = starts.clone()
            distance = (starts[..., :2] * world_direction).sum(-1)
            low, high = ((.12, 0.) if descending else (0., .12))
            hits[..., 2] = torch.where(distance >= .08, high, low)
            if invalid:
                hits[..., 2] = float('inf')
            normals = torch.zeros_like(hits)
            normals[..., 2] = 1
            return hits, None, normals, None

        namespace = dict(torch=torch, ManagerTermBase=Manager, foot_geometry=gait.foot_geometry,
                         raycast_mesh=raycast, RayCaster=NS(meshes={'terrain': object()}))
        path = BASE / 'tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog2/stairs_rewards.py'
        tree = ast.parse(path.read_text())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), str(path), 'exec'), namespace)
        positions = torch.zeros(1, 4, 3)
        positions[..., 2] = .18
        quaternions = torch.zeros(1, 4, 4)
        quaternions[..., 0] = 1
        data = NS(body_link_pos_w=positions, body_link_quat_w=quaternions,
                  body_link_lin_vel_w=torch.zeros(1, 4, 3), body_link_ang_vel_w=torch.zeros(1, 4, 3),
                  root_pos_w=torch.tensor([[0., 0., .36]]), heading_w=torch.tensor([yaw]))
        sensor = NS(data=NS(net_forces_w=torch.zeros(1, 4, 3), current_contact_time=torch.zeros(1, 4)),
                    event=torch.zeros(1, 4, dtype=torch.bool))
        sensor.compute_first_contact = lambda dt: sensor.event
        clock = NS(active=torch.tensor([True]))
        clock.targets = (torch.zeros(1, 4), torch.ones(1, 4), torch.full((1, 4), .5))
        clock.foot_targets = lambda: clock.targets
        env = NS(device='cpu', num_envs=1, common_step_counter=1, step_dt=.02,
                 scene=Scene(robot=NS(data=data), contact_forces=sensor,
                             height_scanner=NS(cfg=NS(mesh_prim_paths=['terrain']), data=NS(ray_hits_w=torch.zeros(1, 9, 3)))),
                 command_manager=NS(get_term=lambda name: NS(gait_clock=clock),
                                    get_command=lambda name: torch.tensor([[*command, 0.]])))
        params = dict(asset_cfg=NS(name='robot', body_ids=[0, 1, 2, 3]),
                      sensor_cfg=NS(name='contact_forces', body_ids=[0, 1, 2, 3]),
                      terrain_sensor_cfg=NS(name='height_scanner'), command_name='base_velocity',
                      foot_offsets=((0., 0., 0.),) * 4)
        terms = [namespace[name](NS(params=params), env) for name in
                 ('Dog2DynamicSwing', 'Dog2EdgeClearance', 'Dog2LandingPosition', 'Dog2StanceStability', 'Dog2DynamicBaseHeight')]
        return env, terms, params, calls

    def test_world_geometry_invariant_under_heading_lateral_and_reverse_commands(self):
        reference = None
        for yaw, command in ((0., (.5, 0.)), (1.57, (.5, 0.)), (0., (0., .5)), (0., (-.5, 0.))):
            env, terms, params, calls = self.make(yaw, command)
            result = torch.stack([term(env, **params) for term in terms])
            if reference is None:
                reference = result
            torch.testing.assert_close(result, reference)
            self.assertEqual(len(calls), 1)
            g = terms[0].geometry
            self.assertTrue(g.locked.all())
            torch.testing.assert_close(g.top, torch.full((1, 4), .12))

    def test_landing_only_on_contact_transition_and_target_survives_phase_change(self):
        env, terms, params, _ = self.make()
        landing = terms[2]
        self.assertEqual(landing(env, **params).item(), 0.)
        g = landing.geometry
        target = g.target.clone()
        env.common_step_counter += 1
        env.scene['robot'].data.body_link_pos_w[..., 0] += .15
        clock = env.command_manager.get_term('').gait_clock
        clock.targets = (torch.ones(1, 4), torch.zeros(1, 4), torch.zeros(1, 4))
        env.scene['contact_forces'].event[:] = True
        self.assertGreater(landing(env, **params).item(), 0.)
        torch.testing.assert_close(g.target, target)
        env.common_step_counter += 1
        env.scene['contact_forces'].event[:] = False
        self.assertEqual(landing(env, **params).item(), 0.)

    def test_reset_discards_previous_episode_target(self):
        env, terms, params, calls = self.make()
        terms[0](env, **params)
        terms[0].reset(torch.tensor([0]))
        self.assertFalse(terms[0].geometry.locked.any())
        terms[0](env, **params)
        self.assertEqual(len(calls), 2)

    def test_invalid_rays_are_finite_and_do_not_create_targets(self):
        env, terms, params, _ = self.make(invalid=True)
        for term in terms:
            self.assertTrue(torch.isfinite(term(env, **params)).all())
        self.assertFalse(terms[0].geometry.locked.any())
        self.assertEqual(terms[0](env, **params).item(), 0.)

    def test_stopped_command_has_no_swing_edge_landing_or_stance_penalty(self):
        env, terms, params, _ = self.make()
        env.command_manager.get_term('').gait_clock.active[:] = False
        env.scene['contact_forces'].event[:] = True
        for term in terms[:4]:
            self.assertEqual(term(env, **params).item(), 0.)

    def test_support_height_follows_actual_load_and_holds_during_flight(self):
        env, terms, params, _ = self.make()
        data = env.scene['robot'].data
        data.body_link_pos_w[0, :2, 0] = .16
        sensor = env.scene['contact_forces']
        sensor.data.current_contact_time[:] = .1
        sensor.data.net_forces_w[0, :, 2] = torch.tensor([90., 90., 10., 10.])
        terms[4](env, **params)
        g = terms[4].geometry
        torch.testing.assert_close(g.reference, torch.tensor([.108]))
        env.common_step_counter += 1
        sensor.data.current_contact_time[:] = 0
        terms[4](env, **params)
        torch.testing.assert_close(g.reference, torch.tensor([.108]))

    def test_reward_does_not_read_unused_dense_scanner(self):
        env, terms, params, _ = self.make()
        class MeshOnlyScanner:
            cfg = env.scene['height_scanner'].cfg
            @property
            def data(self):
                raise AssertionError("Reward triggered the unused dense critic scan")
        terms[0].geometry.scanner = MeshOnlyScanner()
        for term in terms:
            self.assertTrue(torch.isfinite(term(env, **params)).all())

    def test_dynamic_peak_tracks_tread_plus_margin_and_asymmetric_error(self):
        env, terms, params, _ = self.make()
        term = terms[0]
        term(env, **params)
        g = term.geometry
        # For a 12 cm upward step, mid-swing sole target is 16 cm.
        env.scene['robot'].data.body_link_pos_w[..., 2] += .16 - g.sole
        env.common_step_counter += 1
        self.assertAlmostEqual(term(env, **params).item(), 0., places=5)
        env.scene['robot'].data.body_link_pos_w[..., 2] -= .03
        env.common_step_counter += 1
        under = term(env, **params).item()
        env.scene['robot'].data.body_link_pos_w[..., 2] += .06
        env.common_step_counter += 1
        over = term(env, **params).item()
        self.assertAlmostEqual(under, 5 * over, places=5)

    def test_edge_impact_requires_swing_and_edge_proximity(self):
        env, terms, params, _ = self.make()
        term = terms[1]
        term(env, **params)
        env.scene['robot'].data.body_link_pos_w[..., 0] = .06
        env.scene['contact_forces'].data.net_forces_w[..., 0] = 100.
        env.common_step_counter += 1
        self.assertGreater(term(env, **params).item(), 0.)
        clock = env.command_manager.get_term('').gait_clock
        clock.targets = (torch.ones(1, 4), torch.zeros(1, 4), torch.zeros(1, 4))
        env.common_step_counter += 1
        self.assertEqual(term(env, **params).item(), 0.)

    def test_descending_tread_is_detected_without_negative_lift(self):
        env, terms, params, _ = self.make(descending=True)
        self.assertTrue(torch.isfinite(terms[0](env, **params)).all())
        g = terms[0].geometry
        self.assertTrue(g.locked.all())
        torch.testing.assert_close(g.launch, torch.full((1, 4), .12))
        torch.testing.assert_close(g.top, torch.zeros(1, 4))

    def test_late_swing_and_edge_share_a_feasible_safe_height(self):
        env, terms, params, _ = self.make()
        terms[0](env, **params)
        g = terms[0].geometry
        # Late swing while still 2 cm beyond the riser: the old sinusoid
        # requested descent below the edge-clearance safety height.
        env.scene['robot'].data.body_link_pos_w[..., 0] = g.edge[..., 0] + .02
        clock = env.command_manager.get_term('').gait_clock
        clock.targets = (torch.zeros(1, 4), torch.ones(1, 4), torch.full((1, 4), .95))
        env.scene['robot'].data.body_link_pos_w[..., 2] += g.edge_safe_height - g.sole
        env.common_step_counter += 1
        self.assertTrue(g.locked.all())
        self.assertAlmostEqual(terms[0](env, **params).item(), 0., places=6)
        self.assertAlmostEqual(terms[1](env, **params).item(), 0., places=6)


if __name__ == '__main__':
    unittest.main()
