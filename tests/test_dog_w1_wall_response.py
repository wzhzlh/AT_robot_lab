"""Contact latch, staged response and reward anti-exploit numerical tests."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from test_dog_w1_gait import load_namespace, make_env, torch, Manager, PARAMS

TASK = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1"


def response_namespace():
    ns = load_namespace()
    if isinstance(torch, SimpleNamespace):
        torch.exp = np.exp
        torch.cross = lambda a, b, dim: np.cross(a, b, axis=dim)
        torch.linalg.vector_norm = lambda v, dim, keepdim=False: np.linalg.norm(v, axis=dim, keepdims=keepdim).view(type(v))
    tree = ast.parse((TASK / "wall_response.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(tree, "wall_response.py", "exec"), ns)
    return ns


class WallResponseTest(unittest.TestCase):
    def setUp(self):
        self.ns = response_namespace()
        self.env = make_env()
        self.env.scene["robot"].data.root_pos_w = torch.tensor([[.7,0.,.35],[.7,0.,.35]])
        data = self.env.scene["robot"].data
        data.body_pos_w = torch.tensor([[[.9,.2,.08],[.9,-.2,.08],[.4,.2,.08],[.4,-.2,.08]]]*2)
        data.body_com_pos_w = data.body_pos_w.clone()
        data.body_lin_vel_w = torch.zeros(2,4,3)
        data.body_ang_vel_w = torch.zeros(2,4,3)
        data.root_ang_vel_b = torch.zeros(2,3)
        data.joint_pos = torch.zeros(2,4)
        data.default_joint_pos = torch.zeros(2,4)
        for name in ("FL", "FR", "RL", "RR", "base"):
            self.env.scene.sensors[name] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(2,1,1,3)))
        cfg = SimpleNamespace(body_ids=[0,1,2,3], joint_ids=[0,1,2,3], name="contact_forces")
        self.params = dict(feet_cfg=cfg, hip_cfg=cfg, contact_cfg=cfg,
            wheel_sensors=["FL","FR","RL","RR"], trigger_sensors=["FL","FR","base"], base_sensor="base",
            wall_x=1., wall_height=.3, wall_thickness=.08, wall_length=1.8, foot_radius=.0765,
            clearance_margin=.03, contact_threshold=5., response_timeout=6., max_base_contact_time=.5,
            base_margin=.35, foot_margin=.08, lane_half_width=.55, near_margin=.2, target_speed=.4,
            hip_outward_signs=[1.,-1.,-1.,1.], hip_target=.15, hip_sigma=.12, lift_sigma=.06,
            wheel_speed_sigma=.2, collision_force_cap=50., **PARAMS, phase_offset=.5, stance_ratio=.55)
        self.state = self.ns["WallContactState"](SimpleNamespace(params=self.params), self.env)

    def step(self):
        self.env.common_step_counter += 1
        self.state.evaluate(self.env)

    def test_ground_contact_does_not_trigger_wall_and_latch_survives_contact_loss(self):
        self.env.scene.sensors["contact_forces"].data.current_contact_time[:] = .1
        self.step()
        np.testing.assert_array_equal(np.asarray(self.state.stage), [0,0])
        self.env.scene.sensors["FL"].data.force_matrix_w[0,0,0,0] = -10.
        self.step()
        np.testing.assert_array_equal(np.asarray(self.state.stage), [1,0])
        self.env.scene.sensors["FL"].data.force_matrix_w[:] = 0.
        self.step()
        np.testing.assert_array_equal(np.asarray(self.state.stage), [1,0])
        self.assertEqual(float(self.state.values["wheel"][0]), 0.)
        self.assertEqual(float(self.state.values["hip"][0]), 0.)  # no forward motion

    def test_front_then_hind_crossings_pay_only_once_and_timeout(self):
        self.env.scene.sensors["FR"].data.force_matrix_w[:,0,0,0] = -10.
        self.step()
        feet = self.env.scene["robot"].data.body_pos_w
        feet[:,:2,0] = 1.1
        feet[:,:2,2] = .42
        self.step()
        np.testing.assert_array_equal(np.asarray(self.state.stage), [2,2])
        np.testing.assert_allclose(np.asarray(self.state.values["clear"]), [100.,100.])
        self.step()
        np.testing.assert_array_equal(np.asarray(self.state.values["clear"]), [0.,0.])
        self.state.elapsed[:] = 6.
        self.step()
        np.testing.assert_array_equal(np.asarray(self.state.stage), [4,4])
        self.state.reset([0])
        self.assertEqual(int(self.state.stage[0]), 0)
        self.assertFalse(bool(self.state.crossed[0].any()))
        self.assertEqual(int(self.state.stage[1]), 4)

    def test_bounded_hip_and_lift_require_progress_and_correct_stage(self):
        self.env.scene.sensors["FL"].data.force_matrix_w[:,0,0,0] = -10.
        self.step()
        data = self.env.scene["robot"].data
        data.joint_pos[:] = torch.tensor([[.15,-.15,-.15,.15]]*2)
        data.root_pos_w[:,0] += .01
        data.body_pos_w[:,:2,0] += .01
        data.body_pos_w[:,:2,2] = .4065
        self.step()
        self.assertGreater(float(self.state.values["hip"][0]),.99)
        self.assertGreater(float(self.state.values["lift"][0]),0.)
        data.root_pos_w[:,0] += .01
        data.joint_pos[:] = 2.
        self.step()
        self.assertLess(float(self.state.values["hip"][0]),.01)

    def test_contact_force_filter_missing_fails_loudly(self):
        sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w=None))
        with self.assertRaises(RuntimeError):
            self.ns["filtered_force"](sensor)

    def test_normal_tangent_slip_on_ground_and_vertical_wall(self):
        pos = torch.zeros(1,2,3)
        normal = torch.tensor([[[0.,0.,1.],[-1.,0.,0.]]])
        omega = torch.tensor([[[0.,10.,0.],[0.,10.,0.]]])
        vel = torch.tensor([[[.8,0.,0.],[0.,0.,.8]]])
        result = self.ns["contact_tangent_velocity"](vel, omega, pos, pos, normal, .08)
        np.testing.assert_allclose(np.asarray(result),np.zeros((1,2,3)),atol=1e-6)

    def test_base_pushing_fails_after_budget_not_first_touch(self):
        self.env.scene.sensors["base"].data.force_matrix_w[:,0,0,0] = -10.
        self.step()
        self.assertEqual(int(self.state.stage[0]),1)
        for _ in range(27):self.step()
        self.assertEqual(int(self.state.stage[0]),4)

    def test_one_update_per_step_and_clearance_without_height_not_crossing(self):
        self.env.scene.sensors["FL"].data.force_matrix_w[:,0,0,0] = -10.
        self.step()
        before = np.asarray(self.state.elapsed).copy()
        self.state.evaluate(self.env)
        np.testing.assert_array_equal(np.asarray(self.state.elapsed),np.asarray(before))
        self.env.scene["robot"].data.body_pos_w[:,:2,0] = 1.2
        self.step()
        self.assertEqual(int(self.state.stage[0]),1)
        np.testing.assert_array_equal(np.asarray(self.state.values["clear"]),[0.,0.])


if __name__=='__main__': unittest.main()
