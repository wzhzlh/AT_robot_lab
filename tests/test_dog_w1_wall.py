"""Wall success, no-bypass, dwell, one-shot reward and reset checks."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from test_dog_w1_gait import torch, Manager, Scene

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1"


class WallTest(unittest.TestCase):
    def setUp(self):
        ns = {"torch": torch, "ManagerTermBase": Manager}
        tree = ast.parse((TASK / 'wall_rewards.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(tree, 'wall_rewards.py', 'exec'), ns)
        self.ns = ns
        self.data = SimpleNamespace(root_pos_w=torch.tensor([[1.5, 0., .35], [1.5, 0., .35]]),
            body_pos_w=torch.tensor([[[1.3, .22, .08]]*4, [[1.3, .22, .08]]*4]),
            projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]), root_ang_vel_b=torch.zeros(2,3), root_lin_vel_b=torch.zeros(2,3))
        scene = Scene(robot=SimpleNamespace(data=self.data));scene.env_origins=torch.zeros(2,3)
        self.env = SimpleNamespace(scene=scene, num_envs=2, device='cpu', common_step_counter=0, step_dt=.02)
        params = dict(asset_cfg=SimpleNamespace(body_ids=[0,1,2,3]), wall_x=1., wall_thickness=.08, wall_length=1.8,
            progress_scale=3., success_bonus=10., progress_limit=.6, base_margin=.35,
            foot_margin=.08, lane_margin=.3, upright_threshold=.94, max_angular_speed=.6,
            max_linear_speed=.8, min_base_height=.22, max_base_height=.55,
            max_foot_height=.18, stable_duration=.3)
        self.params=params;self.reward=ns['WallTraversalReward'](SimpleNamespace(params=params),self.env)

    def test_all_feet_and_lane_and_stable_landing_required(self):
        f=self.ns['crossing_conditions'];args=(1., .08, .9, .35, .08, .3, .94, .6, .22, .55, .18)
        result=f(self.data.root_pos_w,self.data.body_pos_w,self.data.projected_gravity_b[:,2],
                 torch.zeros(2),*args)
        np.testing.assert_array_equal(np.asarray(result),[True,True])
        self.data.body_pos_w[0,3,0]=.9
        self.data.root_pos_w[1,1]=1.
        result=f(self.data.root_pos_w,self.data.body_pos_w,self.data.projected_gravity_b[:,2],torch.zeros(2),*args)
        np.testing.assert_array_equal(np.asarray(result),[False,False])

    def test_success_dwell_one_shot_and_partial_reset(self):
        for step in range(16):
            self.env.common_step_counter=step
            self.reward.evaluate(self.env)
        self.assertTrue(bool(self.reward.success[0]))
        self.env.common_step_counter+=1
        value=self.reward(self.env,**self.params)
        np.testing.assert_allclose(np.asarray(value),[0.,0.])
        self.reward.reset([0])
        self.assertFalse(bool(self.reward.success[0]))
        self.assertTrue(bool(self.reward.success[1]))

    def test_wall_clearance_only_applies_near_obstacle_and_uses_local_origin(self):
        reward = self.ns["wall_clearance_error"]
        self.data.body_pos_w[0, :, 0] = 1.
        self.data.body_pos_w[0, :, 2] = .0765
        cfg = self.params["asset_cfg"]
        result = reward(self.env, cfg, 1., .08, .3, .0765, .12, .03)
        np.testing.assert_allclose(np.asarray(result), [.33**2, 0.])
        self.env.scene.env_origins[:, 0] = 10.
        self.data.root_pos_w[:, 0] += 10.
        self.data.body_pos_w[:, :, 0] += 10.
        translated = reward(self.env, cfg, 1., .08, .3, .0765, .12, .03)
        np.testing.assert_allclose(np.asarray(translated), np.asarray(result))

    def test_progress_does_not_reward_back_and_forth(self):
        self.data.root_pos_w[:,0]=.5
        first=self.reward(self.env,**self.params)
        self.assertGreater(float(first[0]),0.)
        for step,x in [(1,.2),(2,.5)]:
            self.env.common_step_counter=step;self.data.root_pos_w[:,0]=x
            np.testing.assert_allclose(np.asarray(self.reward(self.env,**self.params)),[0.,0.])

    def test_wall_recipe_is_complete_and_parameters_are_local(self):
        from test_dog_w1_gait import load_namespace, PARAMS
        from test_dog_w1_wall_response import response_namespace
        tree = ast.parse((TASK / "wall_env_cfg.py").read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        names = {n.attr for function in functions for n in ast.walk(function)
                 if isinstance(n, ast.Attribute) and ast.unparse(n.value) in ("cfg.rewards", "rewards")}
        rewards = SimpleNamespace(**{name: SimpleNamespace(weight=0., params={
            "asset_cfg": SimpleNamespace(), "sensor_cfg": SimpleNamespace(),
        }) for name in names})
        cfg = SimpleNamespace(rewards=rewards, leg_joint_names=["legs"], wheel_joint_names=["wheels"],
                              base_link_name="base", foot_link_name=".*_foot",
                              wall_x=1., wall_height=.3, wall_thickness=.08, wall_length=1.8,
                              observations=SimpleNamespace(policy=SimpleNamespace(phase=SimpleNamespace(params=PARAMS))))
        ns = {**load_namespace(), **self.ns, **response_namespace(), "RewTerm": SimpleNamespace,
              "SceneEntityCfg": lambda name, **kw: SimpleNamespace(name=name, **kw)}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "wall_env_cfg.py", "exec"), ns)
        ns["wall_configure_rewards"](cfg)
        self.assertEqual(rewards.base_height_l2.weight, 0.)
        self.assertEqual(rewards.feet_slide.weight, 0.)
        self.assertEqual(rewards.phase_swing_clearance.weight, 0.)
        self.assertEqual(rewards.track_lin_vel_xy_exp.weight, 20.)
        self.assertEqual(rewards.feet_contact.weight, 3.)
        self.assertEqual(rewards.wall_task.params["success_bonus"], 10.)
        self.assertEqual(rewards.wall_task.params["stable_duration"], .3)
        # All computation signatures must accept exactly the locally configured parameters.
        for name in ("wall_task", "wall_foot_clearance", "wall_base_clearance"):
            import inspect
            term = getattr(rewards, name)
            inspect.signature(term.func.__call__ if isinstance(term.func, type) else term.func).bind(
                *([None, self.env] if isinstance(term.func, type) else [self.env]), **term.params)
        source = (TASK / "wall_env_cfg.py").read_text()
        self.assertNotIn("flat_configure_rewards", source)
        self.assertIn("self.rewards = ATDogDogW1RewardsCfg()", source)

    def test_registration_all_wall_runners_exist(self):
        src=(TASK/'__init__.py').read_text()
        for stage in ['Wall']:
            self.assertIn(f'RobotLab-Isaac-Velocity-{stage}-ATDog-DogW1-v0',src)
            for module,suffix in [('rsl_rl_him_cfg','HIMRunnerCfg'),('rsl_rl_ppo_cfg','PPORunnerCfg'),('cusrl_ppo_cfg','TrainerCfg')]:
                self.assertIn(f'class ATDogDogW1{stage}{suffix}',(TASK/'agents'/f'{module}.py').read_text())


if __name__=='__main__':unittest.main()
