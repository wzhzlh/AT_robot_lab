"""HIM training/inference/export scale and history checks without Isaac Sim."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest

import torch

HIM = Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/him'
PACKAGE = 'him_policy_consistency'
package = ModuleType(PACKAGE)
package.__path__ = [str(HIM)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f'{PACKAGE}.{name}', HIM / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Actor = load('him_actor_critic').HIMActorCritic
Exporter = load('exporter')._HIMPolicyExporter


def ast_classes(name):
    path = HIM / f'{name}.py'
    tree = ast.parse(path.read_text())
    namespace = dict(torch=torch)
    nodes = ast.parse('from __future__ import annotations').body
    nodes += [n for n in tree.body if isinstance(n, ast.ClassDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


Runner = ast_classes('him_on_policy_runner')['HIMOnPolicyRunner']
Wrapper = ast_classes('him_vec_env_wrapper')['HIMVecEnvWrapper']


class PolicyConsistencyTests(unittest.TestCase):
    def model(self, frame):
        return Actor(num_actor_obs=282, num_critic_obs=50, num_one_step_obs=47, num_actions=12,
                     estimator_vel_slice=(0, 3),
                     estimator_target_slices=((3, 6), (6, 9), (12, 24), (24, 36), (36, 48), (0, 3)),
                     actor_current_frame=frame)

    def test_training_mean_inference_export_and_jit_agree(self):
        torch.manual_seed(42)
        obs = torch.randn(8, 282)
        for frame in ('first', 'last'):
            model = self.model(frame)
            model.train()
            model.act(obs)
            expected = model.action_mean.detach()
            model.eval()
            exporter = Exporter(model)
            with torch.no_grad():
                for actual in (model.act_inference(obs), exporter(obs), torch.jit.script(exporter)(obs)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                current = obs[:, :47] if frame == 'first' else obs[:, -47:]
                velocity, latent = model.estimator(obs)
                torch.testing.assert_close(expected, model.actor(torch.cat((current, velocity, latent), -1)))

    def test_estimated_velocity_keeps_scaled_teacher_units(self):
        model = self.model('last')
        with torch.no_grad():
            for p in model.estimator.encoder.parameters():
                p.zero_()
            # Physical velocity (0.3, -0.2, 0.1), teacher base_lin_vel scale=2.
            model.estimator.encoder[-1].bias[:3].copy_(torch.tensor([.6, -.4, .2]))
            model.estimator.encoder[-1].bias[3] = 1.
        observed = []
        hook = model.actor.register_forward_pre_hook(lambda _, args: observed.append(args[0].detach().clone()))
        obs = torch.randn(4, 282)
        model.act(obs)
        model.act_inference(obs)
        hook.remove()
        for value in observed:
            torch.testing.assert_close(value[:, 47:50], torch.tensor([[.6, -.4, .2]]).expand(4, -1))
        teacher = torch.zeros(4, 50)
        teacher[:, :3] = torch.tensor([.6, -.4, .2])
        velocity_loss, _ = model.estimator.update(obs, teacher, lr=0.)
        self.assertEqual(velocity_loss, 0.)

    def test_term_major_history_reordering_preserves_frame_order(self):
        w = Wrapper.__new__(Wrapper)
        w.policy_term_slices = {'a': slice(0, 6), 'b': slice(6, 9)}
        w.policy_one_step_term_slices = {'a': slice(0, 2), 'b': slice(2, 3)}
        # Three oldest-to-newest frames per observation term.
        raw = torch.tensor([[10., 11., 20., 21., 30., 31., 12., 22., 32.]])
        for order, expected in (
            ('oldest_to_latest', [10., 11., 12., 20., 21., 22., 30., 31., 32.]),
            ('latest_to_oldest', [30., 31., 32., 20., 21., 22., 10., 11., 12.]),
        ):
            w.history_order = order
            index = w._build_policy_history_index()
            torch.testing.assert_close(raw.index_select(-1, index), torch.tensor([expected]))

    def test_rejects_ignored_normalization_and_wrong_actor_frame(self):
        env = NS(history_length=6, history_order='oldest_to_latest')
        base = dict(num_steps_per_env=48, save_interval=100, algorithm={})
        for policy, extra, message in (
            ({'actor_current_frame': 'first'}, {}, 'actor_current_frame'),
            ({'actor_current_frame': 'last', 'actor_obs_normalization': True}, {}, 'normalization'),
            ({'actor_current_frame': 'last'}, {'empirical_normalization': True}, 'normalization'),
            ({'actor_current_frame': 'last', 'history_order': 'latest_to_oldest'}, {}, 'history_order'),
        ):
            with self.assertRaisesRegex(ValueError, message):
                Runner(env, dict(base, policy=policy, **extra))


if __name__ == '__main__':
    unittest.main()
