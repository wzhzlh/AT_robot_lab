"""Exercise the real checkpoint loader on CPU, without launching Isaac Sim."""
import ast
import copy
from collections import deque
import os
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'source/robot_lab/robot_lab/him/him_on_policy_runner.py'
tree = ast.parse(path.read_text())
ns = dict(torch=torch, copy=copy, os=os, re=re, deque=deque)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), str(path), 'exec'), ns)
Runner = ns['HIMOnPolicyRunner']


def contract(names):
    terms = [dict(name=name, dim=width, func=name, scale=1., noise=None, params={}, clip=None)
             for name, width in names]
    offset, slices = 0, {}
    for term in terms:
        slices[term['name']] = [offset, offset + term['dim']]
        offset += term['dim']
    return dict(format_version=2, groups=dict(policy=[dict(name='actor', dim=4)], critic=terms),
                teacher_target_terms=['joint_pos'], teacher_target_slices=[slices['joint_pos']],
                velocity_slice=slices['base_lin_vel'], history_length=6, policy_corruption=True,
                critic_corruption=False)


class Model(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.actor = nn.Linear(4, 2)
        self.critic = nn.Sequential(nn.Linear(width, 5), nn.ELU(), nn.Linear(5, 1))
        self.estimator = nn.Linear(4, 3)
        self.estimator.optimizer = torch.optim.Adam(self.estimator.parameters())
        self.num_one_step_obs = 4
        self.vel_slice = (0, 3)
        self.target_slices = ((3, 5),)


def runner(c):
    r = Runner.__new__(Runner)
    r.device = 'cpu'
    model = Model(sum(term['dim'] for term in c['groups']['critic']))
    r.alg = NS(actor_critic=model, optimizer=torch.optim.Adam(list(model.actor.parameters()) + list(model.critic.parameters())))
    r.env = NS(get_observation_contract=lambda **kwargs: copy.deepcopy(c))
    r.current_learning_iteration = 123
    return r


class CriticCheckpointTests(unittest.TestCase):
    old = [('base_lin_vel', 3), ('joint_pos', 2), ('height_scan', 7)]
    small = [('base_lin_vel', 3), ('joint_pos', 2)]

    def roundtrip(self, old, new):
        src, dst = runner(old), runner(new)
        with torch.no_grad():
            weight = src.alg.actor_critic.critic[0].weight
            weight.copy_(torch.arange(weight.numel()).reshape_as(weight))
        # Populated optimizer state must not survive an input-layout migration.
        sum(p.sum() for group in src.alg.optimizer.param_groups for p in group['params']).backward()
        src.alg.optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'model.pt')
            src.save(filename)
            dst.load(filename)
        return src, dst

    def test_remove_scan_preserves_actor_teacher_and_iteration(self):
        src, dst = self.roundtrip(contract(self.old), contract(self.small))
        torch.testing.assert_close(dst.alg.actor_critic.critic[0].weight, src.alg.actor_critic.critic[0].weight[:, :5])
        torch.testing.assert_close(dst.alg.actor_critic.actor.weight, src.alg.actor_critic.actor.weight)
        torch.testing.assert_close(dst.alg.actor_critic.estimator.weight, src.alg.actor_critic.estimator.weight)
        self.assertEqual(dst.current_learning_iteration, 123)
        self.assertFalse(dst.alg.optimizer.state)

    def test_add_reorder_and_resize_optional_field(self):
        new = [('height_scan', 9), ('joint_pos', 2), ('base_lin_vel', 3)]
        src, dst = self.roundtrip(contract(self.old), contract(new))
        weight = dst.alg.actor_critic.critic[0].weight
        torch.testing.assert_close(weight[:, :9], torch.zeros_like(weight[:, :9]))
        torch.testing.assert_close(weight[:, 9:11], src.alg.actor_critic.critic[0].weight[:, 3:5])
        torch.testing.assert_close(weight[:, 11:14], src.alg.actor_critic.critic[0].weight[:, :3])
        src, dst = self.roundtrip(contract(self.small), contract(self.old))
        self.assertEqual(torch.count_nonzero(dst.alg.actor_critic.critic[0].weight[:, 5:]).item(), 0)

    def test_same_size_reorder_does_not_load_columns_by_position(self):
        src, dst = self.roundtrip(contract(self.small), contract(list(reversed(self.small))))
        torch.testing.assert_close(dst.alg.actor_critic.critic[0].weight[:, :2], src.alg.actor_critic.critic[0].weight[:, 3:5])
        self.assertFalse(dst.alg.optimizer.state)

    def test_optional_semantics_change_zeroes_only_that_field(self):
        old, new = contract(self.old), contract(self.old)
        new['groups']['critic'][-1]['scale'] = 2.
        src, dst = self.roundtrip(old, new)
        torch.testing.assert_close(dst.alg.actor_critic.critic[0].weight[:, :5], src.alg.actor_critic.critic[0].weight[:, :5])
        self.assertEqual(torch.count_nonzero(dst.alg.actor_critic.critic[0].weight[:, 5:]).item(), 0)

    def test_unchanged_contract_restores_optimizer(self):
        _, dst = self.roundtrip(contract(self.old), contract(self.old))
        self.assertTrue(dst.alg.optimizer.state)

    def test_checkpoint_rejects_current_frame_change(self):
        src, dst = runner(contract(self.small)), runner(contract(self.small))
        src.alg.actor_critic.actor_current_frame = 'last'
        dst.alg.actor_critic.actor_current_frame = 'first'
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'model.pt')
            src.save(filename)
            with self.assertRaisesRegex(ValueError, 'actor_current_frame'):
                dst.load(filename, inference=True)

    def test_reject_actor_teacher_and_duplicate_field_changes(self):
        old = contract(self.old)
        for change in ('actor', 'teacher', 'duplicate', 'slices'):
            new = contract(self.small)
            if change == 'actor':
                new['groups']['policy'][0]['dim'] = 5
            elif change == 'teacher':
                new['groups']['critic'][1]['scale'] = 2.
            elif change == 'duplicate':
                new['groups']['critic'].append(copy.deepcopy(new['groups']['critic'][0]))
            else:
                new['velocity_slice'] = [1, 4]
            self.assertFalse(Runner._can_migrate_critic_inputs(old, new))
            with self.assertRaises(ValueError):
                self.roundtrip(old, new)

    def test_phase_migration_combines_with_removed_scan_when_opted_in(self):
        old = contract([*self.old, ('phase', 2)])
        new = contract([*self.small, ('phase', 2)])
        old['critic_corruption'] = True
        old_func = 'robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations.phase'
        new_func = 'robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations.stairs_phase'
        old_phase = dict(name='phase', dim=2, func=old_func, params={'cycle_time': .5}, noise=None)
        new_phase = dict(name='phase', dim=2, func=new_func, params={'command_name': 'base_velocity'}, noise=None)
        old['groups']['policy'].append(copy.deepcopy(old_phase))
        new['groups']['policy'].append(copy.deepcopy(new_phase))
        old['groups']['critic'][-1] = old_phase
        new['groups']['critic'][-1] = new_phase
        for term in old['groups']['critic'][:2]:
            term['noise'] = {'amplitude': .1}
        src, dst = runner(old), runner(new)
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'model.pt')
            src.save(filename)
            with self.assertRaises(ValueError):
                dst.load(filename)
            dst.load(filename, allow_observation_migration=True)
        torch.testing.assert_close(dst.alg.actor_critic.critic[0].weight[:, :5], src.alg.actor_critic.critic[0].weight[:, :5])
        torch.testing.assert_close(dst.alg.actor_critic.critic[0].weight[:, 5:7], src.alg.actor_critic.critic[0].weight[:, 12:14])

    def test_hidden_architecture_changes_are_rejected(self):
        src = runner(contract(self.old))
        state = src.alg.actor_critic.state_dict()
        state['critic.2.weight'] = torch.zeros(1, 9)
        dst = runner(contract(self.small))
        with self.assertRaises(ValueError):
            dst._migrate_critic_input_weights(state, contract(self.old), contract(self.small))

    def test_weights_only_resets_iteration(self):
        src, dst = runner(contract(self.old)), runner(contract(self.small))
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'model.pt')
            src.save(filename)
            dst.load(filename, weights_only=True)
        self.assertEqual(dst.current_learning_iteration, 0)
        self.assertFalse(dst.alg.optimizer.state)

    def test_inference_uses_same_critic_mapping(self):
        src, dst = runner(contract(self.old)), runner(contract(self.small))
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'model.pt')
            src.save(filename)
            dst.load(filename, inference=True)
        torch.testing.assert_close(dst.alg.actor_critic.actor.weight, src.alg.actor_critic.actor.weight)


if __name__ == '__main__':
    unittest.main()
