"""Regression for per-wheel obstacle gates and wheel-speed pairing."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1'


class TireWheelOrderTest(unittest.TestCase):
    def test_config_pairs_body_and_joint_columns_without_changing_actions(self):
        source = (ROOT / 'tires_env_cfg.py').read_text()
        # The user may disable this optional reward; still validate its editable template.
        start = source.find('    # cfg.rewards.tire_obstacle_climb = RewTerm(')
        if start >= 0:
            stop = source.index('    # )', start) + len('    # )')
            block = source[start:stop]
            source = source[:start] + '\n'.join(
                line.replace('    # ', '    ', 1) for line in block.splitlines()
            ) + source[stop:]
        tree = ast.parse(source)
        reward = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == 'cfg.rewards.tire_obstacle_climb')
        params = next(k.value for k in reward.keywords if k.arg == 'params')
        entries = {ast.literal_eval(k): v for k, v in zip(params.keys, params.values) if k is not None}
        bodies = {k.arg: ast.literal_eval(k.value) for k in entries['asset_cfg'].keywords}
        joints = {k.arg: ast.literal_eval(k.value) for k in entries['joint_cfg'].keywords}
        self.assertEqual(joints['joint_names'], [b + '_joint' for b in bodies['body_names']])
        self.assertTrue(bodies['preserve_order'] and joints['preserve_order'])
        action_order = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                            and ast.unparse(n.targets[0]) == 'wheel_joint_names')
        self.assertEqual(action_order, ['FR_foot_joint', 'FL_foot_joint', 'RR_foot_joint', 'RL_foot_joint'])

    def test_mismatched_configuration_fails_at_initialization(self):
        tree = ast.parse((ROOT / 'tires_traversal.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TireObstacleClimbReward')
        class Base:
            def __init__(self, cfg, env): pass
        ns = {'ManagerTermBase': Base}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'tires_traversal.py', 'exec'), ns)
        bodies = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        cfg = NS(params={'asset_cfg': NS(body_names=bodies, preserve_order=True),
                         'sensor_cfg': NS(body_names=bodies, preserve_order=True),
                         'joint_cfg': NS(joint_names=[b + '_joint' for b in bodies], preserve_order=True)})
        env = NS()
        cfg.params['joint_cfg'].joint_names.reverse()
        with self.assertRaises(ValueError): ns['TireObstacleClimbReward'](cfg, env)


if __name__ == '__main__': unittest.main()
