"""Check device-side log snapshots and one batched aggregation."""
import ast
from collections import deque
from pathlib import Path
import unittest

import torch

path = Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/him/him_on_policy_runner.py'
tree = ast.parse(path.read_text())
namespace = dict(torch=torch, deque=deque)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), str(path), 'exec'), namespace)
Runner = namespace['HIMOnPolicyRunner']


class EpisodeLogTransferTests(unittest.TestCase):
    def check_device(self, device):
        runner = Runner.__new__(Runner)
        original = torch.tensor([1., 3.], device=device)
        logs = []
        runner._append_episode_log(logs, {'log': {
            'Episode_Reward/test': original, 'Metrics/test': 2.,
            'Running/ignored': 100.,
        }})
        self.assertEqual(logs[0]['Episode_Reward/test'].device, original.device)
        original.fill_(100.)
        logs.append({'Episode_Reward/test': torch.tensor(5.), 'Metrics/test': 4.})
        self.assertEqual(dict(runner._aggregate_episode_infos(logs)), {
            'Episode_Reward/test': 3., 'Metrics/test': 3.,
        })

    def test_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_with_mixed_cpu_scalars(self):
        self.check_device('cuda:0')

    def test_curriculum_uses_latest_value_and_is_not_in_episode_average(self):
        logs = []
        Runner._append_episode_log(logs, {'log': {'Curriculum/terrain_levels': torch.tensor(1.)}})
        Runner._append_episode_log(logs, {'log': {'Curriculum/terrain_levels': torch.tensor(3.),
                                                  'Episode_Reward/test': 2.}})
        self.assertEqual(Runner._latest_curriculum_values(logs), {'Curriculum/terrain_levels': 3.})
        self.assertEqual(Runner._episode_metric_batch(logs), {'Episode_Reward/test': (2., 1)})
        self.assertEqual(Runner._latest_curriculum_values([{'Metrics/test': 2.}]), {})

    def test_stair_success_is_weighted_by_eligible_count(self):
        runner = Runner.__new__(Runner)
        runner.episode_metric_window = deque(maxlen=2)
        runner._episode_metric_totals = {}
        self.assertIsNone(runner._weighted_stair_success())
        runner._push_episode_metric_batch({
            'Metrics/base_velocity/stair_success_count': (1., 1),
            'Metrics/base_velocity/stair_eligible_count': (1., 1),
        })
        runner._push_episode_metric_batch({
            'Metrics/base_velocity/stair_success_count': (0., 1),
            'Metrics/base_velocity/stair_eligible_count': (9., 1),
        })
        self.assertAlmostEqual(runner._weighted_stair_success(), 0.1)
        runner._push_episode_metric_batch({
            'Metrics/base_velocity/stair_success_count': (2., 1),
            'Metrics/base_velocity/stair_eligible_count': (2., 1),
        })
        self.assertAlmostEqual(runner._weighted_stair_success(), 2. / 11.)


if __name__ == '__main__':
    unittest.main()
